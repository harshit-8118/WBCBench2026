"""
Complete YOLO + DinoBloom WBC Classification Training Script

This script provides end-to-end training pipeline:
1. Extract crops using YOLO11
2. Train DinoBloom classifier on crops
3. Comprehensive debugging and monitoring
4. TensorBoard integration
5. Model evaluation and visualization

Author: WBC Classification Pipeline
Date: 2024
"""

import os
import random
import json
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    """Central configuration for entire pipeline"""
    
    # ========== Paths ==========
    data_root = "wbc-bench-2026"
    phase1_csv = "phase1_label.csv"
    train_csv = "phase2_train.csv"
    eval_csv = "phase2_eval.csv"
    test_csv = "phase2_test.csv"
    
    phase1_img_dir = "phase1"
    train_img_dir = "phase2/train"
    eval_img_dir = "phase2/eval"
    test_img_dir = "phase2/test"
    
    use_phase1 = True
    
    # ========== YOLO Configuration ==========
    yolo_model_path = "models/yolo_trained.pt"
    yolo_conf_threshold = 0.20
    crop_output_dir = "yolo_crops"
    use_cached_crops = True
    min_crop_size = 80
    max_crop_size = 400
    min_aspect_ratio = 0.3
    max_aspect_ratio = 3.0
    
    # ========== Model Configuration ==========
    model_type = "dinobloom"
    dinobloom_name = "dinov2_vits14"
    dinobloom_weights = "/data/data/WBCBench/models/dinobloom-s.pth"
    
    extract_cls_token = True
    extract_patch_tokens = True
    use_cls_token = False
    
    # CHANGED: Allow gradient flow in few layers
    freeze_backbone = False
    freeze_few_layers = 5  # CHANGED: Increased from 0 to allow 5 layers to train
    
    # ========== Training Configuration ==========
    image_size = 224
    batch_size = 64
    gradient_accumulation_steps = 1
    num_epochs = 30
    warmup_epochs = 5
    
    # Learning rates
    classifier_lr = 1e-4
    backbone_lr = 1e-5
    min_lr = 1e-7
    use_llrd = True
    layer_decay = 0.75
    weight_decay = 0.01
    
    # ========== COSINE WEIGHT DECAY ==========
    use_cosine_weight_decay = True
    weight_decay_schedule_type = "cosine"
    
    # ========== Augmentation ==========
    use_mixup = True
    mixup_alpha = 0.30
    use_cutmix = True
    cutmix_alpha = 1.0
    mixup_prob = 0.30
    mixup_off_epoch = 20
    
    # ========== Advanced Augmentation (Regularization) ==========
    use_advanced_augmentation = True  # ENABLED: Use advanced augmentations for regularization
    augmentation_techniques = {
        'gaussian_noise': True,
        'poisson_noise': True,
        'blur': True,
        'elastic_deform': True,
        'grid_distort': True,
        'optical_distort': True,
        'iso_noise': True,         # Add impulse noise (ISO noise)
        'multiplicative_noise': True,  # Add multiplicative noise
        'color_jitter': True,  # NEW
        'random_brightness_contrast': True,  # NEW
        'random_gamma': True,  # NEW
    }
    
    # ========== Loss Configuration ==========
    loss_type = "ldam"
    focal_gamma = 2.0
    label_smoothing = 0.002
    ldam_max_m = 0.7
    ldam_s = 30
    focal_weight = 0.5
    
    # ========== Early Stopping with Macro F1 Patience ==========
    patience_macro_f1 = 10  # CHANGED: Patience based on macro F1 score
    min_macro_f1_improvement = 0.001  # Minimum improvement to reset patience
    
    # ========== Heterogeneous Batching ==========
    use_heterogeneous_batching = False  # ENABLED: Use heterogeneous batching
    hetero_batch_ratios = [0.5, 0.3, 0.2]
    
    # ========== Stability & Debugging ==========
    max_loss_value = 100.0
    grad_clip = 1.0
    check_nan_every = 100
    
    # ========== Normalization ==========
    # CHANGED: Added ImageNet statistics verification
    use_imagenet_norm = True  # Standard ImageNet normalization
    mean = [0.485, 0.456, 0.406]  # ImageNet mean
    std = [0.229, 0.224, 0.225]   # ImageNet std
    # Alternative: compute from data
    compute_norm_from_data = False
    
    # ========== Image Resizing Strategy ==========
    resize_strategy = "aspect_ratio_pad"  # "aspect_ratio_pad", "center_crop", "resnet_crop"
    # aspect_ratio_pad: Pad to square preserving aspect (current)
    # center_crop: Crop center to square
    # resnet_crop: ResNet-style with scale/aspect augmentation
    
    # ========== Other ==========
    num_workers = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = False
    save_dir = "checkpoints_yolo_dinobloom1"
    save_checkpoint_every = 2
    
    use_tta = True
    tta_augments = 5
    optimize_thresholds = True
    
    # ========== Attention Visualization ==========
    save_attention_maps = True  # CHANGED: Save attention maps for analysis
    attention_output_dir = "attention_visualizations"
    
    class_names = ['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 
                   'PC', 'PLY', 'PMY', 'SNE', 'VLY']
    num_classes = 13
    pooling_type = 'attention'

# =============================================================================
# UTILITIES
# =============================================================================
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

embed_sizes = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536
}

def get_dino_bloom(model_name="dinov2_vitb14", weights_path=None):
    """Load DinoBloom model with pretrained weights"""
    print(f"Loading {model_name} from torch hub...")
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    
    if weights_path and os.path.exists(weights_path):
        print(f"Loading DinoBloom weights from {weights_path}")
        try:
            pretrained = torch.load(weights_path, map_location=torch.device('cpu'))
            new_state_dict = {}
            if 'teacher' in pretrained:
                state_dict = pretrained['teacher']
            else:
                state_dict = pretrained
                
            for key, value in state_dict.items():
                if 'dino_head' in key or "ibot_head" in key:
                    continue
                new_key = key.replace('backbone.', '')
                new_state_dict[new_key] = value

            if 'pos_embed' in new_state_dict:
                pos_embed_shape = new_state_dict['pos_embed'].shape
                model.pos_embed = nn.Parameter(torch.zeros(pos_embed_shape))
            
            model.load_state_dict(new_state_dict, strict=False)
            print("✓ DinoBloom weights loaded successfully")
        except Exception as e:
            print(f"⚠ Error loading DinoBloom weights: {e}")
            print("Using default DINOv2 weights")
    else:
        print("Using default DINOv2 weights (no custom weights provided)")
            
    return model

# =============================================================================
# IMAGE VALIDATION & PREPROCESSING
# =============================================================================
class ImageValidator:
    """Validate and preprocess images"""
    
    @staticmethod
    def is_valid_image(image, min_size=20):
        """Check if image is valid"""
        if image is None or image.size == 0:
            return False, "None or empty"
        
        if len(image.shape) != 3:
            return False, "Invalid shape"
        
        # Check for black/blank images
        if image.mean() < 5:  # Very dark
            return False, "Too dark"
        
        # Check if mostly one color (corrupted)
        if image.std() < 5:
            return False, "No variation"
        
        # Check size
        if image.shape[0] < min_size or image.shape[1] < min_size:
            return False, "Too small"
        
        return True, "valid"
    
    @staticmethod
    def pad_to_square_with_aspect(image, target_size=224, pad_value=255):
        """Pad image to square while preserving aspect ratio"""
        h, w = image.shape[:2]
        
        # Calculate new dimensions preserving aspect ratio
        if h > w:
            new_w = int(w * target_size / h)
            new_h = target_size
        else:
            new_h = int(h * target_size / w)
            new_w = target_size
        
        # Resize maintaining aspect ratio
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create square canvas
        canvas = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)
        
        # Center the resized image
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    @staticmethod
    def adaptive_crop_enhancement(image):
        """Enhance contrast of crop for better feature extraction"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced

# =============================================================================
# YOLO CROP EXTRACTION
# =============================================================================

class YOLOCropExtractor:
    """Extract and filter the single best WBC crop per image using YOLO11"""
    
    def __init__(self, model_path, config):
        self.config = config
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.validator = ImageValidator()
        self.bad_crops_log = []
        print("✓ YOLO model loaded")
    
    def is_valid_crop(self, bbox):
        """Check if crop meets quality criteria"""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        if w < self.config.min_crop_size or h < self.config.min_crop_size:
            return False, "too_small"
        
        if w > self.config.max_crop_size or h > self.config.max_crop_size:
            return False, "too_large"
        
        aspect = w / max(h, 1)
        if aspect < self.config.min_aspect_ratio or aspect > self.config.max_aspect_ratio:
            return False, "bad_aspect"
        
        return True, "ok"

    def extract_crops(self, csv_path, img_dir, output_dir, split_name):
        """Extract crops with edge padding to avoid missing boundaries"""
        print(f"\n{'='*80}")
        print(f"EXTRACTING BEST CROPS: {split_name}")
        print(f"{'='*80}")
        print(f"Edge padding: ±10 pixels")
        
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(csv_path)
        crop_metadata = []
        self.bad_crops_log = []
        
        stats = {
            'total_images': 0,
            'valid_crops': 0,
            'fallback_full_images': 0,
            'filtered_total': 0,
            'invalid_source_images': 0,
            'enhanced_crops': 0,
            'edge_padded': 0
        }
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            img_name = row['ID']
            label = row.get('labels', None)
            img_path = os.path.join(self.config.data_root, row.get('img_dir', img_dir), img_name)
            
            if not os.path.exists(img_path):
                continue
            
            image = cv2.imread(img_path)
            if image is None:
                stats['invalid_source_images'] += 1
                continue
            
            is_valid, reason = self.validator.is_valid_image(image)
            if not is_valid:
                self.bad_crops_log.append({
                    'image': img_name,
                    'reason': f'source_image_{reason}',
                    'label': label
                })
                stats['invalid_source_images'] += 1
                continue
            
            stats['total_images'] += 1
            results = self.model(img_path, conf=self.config.yolo_conf_threshold, verbose=False)[0]
            detections = sorted(results.boxes, key=lambda x: float(x.conf[0]), reverse=True)
            
            best_crop_found = False
            
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                
                # CHANGED: Add 10px padding on each side to avoid missing edges
                edge_padding = 10
                x1_padded = max(0, x1 - edge_padding)
                y1_padded = max(0, y1 - edge_padding)
                x2_padded = min(image.shape[1], x2 + edge_padding)
                y2_padded = min(image.shape[0], y2 + edge_padding)
                
                bbox = [x1_padded, y1_padded, x2_padded, y2_padded]
                
                is_valid, reason = self.is_valid_crop(bbox)
                
                if is_valid:
                    crop = image[y1_padded:y2_padded, x1_padded:x2_padded].copy()
                    
                    crop_valid, crop_reason = self.validator.is_valid_image(crop)
                    if not crop_valid:
                        self.bad_crops_log.append({
                            'image': img_name,
                            'reason': f'crop_{crop_reason}',
                            'label': label,
                            'bbox': bbox
                        })
                        stats['filtered_total'] += 1
                        continue
                    
                    crop = self.validator.adaptive_crop_enhancement(crop)
                    stats['enhanced_crops'] += 1
                    stats['edge_padded'] += 1  # Count padded crops
                    
                    crop_padded = self.validator.pad_to_square_with_aspect(
                        crop, target_size=self.config.image_size, pad_value=255
                    )
                    
                    clean_name = os.path.splitext(img_name)[0]
                    crop_name = f"{clean_name}_best_crop.jpg"
                    crop_path = os.path.join(output_dir, crop_name)
                    
                    cv2.imwrite(crop_path, crop_padded, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    crop_metadata.append({
                        'crop_path': crop_path,
                        'original_image': img_path,
                        'image_id': img_name,
                        'label': label,
                        'bbox': bbox,
                        'original_bbox': [x1, y1, x2, y2],
                        'edge_padded': True,
                        'confidence': conf,
                        'is_full_image': False,
                        'is_padded': True,
                        'original_crop_size': [x2_padded-x1_padded, y2_padded-y1_padded],
                        'is_enhanced': True
                    })
                    
                    stats['valid_crops'] += 1
                    best_crop_found = True
                    break
                else:
                    stats['filtered_total'] += 1
            
            # FALLBACK: If no valid detections found
            if not best_crop_found:
                stats['fallback_full_images'] += 1
                clean_name = os.path.splitext(img_name)[0]
                
                # Enhance and pad full image
                img_enhanced = self.validator.adaptive_crop_enhancement(image)
                img_padded = self.validator.pad_to_square_with_aspect(
                    img_enhanced, target_size=self.config.image_size, pad_value=255
                )
                
                crop_path = os.path.join(output_dir, f"{clean_name}_full_fallback.jpg")
                cv2.imwrite(crop_path, img_padded, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                crop_metadata.append({
                    'crop_path': crop_path,
                    'original_image': img_path,
                    'image_id': img_name,
                    'label': label,
                    'bbox': [0, 0, image.shape[1], image.shape[0]],
                    'confidence': 0.0,
                    'is_full_image': True,
                    'is_padded': True,
                    'is_enhanced': True
                })
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "crop_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(crop_metadata, f, indent=2)
        
        # Save bad crops log for debugging
        if self.bad_crops_log:
            bad_log_path = os.path.join(output_dir, "bad_crops_log.json")
            with open(bad_log_path, 'w') as f:
                json.dump(self.bad_crops_log, f, indent=2)
        
        self._print_stats(stats, split_name, metadata_path, output_dir)
        return crop_metadata, stats

    def _print_stats(self, stats, split_name, metadata_path, output_dir):
        print(f"\n{'='*80}")
        print(f"EXTRACTION SUMMARY: {split_name}")
        print(f"{'='*80}")
        print(f"Total images processed:     {stats['total_images']}")
        print(f"Invalid source images:      {stats['invalid_source_images']}")
        print(f"Valid targeted WBC crops:   {stats['valid_crops']}")
        print(f"Crops with edge padding:    {stats['edge_padded']}")  # NEW
        print(f"Crops enhanced (CLAHE):     {stats['enhanced_crops']}")
        print(f"Fallback full images:       {stats['fallback_full_images']}")
        print(f"Total invalid boxes:        {stats['filtered_total']}")
        print(f"Metadata saved to:          {metadata_path}")
        if self.bad_crops_log:
            print(f"Bad crops log:              {os.path.join(output_dir, 'bad_crops_log.json')}")
            print(f"Bad crops count:            {len(self.bad_crops_log)}")

# =============================================================================
# TRANSFORMS
# =============================================================================
def get_train_transforms():
    """Training augmentations with optional advanced techniques"""
    transforms_list = [
        A.Resize(Config.image_size, Config.image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.2),
    ]
    # ADVANCED: Add more regularization augmentations
    if Config.use_advanced_augmentation:
        if Config.augmentation_techniques.get('color_jitter'):
            transforms_list.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3))
        if Config.augmentation_techniques.get('random_brightness_contrast'):
            transforms_list.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3))
        if Config.augmentation_techniques.get('random_gamma'):
            transforms_list.append(A.RandomGamma(gamma_limit=(80, 120), p=0.3))
        if Config.augmentation_techniques.get('gaussian_noise'):
            transforms_list.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.4))
        if Config.augmentation_techniques.get('blur'):
            transforms_list.append(A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
            ], p=0.3))
        if Config.augmentation_techniques.get('elastic_deform'):
            transforms_list.append(A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2))
        if Config.augmentation_techniques.get('grid_distort'):
            transforms_list.append(A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2))
        if Config.augmentation_techniques.get('optical_distort'):
            transforms_list.append(A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.2))
    
    
    transforms_list.extend([
        A.Normalize(mean=Config.mean, std=Config.std),
        ToTensorV2()
    ])
    return A.Compose(transforms_list)

def get_valid_transforms():
    """Validation transforms - no augmentation"""
    return A.Compose([
        A.Resize(Config.image_size, Config.image_size),
        A.Normalize(mean=Config.mean, std=Config.std),
        ToTensorV2()
    ])

def get_tta_transforms():
    """TTA transforms"""
    base = [
        A.Resize(Config.image_size, Config.image_size),
        A.Normalize(mean=Config.mean, std=Config.std),
        ToTensorV2()
    ]
    return [
        A.Compose(base),
        A.Compose([A.HorizontalFlip(p=1.0)] + base),
        A.Compose([A.VerticalFlip(p=1.0)] + base),
        A.Compose([A.RandomRotate90(p=1.0)] + base),
        A.Compose([A.RandomResizedCrop(height=Config.image_size, width=Config.image_size, scale=(0.9, 1.0))] + base)
    ]

# =============================================================================
# DATASET
# =============================================================================
class WBCCropDataset(Dataset):
    """Dataset for YOLO-extracted crops"""
    
    def __init__(self, crop_metadata, class_to_idx, transform=None, is_test=False):
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.is_test = is_test
        self.invalid_crops = []  # Track invalid crops
        
        # Filter valid crops with labels
        if is_test:
            self.crop_metadata = crop_metadata
        else:
            self.crop_metadata = [
                c for c in crop_metadata 
                if c.get('label') in class_to_idx and c.get('is_valid', True)
            ]
        
        print(f"Dataset size: {len(self.crop_metadata)} crops")
        if self.invalid_crops:
            print(f"  ⚠ Invalid crops filtered: {len(self.invalid_crops)}")
    
    def __len__(self):
        return len(self.crop_metadata)
    
    def __getitem__(self, idx):
        crop_info = self.crop_metadata[idx]
        crop_path = crop_info['crop_path']
        
        # Verify crop file exists
        if not os.path.exists(crop_path):
            print(f"⚠ Crop file not found: {crop_path}")
            self.invalid_crops.append(crop_path)
            # Return black image as fallback
            image = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)
        else:
            try:
                image = np.array(Image.open(crop_path).convert('RGB'))
                # Validate loaded image
                if image.size == 0:
                    raise ValueError("Empty image")
            except Exception as e:
                print(f"✗ Error loading {crop_path}: {e}")
                self.invalid_crops.append(crop_path)
                image = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)

        if self.transform:
            image = self.transform(image=image)['image']

        if self.is_test:
            return {
                'pixel_values': image,
                'image_id': crop_info.get('image_id', ''),
                'crop_path': crop_path
            }
        
        label = self.class_to_idx[crop_info['label']]
        return {
            'pixel_values': image,
            'label': torch.tensor(label, dtype=torch.long)
        }

# =============================================================================
# MIXUP / CUTMIX
# =============================================================================
def mixup_data(x, y, alpha=0.4):
    """MixUp augmentation"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam

def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, num_classes):
    """MixUp loss"""
    y_a_soft = F.one_hot(y_a, num_classes).float()
    y_b_soft = F.one_hot(y_b, num_classes).float()
    return criterion(pred, lam * y_a_soft + (1 - lam) * y_b_soft)

# =============================================================================
# LOSSES
# =============================================================================
class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        
        if targets.dim() > 1:
            target_dist = targets
        else:
            n_classes = inputs.size(1)
            targets_one_hot = F.one_hot(targets, n_classes).float()
            if self.label_smoothing > 0:
                target_dist = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / n_classes
            else:
                target_dist = targets_one_hot
        
        ce_loss = -torch.sum(target_dist * log_probs, dim=-1)
        pt = torch.sum(target_dist * probs, dim=-1)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None and targets.dim() == 1:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class LDAMLoss(nn.Module):
    """LDAM Loss for long-tailed recognition"""
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list).to(Config.device)
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        target_idx = torch.argmax(target, dim=1) if target.dim() > 1 else target
        index.scatter_(1, target_idx.view(-1, 1), True)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.T).view(-1, 1)
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        
        if target.dim() > 1:
            return torch.sum(-target * F.log_softmax(output * self.s, dim=-1), dim=-1).mean()
        return F.cross_entropy(output * self.s, target, weight=self.weight)

class CombinedFocalLDAMLoss(nn.Module):
    """Combined Focal + LDAM Loss"""
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, 
                 gamma=2.0, focal_weight=0.5, label_smoothing=0.002):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=weight, gamma=gamma, 
                                   label_smoothing=label_smoothing)
        self.ldam_loss = LDAMLoss(cls_num_list, max_m, weight, s)
        self.focal_weight = focal_weight
    
    def forward(self, x, target):
        focal = self.focal_loss(x, target)
        ldam = self.ldam_loss(x, target)
        return self.focal_weight * focal + (1 - self.focal_weight) * ldam

# =============================================================================
# MODEL
# =============================================================================
class AspectRatioResizer:
    """Handle different image aspect ratios intelligently"""
    
    @staticmethod
    def resize_aspect_ratio_pad(image, target_size=224, pad_value=255):
        """Pad to square preserving aspect ratio - CURRENT METHOD"""
        h, w = image.shape[:2]
        
        if h > w:
            new_w = int(w * target_size / h)
            new_h = target_size
        else:
            new_h = int(h * target_size / w)
            new_w = target_size
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)
        
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    @staticmethod
    def resize_center_crop(image, target_size=224):
        """Center crop to square"""
        h, w = image.shape[:2]
        
        # Resize to fit target size while maintaining aspect ratio
        scale = target_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Center crop
        y_start = (new_h - target_size) // 2
        x_start = (new_w - target_size) // 2
        
        return resized[y_start:y_start+target_size, x_start:x_start+target_size]
    
    @staticmethod
    def resize_multi_scale(image, target_size=224):
        """Multi-scale approach: resize to multiple sizes and use random one"""
        scales = [0.9, 1.0, 1.1]
        scale = np.random.choice(scales)
        
        h, w = image.shape[:2]
        new_size = int(target_size * scale)
        
        resized = cv2.resize(image, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
        
        # Pad or crop to target size
        if resized.shape[0] > target_size:
            center = new_size // 2
            start = center - target_size // 2
            return resized[start:start+target_size, start:start+target_size]
        else:
            canvas = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
            offset = (target_size - new_size) // 2
            canvas[offset:offset+new_size, offset:offset+new_size] = resized
            return canvas

class WBCClassifier(nn.Module):
    """DinoBloom-based WBC Classifier with Attention Visualization"""
    
    def __init__(self, model_name, num_classes, dropout=0.3, config=None):
        super().__init__()
        self.config = config
        self.model_type = "dinobloom"
        self.pooling_type = config.pooling_type
        
        self.backbone = get_dino_bloom(config.dinobloom_name, config.dinobloom_weights)
        self.hidden_size = embed_sizes[config.dinobloom_name]
        
        self._apply_freeze_strategy(config)
        
        # Pooling
        self.gem_p = nn.Parameter(torch.ones(1) * 3.0)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(5)])
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 4, 1)
        )
        
        # CHANGED: Class-specific attention for interpretability
        self.class_attention = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_classes)
        )
        
        # Classifier
        classifier_input = self.hidden_size * 2 if self.pooling_type == 'both' else self.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input),
            nn.Linear(classifier_input, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
    def _apply_freeze_strategy(self, config):
        """Freeze backbone layers but allow gradient flow in final layers"""
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ Entire backbone frozen")
        elif config.freeze_few_layers:
            # CHANGED: Freeze N layers but allow last ones to train
            n = config.freeze_few_layers
            if hasattr(self.backbone, 'patch_embed'):
                for p in self.backbone.patch_embed.parameters():
                    p.requires_grad = False
            
            if hasattr(self.backbone, 'blocks'):
                total_blocks = len(self.backbone.blocks)
                # Freeze first (total - n) blocks, unfreeze last n blocks
                for i, block in enumerate(self.backbone.blocks):
                    if i < (total_blocks - n):
                        for p in block.parameters():
                            p.requires_grad = False
            
            print(f"✓ First {total_blocks - n} blocks frozen, last {n} unfrozen")
    
    def get_features(self, pixel_values, return_all_features=False):
        """Extract features from backbone"""
        features_dict = self.backbone.forward_features(pixel_values)
        
        cls_token = features_dict['x_norm_clstoken']
        patch_tokens = features_dict['x_norm_patchtokens']
        
        if self.config.use_cls_token:
            patch_tokens_input = cls_token.unsqueeze(1)
        else:
            patch_tokens_input = patch_tokens
        
        if self.pooling_type == 'gem':
            pooled = self.gem_pooling(patch_tokens_input)
        elif self.pooling_type == 'attention':
            pooled = self.attention_pooling(patch_tokens_input)
        elif self.pooling_type == 'both':
            pooled = torch.cat([self.gem_pooling(patch_tokens_input), 
                               self.attention_pooling(patch_tokens_input)], dim=-1)
        else:
            pooled = patch_tokens_input.mean(dim=1)
        
        if return_all_features:
            return {
                'cls_token': cls_token,
                'patch_tokens': patch_tokens,
                'pooled': pooled,
                'attention_weights': F.softmax(self.attention(patch_tokens_input), dim=1)
            }
        return pooled

    def gem_pooling(self, x, eps=1e-6):
        """Generalized Mean Pooling"""
        p = self.gem_p.clamp(min=1.0)
        return (x.clamp(min=eps).pow(p).mean(dim=1)).pow(1.0 / p)
    
    def attention_pooling(self, hidden_states):
        """Attention-based pooling"""
        weights = F.softmax(self.attention(hidden_states), dim=1)
        return torch.sum(weights * hidden_states, dim=1)
    
    # CHANGED: Get class-specific attention for visualization
    def get_class_attention(self, features):
        """Get attention weights per class"""
        if isinstance(features, dict):
            pooled = features['pooled']
        else:
            pooled = features
        
        return self.class_attention(pooled)  # Shape: (batch, num_classes)
        
    def forward(self, pixel_values, return_features=False):
        """Forward pass"""
        features_all = self.get_features(pixel_values, return_all_features=return_features)
        
        if return_features:
            pooled = features_all['pooled']
        else:
            pooled = features_all
        
        if self.training:
            logits = torch.mean(torch.stack([self.classifier(d(pooled)) 
                                            for d in self.dropouts], dim=0), dim=0)
        else:
            logits = self.classifier(pooled)
        
        if return_features:
            return logits, features_all
        return logits

# =============================================================================
# COSINE WEIGHT DECAY SCHEDULER
# =============================================================================
class CosineWeightDecayScheduler:
    """Schedule weight decay with cosine annealing"""
    def __init__(self, optimizer, base_wd, num_epochs, num_steps_per_epoch, 
                 warmup_epochs=0, schedule_type="cosine"):
        self.optimizer = optimizer
        self.base_wd = base_wd
        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_per_epoch
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type
        self.total_steps = num_epochs * num_steps_per_epoch
        self.warmup_steps = warmup_epochs * num_steps_per_epoch
        self.current_step = 0
    
    def step(self):
        """Update weight decay for all param groups"""
        if self.schedule_type == "cosine":
            wd = self._get_cosine_wd()
        elif self.schedule_type == "linear":
            wd = self._get_linear_wd()
        else:
            wd = self.base_wd
        
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = wd
        
        self.current_step += 1
    
    def _get_cosine_wd(self):
        """Cosine annealing for weight decay"""
        if self.current_step < self.warmup_steps:
            return self.base_wd * (self.current_step / self.warmup_steps)
        
        progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.base_wd * 0.5 * (1.0 + np.cos(np.pi * progress))
    
    def _get_linear_wd(self):
        """Linear decay for weight decay"""
        if self.current_step < self.warmup_steps:
            return self.base_wd * (self.current_step / self.warmup_steps)
        
        progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.base_wd * (1.0 - progress)

# =============================================================================
# HETEROGENEOUS BATCH SAMPLER
# =============================================================================
class HeterogeneousBatchSampler:
    """Sample batches with heterogeneous class distribution"""
    def __init__(self, dataset_metadata, class_counts, class_to_idx, batch_size, 
                 batch_ratios=[0.5, 0.3, 0.2], num_classes=13):
        self.dataset_metadata = dataset_metadata
        self.class_counts = class_counts
        self.class_to_idx = class_to_idx
        self.batch_size = batch_size
        self.batch_ratios = batch_ratios
        self.num_classes = num_classes
        
        # Categorize classes by frequency: head (frequent), medium, tail (rare)
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        n_per_group = len(sorted_classes) // len(batch_ratios)
        
        self.head_classes = set([c[0] for c in sorted_classes[:n_per_group]])
        self.medium_classes = set([c[0] for c in sorted_classes[n_per_group:2*n_per_group]])
        self.tail_classes = set([c[0] for c in sorted_classes[2*n_per_group:]])
        
        # Create indices per group
        self.head_indices = [i for i, m in enumerate(dataset_metadata) 
                            if m.get('label') in self.head_classes]
        self.medium_indices = [i for i, m in enumerate(dataset_metadata) 
                              if m.get('label') in self.medium_classes]
        self.tail_indices = [i for i, m in enumerate(dataset_metadata) 
                            if m.get('label') in self.tail_classes]
        
        print(f"Heterogeneous batching - Head: {len(self.head_indices)}, Medium: {len(self.medium_indices)}, Tail: {len(self.tail_indices)}")
    
    def __iter__(self):
        """Generate heterogeneous batches"""
        head_idx = 0
        medium_idx = 0
        tail_idx = 0
        
        batch_composition = [
            int(self.batch_size * self.batch_ratios[0]),  # head
            int(self.batch_size * self.batch_ratios[1]),  # medium
            int(self.batch_size * self.batch_ratios[2])   # tail
        ]
        
        while True:
            batch = []
            
            # Head samples
            for _ in range(batch_composition[0]):
                if head_idx >= len(self.head_indices):
                    return
                batch.append(self.head_indices[head_idx])
                head_idx += 1
            
            # Medium samples
            for _ in range(batch_composition[1]):
                if medium_idx >= len(self.medium_indices):
                    return
                batch.append(self.medium_indices[medium_idx])
                medium_idx += 1
            
            # Tail samples
            for _ in range(batch_composition[2]):
                if tail_idx >= len(self.tail_indices):
                    return
                batch.append(self.tail_indices[tail_idx])
                tail_idx += 1
            
            if len(batch) == self.batch_size:
                yield batch
    
    def __len__(self):
        total = len(self.head_indices) + len(self.medium_indices) + len(self.tail_indices)
        return total // self.batch_size

# =============================================================================
# TRAINING & VALIDATION
# =============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, 
                   device, epoch, scaler, config, num_classes, writer):
    """Train for one epoch with comprehensive monitoring"""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    all_f1_scores = []  # CHANGED: Track F1 per batch
    total_grad_norm = 0.0
    num_batches = 0
    nan_count = 0
    loss_history = []
    
    use_mixup = config.use_mixup and (epoch < config.mixup_off_epoch)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
    for step, batch in enumerate(pbar):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)
        
        if torch.isnan(pixel_values).any() or torch.isnan(labels.float()).any():
            nan_count += 1
            continue
        
        # MixUp/CutMix
        r = random.random()
        apply_mixup = use_mixup and r < config.mixup_prob
        apply_cutmix = config.use_cutmix and use_mixup and r > config.mixup_prob and r < (config.mixup_prob + 0.25)
        
        with autocast(enabled=config.use_amp):
            if apply_mixup:
                pixel_values, y_a, y_b, lam = mixup_data(pixel_values, labels, config.mixup_alpha)
                logits = model(pixel_values)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam, num_classes)
            elif apply_cutmix:
                pixel_values, y_a, y_b, lam = cutmix_data(pixel_values, labels, config.cutmix_alpha)
                logits = model(pixel_values)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam, num_classes)
            else:
                logits = model(pixel_values)
                
                # DEBUG: Check logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"\n⚠ NaN/Inf in logits at step {step}")
                    print(f"  Logits: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
                    nan_count += 1
                    continue
                
                loss = criterion(logits, labels)
        
        # Check for NaN/Inf BEFORE clipping
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠ NaN/Inf loss at step {step}")
            print(f"  Loss value: {loss.item()}")
            print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
            nan_count += 1
            continue
        
        loss_value = loss.item()
        
        # DEBUG: Check if loss is suspiciously low
        if loss_value < 1e-6 and epoch < 5:
            print(f"\n⚠ Suspiciously low loss at step {step}: {loss_value:.8f}")
            print(f"  Logits std: {logits.std():.6f}, mean: {logits.mean():.6f}")
            print(f"  Labels unique: {torch.unique(labels).cpu().numpy()}")
        
        loss_history.append(loss_value)
        
        # Clip loss
        loss = torch.clamp(loss, max=config.max_loss_value)
        
        if config.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # DEBUG: Check gradients
        if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm == 0:
            print(f"\n⚠ Invalid grad norm at step {step}: {grad_norm}")
            optimizer.zero_grad()
            nan_count += 1
            continue
        
        total_grad_norm += grad_norm.item()
        num_batches += 1
        
        if config.use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad()
        scheduler.step()
        
        running_loss += loss_value
        
        # DEBUG: Check predictions distribution
        preds = torch.argmax(logits, dim=1)
        pred_unique = torch.unique(preds)
        if len(pred_unique) == 1 and epoch == 0 and step % 50 == 0:
            print(f"⚠ Warning: All predictions are class {pred_unique[0].item()} at step {step}")
        
        all_preds.extend(preds.cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{loss_value:.6f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
            'grad': f'{grad_norm:.4f}'
        })
    
    avg_loss = running_loss / max(num_batches, 1)
    avg_grad_norm = total_grad_norm / max(num_batches, 1)
    
    # FIX: Handle NaN in F1 calculation
    if len(set(all_labels)) == 1:
        print("⚠ Warning: All labels are same class!")
        f1 = 0.0
    elif len(set(all_preds)) == 1:
        print(f"⚠ Warning: All predictions are class {all_preds[0]}")
        f1 = 0.0
    else:
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        if np.isnan(f1):
            f1 = 0.0
    
    loss_std = np.std(loss_history) if loss_history else 0
    loss_min = np.min(loss_history) if loss_history else 0
    loss_max = np.max(loss_history) if loss_history else 0
    
    if writer:
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Train/Loss_Min', loss_min, epoch)
        writer.add_scalar('Train/Loss_Max', loss_max, epoch)
        writer.add_scalar('Train/Loss_Std', loss_std, epoch)
        writer.add_scalar('Train/F1', f1, epoch)
        writer.add_scalar('Train/GradNorm', avg_grad_norm, epoch)
        writer.add_scalar('Train/NaN_Count', nan_count, epoch)
        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'Train/LR_group_{i}', param_group['lr'], epoch)
    
    print(f"  Loss Stats - Min: {loss_min:.6f}, Max: {loss_max:.6f}, Std: {loss_std:.6f}")
    print(f"  Macro F1: {f1:.4f}, Unique predictions: {len(set(all_preds))}")
    
    return avg_loss, f1

@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, config, writer=None):
    """Validation with macro F1 patience tracking"""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    loss_history = []
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)
        
        with autocast(enabled=config.use_amp):
            logits = model(pixel_values)
            loss = criterion(logits, labels)
        
        if not torch.isnan(loss) and not torch.isinf(loss):
            loss_value = loss.item()
            running_loss += loss_value
            loss_history.append(loss_value)
        
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().detach().numpy())
    
    avg_loss = running_loss / max(len(dataloader), 1)
    
    if len(set(all_labels)) == 1:
        f1 = 0.0
        bal_acc = 0.0
        per_class_f1 = np.zeros(Config.num_classes)
    elif len(set(all_preds)) == 1:
        f1 = 0.0
        bal_acc = 0.0
        per_class_f1 = np.zeros(Config.num_classes)
    else:
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        
        if np.isnan(f1):
            f1 = 0.0
        if np.isnan(bal_acc):
            bal_acc = 0.0
    
    loss_std = np.std(loss_history) if loss_history else 0
    loss_min = np.min(loss_history) if loss_history else 0
    loss_max = np.max(loss_history) if loss_history else 0
    
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/Loss_Min', loss_min, epoch)
        writer.add_scalar('Val/Loss_Max', loss_max, epoch)
        writer.add_scalar('Val/Loss_Std', loss_std, epoch)
        writer.add_scalar('Val/F1_Macro', f1, epoch)  # CHANGED: Explicit macro F1
        writer.add_scalar('Val/BalancedAccuracy', bal_acc, epoch)
        
        for i, class_name in enumerate(Config.class_names):
            writer.add_scalar(f'Val/F1_{class_name}', per_class_f1[i], epoch)
    
    print(f"  Loss Stats - Min: {loss_min:.6f}, Max: {loss_max:.6f}, Std: {loss_std:.6f}")
    print(f"  Macro F1: {f1:.4f}, Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Per-class F1: {[f'{x:.3f}' for x in per_class_f1[:5]]} ...")
    
    return avg_loss, bal_acc, f1, np.array(all_probs), np.array(all_labels)

# =============================================================================
# UTILITIES
# =============================================================================
def get_class_weights(labels, class_to_idx):
    """Compute class weights for imbalanced dataset"""
    counter = Counter(labels)
    weights = []
    for cls in sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x]):
        count = counter.get(cls, 1)
        weights.append(1 / np.sqrt(count))
    weights = np.array(weights)
    return torch.FloatTensor(weights / weights.sum() * len(weights))

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-7):
    """Cosine learning rate schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def optimize_temperatures(valid_targets, valid_probs, num_classes):
    """Optimize temperature scaling for calibration"""
    print("Optimizing temperature scaling...")
    def objective(temps):
        preds = np.argmax(valid_probs / temps, axis=1)
        return -f1_score(valid_targets, preds, average='macro')
    
    result = minimize(objective, np.ones(num_classes), method='Nelder-Mead', 
                     bounds=[(0.1, 2.0)]*num_classes, options={'maxiter': 100})
    print(f"Optimized F1: {-result.fun:.4f}")
    return result.x.tolist()

def apply_temperatures(probs, temperatures):
    """Apply temperature scaling to probabilities"""
    return np.argmax(probs / np.array(temperatures), axis=1)

def save_checkpoint(model, optimizer, scheduler, epoch, f1, temperatures, save_path):
    """Save model checkpoint"""
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'f1': f1,
        'temperatures': temperatures
    }, save_path)

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================
def main():
    print("="*80)
    print("YOLO + DinoBloom WBC Classification Pipeline")
    print("="*80)
    print(f"Device: {Config.device}")
    print(f"Model: {Config.dinobloom_name}")
    print(f"Image Size: {Config.image_size}")
    print(f"Batch Size: {Config.batch_size} x {Config.gradient_accumulation_steps} = {Config.batch_size * Config.gradient_accumulation_steps}")
    print(f"Epochs: {Config.num_epochs}")
    print("="*80)
    
    # Create directories
    os.makedirs(Config.save_dir, exist_ok=True)
    os.makedirs(Config.crop_output_dir, exist_ok=True)
    
    # TensorBoard
    log_dir = os.path.join(Config.save_dir, "tensorboard_logs")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"\n✓ TensorBoard: {log_dir}")
    print(f"  Run: tensorboard --logdir={log_dir}\n")
    
    # Save config
    config_dict = {k: v for k, v in Config.__dict__.items() 
                  if not k.startswith("__") and not callable(v)}
    config_dict["device"] = str(config_dict["device"])
    with open(os.path.join(Config.save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)
    
    # Class mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(Config.class_names)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    # ========== STEP 1: YOLO CROP EXTRACTION ==========
    print("\n" + "="*80)
    print("STEP 1: YOLO CROP EXTRACTION")
    print("="*80)
    print(f"Data root: {Config.data_root}")
    print(f"Crop output dir: {Config.crop_output_dir}")
    
    extractor = YOLOCropExtractor(Config.yolo_model_path, Config)
    
    # Training crops
    train_crop_dir = os.path.join(Config.crop_output_dir, "train")
    train_csv_path = os.path.join(Config.data_root, Config.train_csv)
    
    print(f"\nTrain CSV: {train_csv_path}")
    print(f"  Exists: {os.path.exists(train_csv_path)}")
    
    if Config.use_cached_crops and os.path.exists(os.path.join(train_crop_dir, "crop_metadata.json")):
        print(f"Loading cached training crops from: {train_crop_dir}")
        with open(os.path.join(train_crop_dir, "crop_metadata.json"), 'r') as f:
            train_crop_metadata = json.load(f)
        print(f"Loaded {len(train_crop_metadata)} training crops from cache")
        
        # Verify sample crops exist
        sample_crops = train_crop_metadata[:3]
        for crop in sample_crops:
            exists = os.path.exists(crop['crop_path'])
            print(f"  Sample: {os.path.basename(crop['crop_path'])} - Exists: {exists}")
    else:
        print(f"Extracting training crops to: {train_crop_dir}")
        train_crop_metadata, _ = extractor.extract_crops(
            train_csv_path,
            Config.train_img_dir,
            train_crop_dir,
            "train"
        )
    
    # Phase1 crops
    if Config.use_phase1:
        phase1_crop_dir = os.path.join(Config.crop_output_dir, "phase1")
        phase1_csv_path = os.path.join(Config.data_root, Config.phase1_csv)
        
        print(f"\nPhase1 CSV: {phase1_csv_path}")
        print(f"  Exists: {os.path.exists(phase1_csv_path)}")
        
        if Config.use_cached_crops and os.path.exists(os.path.join(phase1_crop_dir, "crop_metadata.json")):
            print(f"Loading cached phase1 crops from: {phase1_crop_dir}")
            with open(os.path.join(phase1_crop_dir, "crop_metadata.json"), 'r') as f:
                phase1_crop_metadata = json.load(f)
            print(f"Loaded {len(phase1_crop_metadata)} phase1 crops from cache")
        else:
            print(f"Extracting phase1 crops to: {phase1_crop_dir}")
            phase1_crop_metadata, _ = extractor.extract_crops(
                phase1_csv_path,
                Config.phase1_img_dir,
                phase1_crop_dir,
                "phase1"
            )
        train_crop_metadata.extend(phase1_crop_metadata)
    
    # Validation crops
    eval_crop_dir = os.path.join(Config.crop_output_dir, "eval")
    eval_csv_path = os.path.join(Config.data_root, Config.eval_csv)
    
    print(f"\nEval CSV: {eval_csv_path}")
    print(f"  Exists: {os.path.exists(eval_csv_path)}")
    
    if Config.use_cached_crops and os.path.exists(os.path.join(eval_crop_dir, "crop_metadata.json")):
        print(f"Loading cached eval crops from: {eval_crop_dir}")
        with open(os.path.join(eval_crop_dir, "crop_metadata.json"), 'r') as f:
            eval_crop_metadata = json.load(f)
        print(f"Loaded {len(eval_crop_metadata)} eval crops from cache")
        
        # Verify sample crops exist
        sample_crops = eval_crop_metadata[:3]
        for crop in sample_crops:
            exists = os.path.exists(crop['crop_path'])
            print(f"  Sample: {os.path.basename(crop['crop_path'])} - Exists: {exists}")
    else:
        print(f"Extracting eval crops to: {eval_crop_dir}")
        eval_crop_metadata, _ = extractor.extract_crops(
            eval_csv_path,
            Config.eval_img_dir,
            eval_crop_dir,
            "eval"
        )
    
    print(f"\n✓ Total crops - Train: {len(train_crop_metadata)}, Eval: {len(eval_crop_metadata)}")
    
    # ========== STEP 2: CREATE DATASETS ==========
    print("\n" + "="*80)
    print("STEP 2: CREATING DATASETS")
    print("="*80)
    
    train_dataset = WBCCropDataset(train_crop_metadata, class_to_idx, get_train_transforms())
    eval_dataset = WBCCropDataset(eval_crop_metadata, class_to_idx, get_valid_transforms())
    
    # Compute class distribution from crops
    labels = [c['label'] for c in train_crop_metadata if c.get('label') in class_to_idx]
    class_counts = Counter(labels)
    
    print("\nClass distribution in crops:")
    cls_num_list = []
    for cls in Config.class_names:
        count = class_counts.get(cls, 1)
        cls_num_list.append(count)
        print(f"  {cls}: {count}")
    
    # Dataloaders - HETEROGENEOUS BATCHING OPTION
    if Config.use_heterogeneous_batching:
        print("\nUsing heterogeneous batch sampling...")
        hetero_sampler = HeterogeneousBatchSampler(
            train_crop_metadata, class_counts, class_to_idx, 
            Config.batch_size, Config.hetero_batch_ratios, Config.num_classes
        )
        train_loader = DataLoader(train_dataset, batch_sampler=hetero_sampler, 
                                 num_workers=Config.num_workers, pin_memory=True)
    else:
        print("\nUsing weighted random sampling...")
        weights = [1.0 / np.sqrt(class_counts.get(c['label'], 1)) 
                   for c in train_crop_metadata if c.get('label') in class_to_idx]
        sampler = WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, 
                                 sampler=sampler, num_workers=Config.num_workers, 
                                 drop_last=True, pin_memory=True)
    
    eval_loader = DataLoader(eval_dataset, batch_size=Config.batch_size*2, 
                            shuffle=False, num_workers=Config.num_workers, pin_memory=True)
    
    print(f"\n✓ Dataloaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(eval_loader)}")
    
    # ========== STEP 3: MODEL ==========
    print("\n" + "="*80)
    print("STEP 3: MODEL INITIALIZATION")
    print("="*80)
    
    model = WBCClassifier(Config.model_type, Config.num_classes, config=Config).to(Config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ========== STEP 4: OPTIMIZER & SCHEDULER ==========
    if Config.use_llrd:
        params = [
            {'params': model.backbone.parameters(), 'lr': Config.backbone_lr},
            {'params': model.classifier.parameters(), 'lr': Config.classifier_lr},
            {'params': model.attention.parameters(), 'lr': Config.classifier_lr},
            {'params': [model.gem_p], 'lr': Config.classifier_lr}
        ]
        optimizer = AdamW(params, weight_decay=Config.weight_decay)
        print(f"✓ Layer-wise LR: backbone={Config.backbone_lr:.2e}, head={Config.classifier_lr:.2e}")
    else:
        optimizer = AdamW(model.parameters(), lr=Config.classifier_lr, 
                         weight_decay=Config.weight_decay)
    
    # Loss function - FLEXIBLE OPTIONS
    class_weights = get_class_weights(labels, class_to_idx).to(Config.device)
    
    if Config.loss_type == "ldam":
        criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=Config.ldam_max_m, s=Config.ldam_s)
        print(f"✓ Loss: LDAM (max_m={Config.ldam_max_m}, s={Config.ldam_s})")
    elif Config.loss_type == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=Config.focal_gamma, 
                            label_smoothing=Config.label_smoothing)
        print(f"✓ Loss: Focal (gamma={Config.focal_gamma})")
    elif Config.loss_type == "focal_ldam":
        criterion = CombinedFocalLDAMLoss(cls_num_list=cls_num_list, max_m=Config.ldam_max_m, 
                                         weight=class_weights, s=Config.ldam_s, 
                                         gamma=Config.focal_gamma, 
                                         focal_weight=Config.focal_weight,
                                         label_smoothing=Config.label_smoothing)
        print(f"✓ Loss: Combined Focal+LDAM (focal_weight={Config.focal_weight})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=Config.label_smoothing)
        print(f"✓ Loss: CrossEntropy")
    
    # Scheduler
    steps_per_epoch = len(train_loader)
    total_steps = Config.num_epochs * steps_per_epoch
    warmup_steps = Config.warmup_epochs * steps_per_epoch
    

    
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, Config.min_lr)
    
    # COSINE WEIGHT DECAY SCHEDULER - NEW
    if Config.use_cosine_weight_decay:
        wd_scheduler = CosineWeightDecayScheduler(
            optimizer, Config.weight_decay, Config.num_epochs, 
            steps_per_epoch, Config.warmup_epochs, 
            Config.weight_decay_schedule_type
        )
        print(f"✓ Cosine Weight Decay: type={Config.weight_decay_schedule_type}")
    else:
        wd_scheduler = None
    
    scaler = GradScaler(enabled=Config.use_amp)
    
    print(f"✓ Scheduler: Cosine with warmup ({Config.warmup_epochs} epochs)")
    
    # ========== STEP 5: TRAINING LOOP ==========
    print("\n" + "="*80)
    print("STEP 5: TRAINING")
    print("="*80)
    
    best_f1 = 0.0
    best_temperatures = [1.0] * Config.num_classes
    patience = 0
    patience_macro_f1 = 0  # CHANGED: Patience counter for macro F1
    patience_limit = Config.patience_macro_f1  # CHANGED: Use config patience
    
    for epoch in range(Config.num_epochs):
        # LDAM DRW
        if Config.loss_type in ["ldam", "focal_ldam"] and epoch >= 30:
            idx = min((epoch - 30) // 30, 1)
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], np.array(cls_num_list))
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            criterion.weight = torch.FloatTensor(per_cls_weights).to(Config.device)
        
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            Config.device, epoch, scaler, Config, Config.num_classes, writer
        )
        
        # COSINE WEIGHT DECAY - STEP
        if wd_scheduler:
            wd_scheduler.step()
        
        val_loss, val_bal, val_f1, val_probs, val_labels = validate(
            model, eval_loader, criterion, Config.device, epoch, Config, writer
        )
        
        print(f"\nEpoch {epoch+1}/{Config.num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Macro F1: {val_f1:.4f}, BalAcc: {val_bal:.4f}")
        
        # CHANGED: Patience based on macro F1 improvement
        if val_f1 > best_f1 + Config.min_macro_f1_improvement:
            best_f1 = val_f1
            patience_macro_f1 = 0
            print(f"  ✓ New best Macro F1: {best_f1:.4f} (improvement: +{val_f1-best_f1:.4f})")
            
            if Config.optimize_thresholds:
                best_temperatures = optimize_temperatures(val_labels, val_probs, Config.num_classes)
            
            save_checkpoint(model, optimizer, scheduler, epoch, best_f1, best_temperatures,
                          os.path.join(Config.save_dir, "best_model.pth"))
        else:
            patience_macro_f1 += 1
            print(f"  Patience: {patience_macro_f1}/{patience_limit}")
            if patience_macro_f1 >= patience_limit and epoch > Config.warmup_epochs:
                print(f"  ⚠ Early stopping: Macro F1 not improved for {patience_limit} epochs")
                break
        
        if (epoch + 1) % Config.save_checkpoint_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_f1, best_temperatures,
                          os.path.join(Config.save_dir, f"checkpoint_epoch_{epoch+1}.pth"))
    
    writer.close()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Validation F1: {best_f1:.4f}")
    print(f"Model saved: {Config.save_dir}/best_model.pth")
    
    # ========== STEP 6: TEST PREDICTION ==========
    print("\n" + "="*80)
    print("STEP 6: TEST PREDICTION")
    print("="*80)
    
    # Extract test crops
    test_crop_dir = os.path.join(Config.crop_output_dir, "test")
    test_csv_path = os.path.join(Config.data_root, Config.test_csv)
    
    print(f"Test CSV: {test_csv_path}")
    print(f"  Exists: {os.path.exists(test_csv_path)}")
    
    if Config.use_cached_crops and os.path.exists(os.path.join(test_crop_dir, "crop_metadata.json")):
        print(f"Loading cached test crops from: {test_crop_dir}")
        with open(os.path.join(test_crop_dir, "crop_metadata.json"), 'r') as f:
            test_crop_metadata = json.load(f)
        print(f"Loaded {len(test_crop_metadata)} test crops from cache")
    else:
        print(f"Extracting test crops to: {test_crop_dir}")
        test_crop_metadata, _ = extractor.extract_crops(
            test_csv_path,
            Config.test_img_dir,
            test_crop_dir,
            "test"
        )

    # Load best model
    checkpoint = torch.load(os.path.join(Config.save_dir, "best_model.pth"), weights_only=False)
    model.load_state_dict(checkpoint['model'])
    best_temperatures = checkpoint['temperatures']
    model.eval()
    
    test_dataset = WBCCropDataset(test_crop_metadata, class_to_idx, 
                                  get_valid_transforms(), is_test=True)
    
    # TTA prediction
    if Config.use_tta:
        print("Running TTA prediction...")
        all_probs = []
        image_ids = []
        
        for i, tfs in enumerate(get_tta_transforms()):
            print(f"  TTA {i+1}/{len(get_tta_transforms())}")
            test_dataset.transform = tfs
            loader = DataLoader(test_dataset, batch_size=Config.batch_size*2, 
                              shuffle=False, num_workers=Config.num_workers)
            
            step_probs = []
            ids_list = []
            
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"TTA {i+1}"):
                    if i == 0:
                        ids_list.extend(batch['image_id'])
                    with autocast(enabled=Config.use_amp):
                        logits = model(batch['pixel_values'].to(Config.device))
                        step_probs.append(F.softmax(logits, dim=1).cpu())
            
            all_probs.append(torch.cat(step_probs))
            if i == 0:
                image_ids = ids_list
        
        avg_probs = torch.stack(all_probs).mean(dim=0).numpy()
    else:
        print("Running single prediction...")
        loader = DataLoader(test_dataset, batch_size=Config.batch_size*2, 
                          shuffle=False, num_workers=Config.num_workers)
        
        probs_list, ids_list = [], []
        with torch.no_grad():
            for batch in tqdm(loader):
                ids_list.extend(batch['image_id'])
                logits = model(batch['pixel_values'].to(Config.device))
                probs_list.append(F.softmax(logits, dim=1).cpu())
        
        avg_probs = torch.cat(probs_list).numpy()
        image_ids = ids_list
    
    # Apply temperature scaling
    final_preds = apply_temperatures(avg_probs, best_temperatures)
    
    # Save submission
    submission = pd.DataFrame({
        'ID': image_ids,
        'labels': [idx_to_class[p] for p in final_preds]
    })
    submission_path = os.path.join(Config.save_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    
    print(f"\n✓ Submission saved: {submission_path}")
    print(f"✓ Predictions: {len(submission)}")
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()