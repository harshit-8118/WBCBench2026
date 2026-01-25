import os
import random
import json
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoModel
import timm  # Needed for ConvNeXt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

torch.cuda.empty_cache()
# --------------------------
# 1. Helpers & Original DinoBloom
# --------------------------
embed_sizes = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536
}

def get_dino_bloom(model_name="dinov2_vitl14", weights_path=None):
    """Load DinoBloom model (Preserved)"""
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
                    pass
                else:
                    new_key = key.replace('backbone.', '')
                    new_state_dict[new_key] = value

            if 'pos_embed' in new_state_dict:
                pos_embed_shape = new_state_dict['pos_embed'].shape
                model.pos_embed = nn.Parameter(torch.zeros(pos_embed_shape))
            
            model.load_state_dict(new_state_dict, strict=False)
            print("DinoBloom weights loaded successfully")
        except Exception as e:
            print(f"Error loading DinoBloom weights: {e}")
            print("Using default DINOv2 weights")
            
    return model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --------------------------
# 2. Configuration
# --------------------------
class Config:
    # Paths
    data_root = "/data/data/WBCBench/wbc-bench-2026"
    phase1_csv = "phase1_label.csv"
    train_csv = "phase2_train.csv"
    eval_csv = "phase2_eval.csv"
    test_csv = "phase2_test.csv"
    
    # Image folders
    phase1_img_dir = "phase1"
    train_img_dir = "phase2/train"
    eval_img_dir = "phase2/eval"
    test_img_dir = "phase2/test"
    
    use_phase1 = True
    
    # --- Model Selection ---
    # Options: "medsiglip", "dinobloom", "convnext"
    model_type = "dinobloom" 
    
    # Backbone Names
    medsiglip_name = "google/medsiglip-448"
    dinobloom_name = "dinov2_vits14"
    dinobloom_weights = "data/data/WBCBench/models/dinobloom-s.pth" # s  to l
    convnext_name = "convnextv2_base.fcmae_ft_in22k_in1k_384"
    
    # --- Training ---
    freeze_backbone = False
    freeze_few_layers = 3   #either int or None

    use_cls_token = True
    
    # Image Size
    image_size = 336  

    batch_size = 128
    gradient_accumulation_steps = 1
    effective_batch_size = batch_size * gradient_accumulation_steps
    num_epochs = 50
    warmup_epochs = 5
    
    # --- Learning Rates & Optimizer ---
    classifier_lr = 2e-5
    backbone_lr = 1e-6
    min_lr = 1e-6
    weight_decay = 0.0001
    
    # NEW: Layer-wise Learning Rate Decay (Helpful for Pre-trained models)
    use_llrd = True
    layer_decay = 0.8
    
    # --- Augmentation ---
    use_mixup = True
    mixup_alpha = 0.15
    use_cutmix = True
    cutmix_alpha = 1.0
    mixup_prob = 0.15
    
    # NEW: Disable MixUp in last N epochs to let model settle on real distribution
    mixup_off_epoch = 0  
    
    # --- Loss Function ---
    # Options: "focal", "ldam", "ce"
    loss_type = "ldam" 
    
    # Focal Settings
    focal_gamma = 2.0
    label_smoothing = 0.002
    # LDAM Settings (New)
    ldam_max_m = 0.7
    ldam_s = 30

    
    # --- Other ---
    num_workers = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = True
    save_dir = "checkpoints_harshit"
    
    # TTA
    use_tta = True
    tta_augments = 5
    
    # Post-Processing
    optimize_thresholds = True  # NEW: Optimize F1 thresholds on valid set
    
    # Class names
    class_names = ['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY']
    num_classes = 13
    
    # Pooling: 'attention', 'gem', 'both'
    pooling_type = 'none'

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

# --------------------------
# 3. Transforms (Enhanced)
# --------------------------

def get_train_transforms(image_size):
    """Strong augmentations for training"""
    return A.Compose([
        A.CenterCrop(height=280, width=280, p=1.0),
        A.Resize(image_size, image_size), 
        # A.RandomScale(scale_limit=0.2, p=1.0), 
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.5),
        
        # # Texture Augmentations (Helpful for WBC)
        # A.OneOf([
        #     A.GaussNoise(var_limit=(10.0, 50.0)),
        #     A.GaussianBlur(blur_limit=(3, 7)),
        #     A.MotionBlur(blur_limit=5),
        #     A.ISONoise(p=1.0),
        # ], p=0.3),
        
        # # Distortion
        # A.OneOf([
        #     A.OpticalDistortion(distort_limit=0.3),
        #     A.GridDistortion(distort_limit=0.3),
        #     A.ElasticTransform(alpha=1, sigma=50, p=0.5),
        # ], p=0.3),
        
        # # Color
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        
        # A.CoarseDropout(max_holes=8, max_height=image_size//8, max_width=image_size//8, 
        #                 min_holes=1, fill_value=0, p=0.3),
        A.Normalize(mean=Config.mean, std=Config.std),
        ToTensorV2()
    ])

def get_valid_transforms(image_size):
    return A.Compose([
        A.CenterCrop(height=300, width=300, p=1.0),
        A.Resize(image_size, image_size), 
        # A.RandomScale(scale_limit=0.2, p=1.0), 
        A.Normalize(mean=Config.mean, std=Config.std),
        ToTensorV2()
    ])

def get_tta_transforms(image_size):
    """Expanded TTA Transforms"""
    base = [
            A.CenterCrop(height=300, width=300, p=1.0),
            A.Resize(image_size, image_size), 
            # A.RandomScale(scale_limit=0.2, p=1.0), 
            A.Normalize(mean=Config.mean, std=Config.std), ToTensorV2()]
    return [
        A.Compose(base),                                        # Original
        A.Compose([A.HorizontalFlip(p=1.0)] + base),           # HFlip
        A.Compose([A.VerticalFlip(p=1.0)] + base),             # VFlip
        A.Compose([A.RandomRotate90(p=1.0)] + base),           # Rotate90
        A.Compose([A.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0))] + base) # Zoom
    ]

# --------------------------
# 4. Losses: Focal (Old) & LDAM (New)
# --------------------------
class FocalLoss(nn.Module):
    """Focal Loss with label smoothing (Preserved)"""
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
            
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss.sum()

class LDAMLoss(nn.Module):
    """
    LDAM Loss (New) - Best for Long-Tailed datasets
    """
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(Config.device)
        self.m_list = m_list
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        
        # Handle soft labels (MixUp) vs Hard labels
        if target.dim() > 1:
            # For MixUp with LDAM, we use the argmax to determine margin class,
            # but compute loss against soft targets.
            target_idx = torch.argmax(target, dim=1)
        else:
            target_idx = target
            
        index.scatter_(1, target_idx.data.view(-1, 1), 1)
        
        index_float = index.type(torch.FloatTensor).to(Config.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        
        if target.dim() > 1:
             # Soft Label LDAM
             return torch.sum(-target * F.log_softmax(output * self.s, dim=-1), dim=-1).mean()
             
        return F.cross_entropy(output * self.s, target, weight=self.weight)

# --------------------------
# 5. Dataset & MixUp
# --------------------------
class WBCDataset(Dataset):
    """Dataset with albumentations augmentations (Preserved)"""
    def __init__(self, csv_path, img_dir, class_to_idx, transform=None, is_test=False):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.is_test = is_test
        
        if not is_test:
            self.df = self.df.dropna(subset=['labels'])
            self.df = self.df[self.df['labels'].isin(class_to_idx.keys())]
        
        self.image_ids = self.df['ID'].tolist()
        if 'img_dir' in self.df.columns:
            self.img_dirs = self.df['img_dir'].tolist()
        else:
            self.img_dirs = [img_dir] * len(self.image_ids)
            
        if not is_test:
            self.labels = self.df['labels'].tolist()
            # New: Store indices for easier sampling
            self.label_indices = [self.class_to_idx[l] for l in self.labels]
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_dir = self.img_dirs[idx]
        img_path = os.path.join(Config.data_root, img_dir, img_name)
        
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            image = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)
        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.show()

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # plt.imshow(image.permute(1,2,0).cpu().numpy())
        # plt.show()

        if self.is_test:
            return {'pixel_values': image, 'image_id': img_name}
        
        label = self.class_to_idx[self.labels[idx]]
        return {'pixel_values': image, 'label': torch.tensor(label, dtype=torch.long)}

def mixup_data(x, y, alpha=0.4):
    """MixUp augmentation"""
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation"""
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, num_classes):
    y_a_soft = F.one_hot(y_a, num_classes).float()
    y_b_soft = F.one_hot(y_b, num_classes).float()
    mixed_y = lam * y_a_soft + (1 - lam) * y_b_soft
    return criterion(pred, mixed_y)

# --------------------------
# 6. Classifier (Supports MedSigLIP, Dino, ConvNeXt)
# --------------------------
class WBCClassifier(nn.Module):
    """Universal Classifier"""
    def __init__(self, model_name, num_classes, dropout=0.3, config=None):
        super().__init__()
        self.config = config
        self.model_type = config.model_type if config else "medsiglip"
        self.pooling_type = config.pooling_type if config else "attention"
        
        # --- 1. Load Backbone ---
        if self.model_type == "dinobloom":
            self.backbone = get_dino_bloom(config.dinobloom_name, config.dinobloom_weights)
            hidden_size = embed_sizes[config.dinobloom_name]
            
        elif self.model_type == "convnext":
            # ConvNeXt V2 (timm)
            print(f"Loading {config.convnext_name}")
            self.backbone = timm.create_model(config.convnext_name, pretrained=True, num_classes=0, global_pool='')
            # Get feature dim
            with torch.no_grad():
                dummy = torch.randn(1, 3, config.image_size, config.image_size)
                fts = self.backbone(dummy)
                hidden_size = fts.shape[1]
                
        else: # medsiglip
            self.backbone = AutoModel.from_pretrained(config.medsiglip_name)
            if hasattr(self.backbone, 'vision_model'):
                hidden_size = self.backbone.vision_model.config.hidden_size
            else:
                hidden_size = self.backbone.config.hidden_size
        
        self._apply_freeze_strategy(config)
        
    
        self.hidden_size = hidden_size
        
        # --- 2. Pooling & Heads ---
        self.gem_p = nn.Parameter(torch.ones(1) * 3.0)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(5)])
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Calculate input size based on pooling
        if self.pooling_type == 'both': classifier_input = hidden_size * 2
        else: classifier_input = hidden_size
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input),
            nn.Linear(classifier_input, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def _apply_freeze_strategy(self, config):
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Strategy: Entire backbone frozen.")
        
        elif config.freeze_few_layers is not None:
            n = config.freeze_few_layers
            print(f"Strategy: Freezing first {n} blocks/stages of {config.model_type}")
            
            if config.model_type in ["dinobloom", "medsiglip"]:
                # Access the ViT blocks (usually .blocks or .encoder.layer)
                # This works for both Dino (timm/hub) and MedSigLIP (HF)
                backbone_base = self.backbone.vision_model if hasattr(self.backbone, 'vision_model') else self.backbone
                
                # Freeze embeddings first
                if hasattr(backbone_base, 'embeddings'):
                    for p in backbone_base.embeddings.parameters(): p.requires_grad = False
                if hasattr(backbone_base, 'patch_embed'):
                    for p in backbone_base.patch_embed.parameters(): p.requires_grad = False

                # Freeze first N blocks
                layers = None
                if hasattr(backbone_base, 'blocks'): layers = backbone_base.blocks
                elif hasattr(backbone_base, 'encoder') and hasattr(backbone_base.encoder, 'layer'):
                    layers = backbone_base.encoder.layer
                
                if layers is not None:
                    for i, block in enumerate(layers):
                        if i < n:
                            for p in block.parameters(): p.requires_grad = False
                        else: break
            
            elif config.model_type == "convnext":
                # ConvNeXt has 'stages' (usually 4)
                for i, stage in enumerate(self.backbone.stages):
                    if i < n:
                        for p in stage.parameters(): p.requires_grad = False
                    else: break

    def gem_pooling(self, x, eps=1e-6):
        """Generalized Mean Pooling"""
        # x: (B, Seq, C) OR (B, C, H, W)
        if x.dim() == 4: # ConvNeXt: (B, C, H, W)
            return F.avg_pool2d(x.clamp(min=eps).pow(self.gem_p), (x.size(-2), x.size(-1))).pow(1./self.gem_p).flatten(1)
            
        # Transformer: (B, Seq, C)
        p = self.gem_p.clamp(min=1.0)
        return (x.clamp(min=eps).pow(p).mean(dim=1)).pow(1.0 / p)
    
    def attention_pooling(self, hidden_states):
        """Attention Pooling (Only for transformers with Seq dim)"""
        # hidden_states: (B, seq_len, hidden_size)
        weights = F.softmax(self.attention(hidden_states), dim=1)
        return torch.sum(weights * hidden_states, dim=1)
        
    def forward(self, pixel_values):
        # Extract features based on model type
        if self.model_type == "dinobloom":
            features_dict = self.backbone.forward_features(pixel_values)
            if self.config.use_cls_token:
                patch_tokens = features_dict['x_norm_clstoken'].unsqueeze(1)  # (B, 1, C)
            else:
                patch_tokens = features_dict['x_norm_patchtokens']            
            if self.pooling_type == 'gem': f = self.gem_pooling(patch_tokens)
            elif self.pooling_type == 'attention': f = self.attention_pooling(patch_tokens)
            elif self.pooling_type=='both': f = torch.cat([self.gem_pooling(patch_tokens), self.attention_pooling(patch_tokens)], dim=-1)
            else: f = patch_tokens.squeeze(1)

        elif self.model_type == "convnext":
            features = self.backbone(pixel_values) # (B, C, H, W)
            # ConvNeXt is purely spatial, so we use GEM pooling. 
            # Attention pooling on 2D map requires flattening, let's stick to GEM for ConvNeXt part 
            # or replicate it to fit 'both' interface
            pooled = self.gem_pooling(features)
            if self.pooling_type == 'both':
                f = torch.cat([pooled, pooled], dim=-1)
            else:
                f = pooled
                
        else: # medsiglip
            outputs = self.backbone.vision_model(pixel_values=pixel_values)
            hidden_states = outputs.last_hidden_state
            
            if self.pooling_type == 'gem': f = self.gem_pooling(hidden_states)
            elif self.pooling_type == 'attention': f = self.attention_pooling(hidden_states)
            else: f = torch.cat([self.gem_pooling(hidden_states), self.attention_pooling(hidden_states)], dim=-1)

        # Multi-sample Dropout
        if self.training:
            logits = torch.mean(torch.stack([self.classifier(d(f)) for d in self.dropouts], dim=0), dim=0)
        else:
            logits = self.classifier(f)
            
        return logits

# --------------------------
# 7. Utils: Weights, Optimizer, Thresholds
# --------------------------
def get_class_weights(labels, class_to_idx):
    counter = Counter(labels)
    total = sum(counter.values())
    weights = []
    for cls in sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x]):
        count = counter.get(cls, 1)
        # Sqrt inverse is often better than linear inverse for extreme imbalance
        weight = 1 / (np.sqrt(count)) 
        weights.append(weight)
    weights = np.array(weights)
    weights = weights / weights.sum() * len(weights)
    return torch.FloatTensor(weights)

def optimize_temperatures(valid_targets, valid_probs, num_classes):
    """
    Optimizes a temperature vector to maximize Macro F1 in a 
    multi-class (single-label) setup.
    """
    print("Optimizing Class Temperatures...")
    
    # Define the objective function to MINIMIZE (so we return -F1)
    def objective(temps):
        # Apply temperatures: probs / temps
        # We use argmax to respect the multi-class nature
        preds = np.argmax(valid_probs / temps, axis=1)
        score = f1_score(valid_targets, preds, average='macro')
        return -score

    # Initial guess: all temperatures = 1.0
    initial_temps = np.ones(num_classes)
    
    # Constraints: Temperatures should stay within a reasonable range
    # (e.g., 0.1 to 2.0). 
    bounds = [(0.1, 2.0) for _ in range(num_classes)]

    # Use Nelder-Mead or Powell for derivative-free optimization
    result = minimize(
        objective, 
        initial_temps, 
        method='Nelder-Mead', 
        bounds=bounds,
        options={'maxiter': 100}
    )

    best_temps = result.x
    final_f1 = -result.fun
    print(f"--> Optimization complete. Best Macro F1: {final_f1:.4f}")
    return best_temps.tolist()

def apply_temperatures(probs, temperatures):
    """
    Adjusts probabilities by temperatures and returns the argmax class.
    """
    # Using the formula: class = argmax(p_i / tau_i)
    adjusted_probs = probs / np.array(temperatures)
    return np.argmax(adjusted_probs, axis=1)

# --------------------------
# 8. Training & Validation
# --------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, scaler, config, num_classes):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Dynamic MixUp: Turn off in later epochs
    use_mixup_epoch = config.use_mixup and (epoch < config.mixup_off_epoch)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for step, batch in enumerate(pbar):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)
        
        # MixUp Logic
        r = random.random()
        apply_mixup = use_mixup_epoch and r < config.mixup_prob
        apply_cutmix = config.use_cutmix and use_mixup_epoch and r > config.mixup_prob and r < (config.mixup_prob*2)
        
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
                loss = criterion(logits, labels)
            
            loss = loss / config.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
        running_loss += loss.item() * config.gradient_accumulation_steps
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item()*config.gradient_accumulation_steps:.4f}'})
        
    avg_loss = running_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, f1

@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, config):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)
        
        with autocast(enabled=config.use_amp):
            logits = model(pixel_values)
            loss = criterion(logits, labels)
        
        running_loss += loss.item()
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
    f1 = f1_score(all_labels, all_preds, average='macro')
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return running_loss/len(dataloader), bal_acc, f1, np.array(all_probs), np.array(all_labels)

# --------------------------
# 9. Main
# --------------------------
def main():
    print(f"Model: {Config.model_type} | Loss: {Config.loss_type} | Res: {Config.image_size}")
    os.makedirs(Config.save_dir, exist_ok=True)

    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=os.path.join(Config.save_dir, "logs"))
    
    #save the config parameters as json:
    config_dict = {
        k: v for k, v in Config.__dict__.items() 
        if not k.startswith("__") and not callable(v)
    }

    if "device" in config_dict:
        config_dict["device"] = str(config_dict["device"])

    

    with open(os.path.join(Config.save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    print("Configuration saved to:", str(os.path.join(Config.save_dir, "config.json")))



    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    lrs=[]

    # Class mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(Config.class_names)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    # 1. Load & Combine Data
    train_df = pd.read_csv(os.path.join(Config.data_root, Config.train_csv))
    train_df['img_dir'] = Config.train_img_dir
    
    if Config.use_phase1:
        p1_df = pd.read_csv(os.path.join(Config.data_root, Config.phase1_csv))
        p1_df = p1_df[['ID', 'labels']].copy()
        p1_df['img_dir'] = Config.phase1_img_dir
        train_df = pd.concat([train_df, p1_df], ignore_index=True)
        print(f"Combined Data Size: {len(train_df)}")
        
    train_df.to_csv("combined_train.csv", index=False)
    
    # 2. Datasets
    train_dataset = WBCDataset("combined_train.csv", Config.train_img_dir, class_to_idx, 
                               transform=get_train_transforms(Config.image_size))
    eval_dataset = WBCDataset(os.path.join(Config.data_root, Config.eval_csv), Config.eval_img_dir, class_to_idx,
                              transform=get_valid_transforms(Config.image_size))
    test_dataset = WBCDataset(os.path.join(Config.data_root, Config.test_csv), Config.test_img_dir, class_to_idx,
                              transform=get_valid_transforms(Config.image_size), is_test=True)
                              
    # 3. Weighted Sampler (Sqrt inverse)
    labels = [class_to_idx[l] for l in train_dataset.labels]
    class_counts = Counter(labels)
    # List for LDAM
    cls_num_list = [class_counts[i] for i in range(Config.num_classes)]
    
    weights = [1.0 / np.sqrt(class_counts[l]) for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, sampler=sampler, 
                              num_workers=Config.num_workers, drop_last=True, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=Config.batch_size*2, shuffle=False, 
                             num_workers=Config.num_workers)
                             
    # 4. Model & Optimizer
    model = WBCClassifier(Config.model_type, Config.num_classes, config=Config).to(Config.device)
    
    # Layer-wise LR or Standard
    if Config.use_llrd:
        # Simplified grouping: backbone vs head
        params = [
            {'params': model.backbone.parameters(), 'lr': Config.backbone_lr},
            {'params': model.classifier.parameters(), 'lr': Config.classifier_lr},
            {'params': model.attention.parameters(), 'lr': Config.classifier_lr},
            {'params': [model.gem_p], 'lr': Config.classifier_lr}
        ]
        optimizer = AdamW(params, weight_decay=Config.weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=Config.classifier_lr, weight_decay=Config.weight_decay)

    
    # 5. Loss Function Selection
    class_weights = get_class_weights(train_dataset.labels, class_to_idx).to(Config.device)
    
    if Config.loss_type == "ldam":
        print("Using LDAM Loss")
        criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=Config.ldam_max_m, s=Config.ldam_s)
    elif Config.loss_type == "focal":
        print("Using Focal Loss")
        criterion = FocalLoss(alpha=class_weights, gamma=Config.focal_gamma, label_smoothing=Config.label_smoothing)
    else:
        print("Using CrossEntropy")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=Config.label_smoothing)
        
    
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader)*5, eta_min=Config.min_lr)

    steps_per_epoch = len(train_loader)
    # warmup_steps = Config.warmup_epochs * steps_per_epoch
    # total_steps = Config.num_epochs * steps_per_epoch
    # cosine_steps = total_steps - warmup_steps

    # # 1. Linear Warmup: Start from a small fraction of the initial LR and go up to 1.0
    # warmup_scheduler = LinearLR(
    #     optimizer, 
    #     start_factor=0.1, # Start at 10% of the set LR
    #     end_factor=1.0, 
    #     total_iters=warmup_steps
    # )

    # # 2. Cosine Decay: Start from the set LR and go down to min_lr
    # cosine_scheduler = CosineAnnealingLR(
    #     optimizer, 
    #     T_max=cosine_steps, 
    #     eta_min=Config.min_lr
    # )

    # # 3. Combine them
    # scheduler = SequentialLR(
    #     optimizer, 
    #     schedulers=[warmup_scheduler, cosine_scheduler], 
    #     milestones=[warmup_steps]
    # )

    total_steps = Config.num_epochs * steps_per_epoch
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=2*total_steps, 
        eta_min=Config.min_lr
    )

    scaler = torch.amp.GradScaler(enabled=Config.use_amp)
    
    # 6. Loop
    best_f1 = 0.0
    best_temperatures = [1.0] * Config.num_classes

    for epoch in range(Config.num_epochs):
        # Delayed Re-Weighting for LDAM
        if Config.loss_type == "ldam":
            print("DRW: Enabling class weights for LDAM")
            # criterion.weight = class_weights
            idx = epoch // 30
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            criterion.weight = per_cls_weights

        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, 
                                               Config.device, epoch, scaler, Config, Config.num_classes)
        
                                               
        val_loss, val_bal, val_f1, val_probs, val_labels = validate(model, eval_loader, criterion, Config.device, epoch, Config)
        
        # --- NEW: TensorBoard Logging ---
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Metrics/Train_F1', train_f1, epoch)
        writer.add_scalar('Metrics/Val_F1', val_f1, epoch)
        writer.add_scalar('Metrics/Val_Balanced_Acc', val_bal, epoch)
        writer.add_scalar('Backbone Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Head Learning_Rate', optimizer.param_groups[-1]['lr'], epoch)

        # Log histograms of model parameters (optional, but useful for debugging)
        # for name, param in model.named_parameters():
        #     writer.add_histogram(name, param, epoch)

        print(f"Epoch {epoch+1} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        print(f"Epoch {epoch+1} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | Val BalAcc: {val_bal:.4f}")
        
        lrs.append(optimizer.param_groups[0]['lr'])

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"--> Optimization found better F1: {best_f1:.4f}")

            # Threshold Optimization
            if Config.optimize_thresholds:
                best_temperatures = optimize_temperatures(val_labels, val_probs, Config.num_classes)
                # Check score with new thresholds

                opt_preds = apply_temperatures(val_probs, best_temperatures)
                opt_f1 = f1_score(val_labels, opt_preds, average='macro')
                print(f"--> Optimization found better F1: {opt_f1:.4f}")
            
            torch.save({
                'model': model.state_dict(),
                'temperatures': best_temperatures,
                'f1': best_f1
            }, os.path.join(Config.save_dir, "best_model.pth"))


    writer.close()
        
    # 7. Final Prediction (TTA + Thresholds)
    print("\nPredicting with Best Model...")
    checkpoint = torch.load(os.path.join(Config.save_dir, "best_model.pth"), weights_only=False)
    model.load_state_dict(checkpoint['model'])
    best_temperatures = checkpoint['temperatures']
    print("Loaded Best F1:", checkpoint['f1'])
    print("Best Temperatures:", best_temperatures)
    model.eval()
    
    if Config.use_tta:
        tta_transforms = get_tta_transforms(Config.image_size)
        all_final_probs = []
        ids = []
        for i, tfs in enumerate(tta_transforms):
            print(f"TTA {i+1}")
            test_dataset.transform = tfs
            loader = DataLoader(test_dataset, batch_size=Config.batch_size*2, shuffle=False)
            step_probs = []
            if i==0: ids_list = []
            with torch.no_grad():
                for batch in tqdm(loader):
                    if i==0: ids_list.extend(batch['image_id'])
                    with autocast(enabled=Config.use_amp):
                        out = model(batch['pixel_values'].to(Config.device))
                        step_probs.append(F.softmax(out, dim=1).cpu())
            all_final_probs.append(torch.cat(step_probs))
            if i==0: ids = ids_list
        
        avg_probs = torch.stack(all_final_probs).mean(dim=0).numpy()
    else:
        # No TTA
        test_dataset.transform = get_valid_transforms(Config.image_size)
        loader = DataLoader(test_dataset, batch_size=Config.batch_size*2, shuffle=False)
        probs_list, ids_list = [], []
        with torch.no_grad():
            for batch in tqdm(loader):
                ids_list.extend(batch['image_id'])
                out = model(batch['pixel_values'].to(Config.device))
                probs_list.append(F.softmax(out, dim=1).cpu())
        avg_probs = torch.cat(probs_list).numpy()
        ids = ids_list

    # Apply optimized temperatures
    final_preds = apply_temperatures(avg_probs, best_temperatures)

    # final_preds = avg_probs.argmax(axis=1)

    sub = pd.DataFrame({'ID': ids, 'labels': [idx_to_class[p] for p in final_preds]})
    sub.to_csv(os.path.join(Config.save_dir, "submission_ultimate.csv"), index=False)
    print("Done.")

if __name__ == "__main__":
    main()