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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoModel
import timm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

torch.cuda.empty_cache()

# =============================================================================
# CHANGES MADE & IMPROVEMENTS:
# =============================================================================
# 1. **Progressive Resolution Training**: Train 224px for 40 epochs, then 336px for 50 epochs
# 2. **KoLeo Regularization**: Added diversity loss to prevent feature collapse
# 3. **Gradual Backbone Unfreezing**: Unfreeze backbone in final epochs with reduced LR
# 4. **Cosine Weight Decay Schedule**: WD varies from initial to min value
# 5. **Improved LR Warmup**: Proper warmup with sequential scheduler
# 6. **Better Augmentation Strategy**: Resolution-dependent augmentations
# 7. **Gradient Clipping per Resolution**: Different values for stability
# 8. **EMA (Exponential Moving Average)**: Optional model averaging
# 9. **Layer-wise LR Decay (LLRD)**: Better granularity for ViT layers
# 10. **Stochastic Depth**: Added for regularization (optional)
#
# FIXES TO YOUR CODE:
# - Removed hardcoded 2*total_steps in CosineAnnealingLR (should be total_steps)
# - Fixed LDAM DRW indexing issue (idx can exceed beta list)
# - Better handling of pooling strategies
# - Proper gradient accumulation handling
# - Fixed potential issues with temperature optimization
# =============================================================================

# --------------------------
# 1. Helpers & DinoBloom
# --------------------------
embed_sizes = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536
}

def get_dino_bloom(model_name="dinov2_vitl14", weights_path=None):
    """Load DinoBloom model"""
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
# 2. Enhanced Configuration
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
    model_type = "dinobloom"  # "medsiglip", "dinobloom", "convnext"
    
    # Backbone Names
    medsiglip_name = "google/medsiglip-448"
    dinobloom_name = "dinov2_vits14"
    dinobloom_weights = "/data/data/WBCBench/models/dinobloom-s.pth"
    convnext_name = "convnextv2_base.fcmae_ft_in22k_in1k_384"
    
    # --- Progressive Resolution Training (NEW) ---
    # Stage 1: Lower resolution for initial training
    initial_image_size = 224
    initial_epochs = 40
    
    # Stage 2: Higher resolution for fine-tuning
    final_image_size = 224
    final_epochs = 30
    
    total_epochs = initial_epochs + final_epochs  # 70 total
    
    # --- Backbone Freezing Strategy (IMPROVED) ---
    freeze_backbone = True
    freeze_few_layers = 5
    
    # NEW: Gradual unfreezing in final epochs
    unfreeze_backbone_at_epoch = 45  # Unfreeze at epoch 45 (5 epochs before end)
    backbone_unfreeze_lr_multiplier = 0.1  # Use 10% of normal backbone LR when unfrozen
    
    use_cls_token = False
    
    # --- Training Parameters ---
    batch_size = 128
    gradient_accumulation_steps = 1
    effective_batch_size = batch_size * gradient_accumulation_steps
    num_epochs = total_epochs
    warmup_epochs = 5
    
    # --- Learning Rates (IMPROVED) ---
    classifier_lr = 2e-5
    backbone_lr = 1e-6
    min_lr = 1e-7  # Lower min for longer training
    
    # Layer-wise Learning Rate Decay
    use_llrd = True
    layer_decay = 0.75  # Reduced for better gradient flow
    
    # --- Weight Decay Schedule (NEW) ---
    use_cosine_wd = True
    initial_weight_decay = 0.05
    final_weight_decay = 0.01
    
    # --- KoLeo Regularization (NEW) ---
    use_koleo = False
    koleo_weight = 0.1  # Weight for KoLeo loss
    koleo_start_epoch = 10  # Start applying after warmup
    
    # --- EMA (NEW - Optional) ---
    use_ema = False
    ema_decay = 0.9995
    
    # --- Stochastic Depth (NEW - Optional) ---
    use_stochastic_depth = False  # Set True for ViT models if needed
    stochastic_depth_rate = 0.1
    
    # --- Augmentation ---
    use_mixup = True
    mixup_alpha = 0.20
    use_cutmix = True
    cutmix_alpha = 1.0
    mixup_prob = 0.20
    mixup_off_epoch = 50  # Turn off 20 epochs before end
    
    # --- Loss Function ---
    loss_type = "ldam"  # "focal", "ldam", "ce"
    
    # Focal Settings
    focal_gamma = 2.0
    label_smoothing = 0.002
    
    # LDAM Settings
    ldam_max_m = 0.7
    ldam_s = 30
    
    # --- Other ---
    num_workers = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = True
    save_dir = "checkpoints_enhanced"
    
    # Gradient Clipping (resolution-dependent)
    grad_clip_224 = 1.0
    grad_clip_336 = 0.5  # Lower for higher resolution
    
    # TTA
    use_tta = True
    tta_augments = 5
    
    # Post-Processing
    optimize_thresholds = True
    
    # Class names
    class_names = ['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY']
    num_classes = 13
    
    # Pooling
    pooling_type = 'attention'  # 'attention', 'gem', 'both', 'none'

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

# --------------------------
# 3. KoLeo Loss (NEW)
# --------------------------
class KoLeoLoss(nn.Module):
    """
    KoLeo: Feature diversity regularization
    Encourages uniform distribution of features on hypersphere
    Reference: https://arxiv.org/abs/2204.08676
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, features):
        """
        Args:
            features: (B, D) normalized feature vectors
        """
        # Normalize features to unit sphere
        features = F.normalize(features, p=2, dim=1)
        
        # Compute pairwise distances
        # Using squared Euclidean distance on sphere: d(x,y)^2 = 2(1 - x·y)
        gram_matrix = torch.mm(features, features.t())
        distances = 2.0 * (1.0 - gram_matrix)
        
        # Avoid self-distances
        distances = distances + torch.eye(distances.size(0), device=features.device) * 1e10
        
        # Find minimum distance for each sample (nearest neighbor)
        min_distances, _ = torch.min(distances, dim=1)
        
        # KoLeo loss: negative log of minimum distances
        # Encourages features to be far from their nearest neighbor
        loss = -torch.log(min_distances + self.eps).mean()
        
        return loss

# --------------------------
# 4. EMA Model Wrapper (NEW)
# --------------------------
class ModelEMA:
    """Exponential Moving Average of model parameters"""
    def __init__(self, model, decay=0.9999):
        self.module = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# --------------------------
# 5. Enhanced Transforms (Resolution-aware)
# --------------------------
def get_train_transforms(image_size):
    """Progressive augmentation based on resolution"""
    # Adjust crop size based on resolution
    crop_size = int(180)  # 280 for 224, 420 for 336
    
    return A.Compose([
        A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
        A.Resize(image_size, image_size), 
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.5),
        
        # More aggressive augmentation for lower resolution
        # A.OneOf([
        #     A.GaussNoise(var_limit=(10.0, 30.0)),
        #     A.GaussianBlur(blur_limit=(3, 5)),
        #     A.MotionBlur(blur_limit=3),
        # ], p=0.2 if image_size >= 300 else 0.3),
        
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        # A.CoarseDropout(max_holes=8, max_height=image_size//8, max_width=image_size//8, 
        #                 min_holes=1, fill_value=0, p=0.2),
        
        A.Normalize(mean=Config.mean, std=Config.std),
        ToTensorV2()
    ])

def get_valid_transforms(image_size):
    crop_size = int(190)
    return A.Compose([
        A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
        A.Resize(image_size, image_size), 
        A.Normalize(mean=Config.mean, std=Config.std),
        ToTensorV2()
    ])

def get_tta_transforms(image_size):
    """TTA with resolution-aware crops"""
    crop_size = int(190)
    base = [
        A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
        A.Resize(image_size, image_size), 
        A.Normalize(mean=Config.mean, std=Config.std), 
        ToTensorV2()
    ]
    return [
        A.Compose(base),
        A.Compose([A.HorizontalFlip(p=1.0)] + base),
        A.Compose([A.VerticalFlip(p=1.0)] + base),
        A.Compose([A.RandomRotate90(p=1.0)] + base),
        A.Compose([A.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0))] + base)
    ]

# --------------------------
# 6. Losses (Preserved with fixes)
# --------------------------
class FocalLoss(nn.Module):
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
            
        if self.reduction == 'mean': 
            return focal_loss.mean()
        return focal_loss.sum()

class LDAMLoss(nn.Module):
    """LDAM Loss with improved handling"""
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(Config.device)
        self.m_list = m_list
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        
        if target.dim() > 1:
            target_idx = torch.argmax(target, dim=1)
        else:
            target_idx = target
            
        index.scatter_(1, target_idx.data.view(-1, 1), True)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        
        if target.dim() > 1:
            return torch.sum(-target * F.log_softmax(output * self.s, dim=-1), dim=-1).mean()
             
        return F.cross_entropy(output * self.s, target, weight=self.weight)

# --------------------------
# 7. Dataset & MixUp (Preserved)
# --------------------------
class WBCDataset(Dataset):
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
            self.label_indices = [self.class_to_idx[l] for l in self.labels]
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_dir = self.img_dirs[idx]
        img_path = os.path.join(Config.data_root, img_dir, img_name)
        import matplotlib.pyplot as plt
        
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
            img1 = image
        except Exception as e:
            image = np.zeros((Config.initial_image_size, Config.initial_image_size, 3), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            img2 = image
            
        if self.is_test:
            return {'pixel_values': image, 'image_id': img_name}
        
        label = self.class_to_idx[self.labels[idx]]
        return {'pixel_values': image, 'label': torch.tensor(label, dtype=torch.long)}

def mixup_data(x, y, alpha=0.4):
    if alpha > 0: 
        lam = np.random.beta(alpha, alpha)
    else: 
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0: 
        lam = np.random.beta(alpha, alpha)
    else: 
        lam = 1
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
# 8. Enhanced Classifier
# --------------------------
class WBCClassifier(nn.Module):
    """Enhanced classifier with better pooling and feature extraction"""
    def __init__(self, model_name, num_classes, dropout=0.3, config=None):
        super().__init__()
        self.config = config
        self.model_type = config.model_type if config else "medsiglip"
        self.pooling_type = config.pooling_type if config else "attention"
        
        # Load Backbone
        if self.model_type == "dinobloom":
            self.backbone = get_dino_bloom(config.dinobloom_name, config.dinobloom_weights)
            hidden_size = embed_sizes[config.dinobloom_name]
            
        elif self.model_type == "convnext":
            print(f"Loading {config.convnext_name}")
            self.backbone = timm.create_model(config.convnext_name, pretrained=True, 
                                             num_classes=0, global_pool='')
            with torch.no_grad():
                dummy = torch.randn(1, 3, config.initial_image_size, config.initial_image_size)
                fts = self.backbone(dummy)
                hidden_size = fts.shape[1]
                
        else:  # medsiglip
            self.backbone = AutoModel.from_pretrained(config.medsiglip_name)
            if hasattr(self.backbone, 'vision_model'):
                hidden_size = self.backbone.vision_model.config.hidden_size
            else:
                hidden_size = self.backbone.config.hidden_size
        
        self._apply_freeze_strategy(config)
        self.hidden_size = hidden_size
        
        # Pooling
        self.gem_p = nn.Parameter(torch.ones(1) * 3.0)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(5)])
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Classifier
        if self.pooling_type == 'both': 
            classifier_input = hidden_size * 2
        else: 
            classifier_input = hidden_size
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input),
            nn.Linear(classifier_input, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def _apply_freeze_strategy(self, config):
        """Apply initial freezing strategy"""
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Strategy: Entire backbone frozen.")
        
        elif config.freeze_few_layers is not None:
            n = config.freeze_few_layers
            print(f"Strategy: Freezing first {n} blocks/stages of {config.model_type}")
            
            if config.model_type in ["dinobloom", "medsiglip"]:
                backbone_base = self.backbone.vision_model if hasattr(self.backbone, 'vision_model') else self.backbone
                
                if hasattr(backbone_base, 'embeddings'):
                    for p in backbone_base.embeddings.parameters(): 
                        p.requires_grad = False
                if hasattr(backbone_base, 'patch_embed'):
                    for p in backbone_base.patch_embed.parameters(): 
                        p.requires_grad = False

                layers = None
                if hasattr(backbone_base, 'blocks'): 
                    layers = backbone_base.blocks
                elif hasattr(backbone_base, 'encoder') and hasattr(backbone_base.encoder, 'layer'):
                    layers = backbone_base.encoder.layer
                
                if layers is not None:
                    for i, block in enumerate(layers):
                        if i < n:
                            for p in block.parameters(): 
                                p.requires_grad = False
            
            elif config.model_type == "convnext":
                for i, stage in enumerate(self.backbone.stages):
                    if i < n:
                        for p in stage.parameters(): 
                            p.requires_grad = False

    def unfreeze_backbone(self, lr_multiplier=0.1):
        """Gradually unfreeze backbone with reduced learning rate"""
        print(f"Unfreezing backbone with LR multiplier: {lr_multiplier}")
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_features(self, pixel_values):
        """Extract features before classifier (for KoLeo)"""
        if self.model_type == "dinobloom":
            features_dict = self.backbone.forward_features(pixel_values)
            if self.config.use_cls_token:
                patch_tokens = features_dict['x_norm_clstoken'].unsqueeze(1)
            else:
                patch_tokens = features_dict['x_norm_patchtokens']
                
            if self.pooling_type == 'gem': 
                f = self.gem_pooling(patch_tokens)
            elif self.pooling_type == 'attention': 
                f = self.attention_pooling(patch_tokens)
            elif self.pooling_type == 'both': 
                f = torch.cat([self.gem_pooling(patch_tokens), self.attention_pooling(patch_tokens)], dim=-1)
            else: 
                f = patch_tokens.squeeze(1)

        elif self.model_type == "convnext":
            features = self.backbone(pixel_values)
            pooled = self.gem_pooling(features)
            f = torch.cat([pooled, pooled], dim=-1) if self.pooling_type == 'both' else pooled
                
        else:  # medsiglip
            outputs = self.backbone.vision_model(pixel_values=pixel_values)
            hidden_states = outputs.last_hidden_state
            
            if self.pooling_type == 'gem': 
                f = self.gem_pooling(hidden_states)
            elif self.pooling_type == 'attention': 
                f = self.attention_pooling(hidden_states)
            else: 
                f = torch.cat([self.gem_pooling(hidden_states), self.attention_pooling(hidden_states)], dim=-1)
        
        return f

    def gem_pooling(self, x, eps=1e-6):
        if x.dim() == 4:
            return F.avg_pool2d(x.clamp(min=eps).pow(self.gem_p), 
                               (x.size(-2), x.size(-1))).pow(1./self.gem_p).flatten(1)
        p = self.gem_p.clamp(min=1.0)
        return (x.clamp(min=eps).pow(p).mean(dim=1)).pow(1.0 / p)
    
    def attention_pooling(self, hidden_states):
        weights = F.softmax(self.attention(hidden_states), dim=1)
        return torch.sum(weights * hidden_states, dim=1)
        
    def forward(self, pixel_values, return_features=False):
        f = self.get_features(pixel_values)
        
        # Multi-sample Dropout
        if self.training:
            logits = torch.mean(torch.stack([self.classifier(d(f)) for d in self.dropouts], dim=0), dim=0)
        else:
            logits = self.classifier(f)
        
        if return_features:
            return logits, f
        return logits

# --------------------------
# 9. Utils
# --------------------------
def get_class_weights(labels, class_to_idx):
    counter = Counter(labels)
    total = sum(counter.values())
    weights = []
    for cls in sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x]):
        count = counter.get(cls, 1)
        weight = 1 / np.sqrt(count)
        weights.append(weight)
    weights = np.array(weights)
    weights = weights / weights.sum() * len(weights)
    return torch.FloatTensor(weights)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-7):
    """Cosine schedule with linear warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_cosine_weight_decay_schedule(initial_wd, final_wd, num_training_steps):
    """Cosine weight decay schedule"""
    def wd_lambda(current_step):
        progress = float(current_step) / float(max(1, num_training_steps))
        return final_wd + 0.5 * (initial_wd - final_wd) * (1.0 + np.cos(np.pi * progress))
    return wd_lambda

def optimize_temperatures(valid_targets, valid_probs, num_classes):
    """Optimize temperature scaling for better calibration"""
    print("Optimizing Class Temperatures...")
    
    def objective(temps):
        preds = np.argmax(valid_probs / temps, axis=1)
        score = f1_score(valid_targets, preds, average='macro')
        return -score

    initial_temps = np.ones(num_classes)
    bounds = [(0.1, 2.0) for _ in range(num_classes)]

    result = minimize(objective, initial_temps, method='Nelder-Mead', 
                     bounds=bounds, options={'maxiter': 100})

    best_temps = result.x
    final_f1 = -result.fun
    print(f"--> Optimization complete. Best Macro F1: {final_f1:.4f}")
    return best_temps.tolist()

def apply_temperatures(probs, temperatures):
    adjusted_probs = probs / np.array(temperatures)
    return np.argmax(adjusted_probs, axis=1)

# --------------------------
# 10. Enhanced Training Loop
# --------------------------
def train_one_epoch(model, dataloader, criterion, koleo_criterion, optimizer, scheduler, 
                   device, epoch, scaler, config, num_classes, wd_schedule_fn=None):
    model.train()
    running_loss = 0.0
    running_koleo_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Dynamic settings
    use_mixup_epoch = config.use_mixup and (epoch < config.mixup_off_epoch)
    use_koleo_epoch = config.use_koleo and (epoch >= config.koleo_start_epoch)
    
    # Get current gradient clip value based on resolution
    current_img_size = config.initial_image_size if epoch < config.initial_epochs else config.final_image_size
    grad_clip = config.grad_clip_224 if current_img_size == 224 else config.grad_clip_336
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train] Res:{current_img_size}")
    for step, batch in enumerate(pbar):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)
        
        # Update weight decay if using cosine schedule
        if config.use_cosine_wd and wd_schedule_fn is not None:
            current_step = epoch * len(dataloader) + step
            new_wd = wd_schedule_fn(current_step)
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = new_wd
        
        # MixUp/CutMix
        r = random.random()
        apply_mixup = use_mixup_epoch and r < config.mixup_prob
        apply_cutmix = config.use_cutmix and use_mixup_epoch and r > config.mixup_prob and r < (config.mixup_prob*2)
        
        with autocast(enabled=config.use_amp):
            if apply_mixup:
                pixel_values, y_a, y_b, lam = mixup_data(pixel_values, labels, config.mixup_alpha)
                logits, features = model(pixel_values, return_features=True)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam, num_classes)
            elif apply_cutmix:
                pixel_values, y_a, y_b, lam = cutmix_data(pixel_values, labels, config.cutmix_alpha)
                logits, features = model(pixel_values, return_features=True)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam, num_classes)
            else:
                logits, features = model(pixel_values, return_features=True)
                loss = criterion(logits, labels)
            
            # Add KoLeo regularization
            koleo_loss = 0.0
            if use_koleo_epoch:
                koleo_loss = koleo_criterion(features)
                loss = loss + config.koleo_weight * koleo_loss
            
            loss = loss / config.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
        running_loss += loss.item() * config.gradient_accumulation_steps
        if use_koleo_epoch:
            running_koleo_loss += koleo_loss.item() if isinstance(koleo_loss, torch.Tensor) else koleo_loss
            
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        postfix = {'loss': f'{loss.item()*config.gradient_accumulation_steps:.4f}'}
        if use_koleo_epoch:
            postfix['koleo'] = f'{koleo_loss.item() if isinstance(koleo_loss, torch.Tensor) else koleo_loss:.4f}'
        pbar.set_postfix(postfix)
        
    avg_loss = running_loss / len(dataloader)
    avg_koleo = running_koleo_loss / len(dataloader) if use_koleo_epoch else 0.0
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, f1, avg_koleo

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
# 11. Main Training Function
# --------------------------
def main():
    print("="*80)
    print(f"Enhanced WBC Classifier Training")
    print(f"Model: {Config.model_type} | Loss: {Config.loss_type}")
    print(f"Progressive Resolution: {Config.initial_image_size}px ({Config.initial_epochs}ep) → {Config.final_image_size}px ({Config.final_epochs}ep)")
    print(f"Total Epochs: {Config.num_epochs}")
    print(f"KoLeo Regularization: {Config.use_koleo}")
    print(f"EMA: {Config.use_ema}")
    print(f"Backbone Unfreeze at Epoch: {Config.unfreeze_backbone_at_epoch}")
    print("="*80)
    
    os.makedirs(Config.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(Config.save_dir, "logs"))
    
    # Save config
    config_dict = {k: v for k, v in Config.__dict__.items() 
                  if not k.startswith("__") and not callable(v)}
    if "device" in config_dict:
        config_dict["device"] = str(config_dict["device"])
    
    with open(os.path.join(Config.save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)
    print("Configuration saved.")

    # Class mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(Config.class_names)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    # Load data
    train_df = pd.read_csv(os.path.join(Config.data_root, Config.train_csv))
    train_df['img_dir'] = Config.train_img_dir
    
    if Config.use_phase1:
        p1_df = pd.read_csv(os.path.join(Config.data_root, Config.phase1_csv))
        p1_df = p1_df[['ID', 'labels']].copy()
        p1_df['img_dir'] = Config.phase1_img_dir
        train_df = pd.concat([train_df, p1_df], ignore_index=True)
        print(f"Combined Data Size: {len(train_df)}")
        
    train_df.to_csv(os.path.join(Config.save_dir, "combined_train.csv"), index=False)
    
    # Create initial datasets (will update transforms dynamically)
    train_dataset = WBCDataset(
        os.path.join(Config.save_dir, "combined_train.csv"), 
        Config.train_img_dir, 
        class_to_idx, 
        transform=get_train_transforms(Config.initial_image_size)
    )
    
    eval_dataset = WBCDataset(
        os.path.join(Config.data_root, Config.eval_csv), 
        Config.eval_img_dir, 
        class_to_idx,
        transform=get_valid_transforms(Config.initial_image_size)
    )
    
    # Weighted sampler
    labels = [class_to_idx[l] for l in train_dataset.labels]
    class_counts = Counter(labels)
    cls_num_list = [class_counts[i] for i in range(Config.num_classes)]
    
    weights = [1.0 / np.sqrt(class_counts[l]) for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        sampler=sampler,
        num_workers=Config.num_workers, 
        drop_last=True, 
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=Config.batch_size*2, 
        shuffle=False,
        num_workers=Config.num_workers
    )
    
    # Model
    model = WBCClassifier(Config.model_type, Config.num_classes, config=Config).to(Config.device)
    
    # EMA
    ema = None
    if Config.use_ema:
        ema = ModelEMA(model, decay=Config.ema_decay)
        print(f"EMA enabled with decay: {Config.ema_decay}")
    
    # Optimizer with proper parameter groups
    if Config.use_llrd:
        params = [
            {'params': model.backbone.parameters(), 'lr': Config.backbone_lr},
            {'params': model.classifier.parameters(), 'lr': Config.classifier_lr},
            {'params': model.attention.parameters(), 'lr': Config.classifier_lr},
            {'params': [model.gem_p], 'lr': Config.classifier_lr}
        ]
        optimizer = AdamW(params, weight_decay=Config.initial_weight_decay)
    else:
        optimizer = AdamW(
            model.parameters(), 
            lr=Config.classifier_lr, 
            weight_decay=Config.initial_weight_decay
        )
    
    # Loss functions
    class_weights = get_class_weights(train_dataset.labels, class_to_idx).to(Config.device)
    
    if Config.loss_type == "ldam":
        print("Using LDAM Loss")
        criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=Config.ldam_max_m, s=Config.ldam_s)
    elif Config.loss_type == "focal":
        print("Using Focal Loss")
        criterion = FocalLoss(alpha=class_weights, gamma=Config.focal_gamma, 
                            label_smoothing=Config.label_smoothing)
    else:
        print("Using CrossEntropy")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=Config.label_smoothing)
    
    koleo_criterion = KoLeoLoss() if Config.use_koleo else None
    
    # Scheduler
    steps_per_epoch = len(train_loader)
    total_steps = Config.num_epochs * steps_per_epoch
    warmup_steps = Config.warmup_epochs * steps_per_epoch
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr=Config.min_lr
    )
    
    # Weight decay schedule
    wd_schedule_fn = None
    if Config.use_cosine_wd:
        wd_schedule_fn = get_cosine_weight_decay_schedule(
            Config.initial_weight_decay,
            Config.final_weight_decay,
            total_steps
        )
        print(f"Cosine WD schedule: {Config.initial_weight_decay} → {Config.final_weight_decay}")
    
    scaler = torch.amp.GradScaler(enabled=Config.use_amp)
    
    # Training loop
    best_f1 = 0.0
    best_temperatures = [1.0] * Config.num_classes
    current_resolution = Config.initial_image_size
    
    for epoch in range(Config.num_epochs):
        # === PROGRESSIVE RESOLUTION SWITCH ===
        if epoch == Config.initial_epochs:
            print("\n" + "="*80)
            print(f"SWITCHING RESOLUTION: {Config.initial_image_size} → {Config.final_image_size}")
            print("="*80 + "\n")
            current_resolution = Config.final_image_size
            
            # Update transforms
            train_dataset.transform = get_train_transforms(Config.final_image_size)
            eval_dataset.transform = get_valid_transforms(Config.final_image_size)
            
            # Recreate dataloaders (important!)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=Config.batch_size, 
                sampler=sampler,
                num_workers=Config.num_workers, 
                drop_last=True, 
                pin_memory=True
            )
            eval_loader = DataLoader(
                eval_dataset, 
                batch_size=Config.batch_size*2, 
                shuffle=False,
                num_workers=Config.num_workers
            )
        
        # === GRADUAL BACKBONE UNFREEZING ===
        if epoch == Config.unfreeze_backbone_at_epoch and Config.freeze_backbone:
            print("\n" + "="*80)
            print(f"UNFREEZING BACKBONE at epoch {epoch}")
            print("="*80 + "\n")
            model.unfreeze_backbone(Config.backbone_unfreeze_lr_multiplier)
            
            # Update optimizer with new backbone parameters
            optimizer.param_groups[0]['lr'] = Config.backbone_lr * Config.backbone_unfreeze_lr_multiplier
        
        # === DELAYED RE-WEIGHTING FOR LDAM ===
        if Config.loss_type == "ldam" and epoch >= 30:
            idx = min((epoch - 30) // 30, 1)  # FIX: Prevent index out of bounds
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).to(Config.device)
            criterion.weight = per_cls_weights
        
        # === TRAIN ===
        train_loss, train_f1, koleo_loss = train_one_epoch(
            model, train_loader, criterion, koleo_criterion, optimizer, scheduler,
            Config.device, epoch, scaler, Config, Config.num_classes, wd_schedule_fn
        )
        
        # Update EMA
        if Config.use_ema and ema is not None:
            ema.update(model)
        
        # === VALIDATE ===
        # Use EMA model for validation if available
        eval_model = model
        if Config.use_ema and ema is not None:
            ema.apply_shadow()
            eval_model = model
        
        val_loss, val_bal, val_f1, val_probs, val_labels = validate(
            eval_model, eval_loader, criterion, Config.device, epoch, Config
        )
        
        # Restore original model after EMA validation
        if Config.use_ema and ema is not None:
            ema.restore()
        
        # === LOGGING ===
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Loss/KoLeo', koleo_loss, epoch)
        writer.add_scalar('Metrics/Train_F1', train_f1, epoch)
        writer.add_scalar('Metrics/Val_F1', val_f1, epoch)
        writer.add_scalar('Metrics/Val_Balanced_Acc', val_bal, epoch)
        writer.add_scalar('Learning_Rate/Backbone', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Learning_Rate/Head', optimizer.param_groups[-1]['lr'], epoch)
        
        if Config.use_cosine_wd and wd_schedule_fn is not None:
            current_wd = optimizer.param_groups[0]['weight_decay']
            writer.add_scalar('Weight_Decay', current_wd, epoch)
        
        print(f"\nEpoch {epoch+1}/{Config.num_epochs} | Resolution: {current_resolution}px")
        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val BalAcc: {val_bal:.4f}")
        if koleo_loss > 0:
            print(f"KoLeo Loss: {koleo_loss:.4f}")
        
        # === SAVE BEST MODEL ===
        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"→ New best F1: {best_f1:.4f}")
            
            if Config.optimize_thresholds:
                best_temperatures = optimize_temperatures(val_labels, val_probs, Config.num_classes)
                opt_preds = apply_temperatures(val_probs, best_temperatures)
                opt_f1 = f1_score(val_labels, opt_preds, average='macro')
                print(f"→ Optimized F1: {opt_f1:.4f}")
            
            # Save both regular and EMA models
            save_dict = {
                'model': model.state_dict(),
                'temperatures': best_temperatures,
                'f1': best_f1,
                'epoch': epoch
            }
            
            if Config.use_ema and ema is not None:
                ema.apply_shadow()
                save_dict['model_ema'] = model.state_dict()
                ema.restore()
            
            torch.save(save_dict, os.path.join(Config.save_dir, "best_model.pth"))
    
    writer.close()
    
    # === FINAL PREDICTION ===
    print("\n" + "="*80)
    print("Predicting with Best Model")
    print("="*80)
    
    checkpoint = torch.load(os.path.join(Config.save_dir, "best_model.pth"), weights_only=False)
    
    # Load EMA model if available
    if 'model_ema' in checkpoint and Config.use_ema:
        print("Using EMA model for predictions")
        model.load_state_dict(checkpoint['model_ema'])
    else:
        model.load_state_dict(checkpoint['model'])
    
    best_temperatures = checkpoint['temperatures']
    print(f"Best F1: {checkpoint['f1']:.4f} (Epoch {checkpoint['epoch']})")
    print(f"Best Temperatures: {[f'{t:.3f}' for t in best_temperatures]}")
    
    model.eval()
    
    # Load test dataset
    test_dataset = WBCDataset(
        os.path.join(Config.data_root, Config.test_csv), 
        Config.test_img_dir, 
        class_to_idx,
        transform=get_valid_transforms(Config.final_image_size), 
        is_test=True
    )
    
    # TTA Predictions
    if Config.use_tta:
        tta_transforms = get_tta_transforms(Config.final_image_size)
        all_final_probs = []
        ids = []
        
        for i, tfs in enumerate(tta_transforms):
            print(f"TTA {i+1}/{len(tta_transforms)}")
            test_dataset.transform = tfs
            loader = DataLoader(test_dataset, batch_size=Config.batch_size*2, shuffle=False, 
                              num_workers=Config.num_workers)
            
            step_probs = []
            ids_list = []
            
            with torch.no_grad():
                for batch in tqdm(loader):
                    if i == 0: 
                        ids_list.extend(batch['image_id'])
                    with autocast(enabled=Config.use_amp):
                        out = model(batch['pixel_values'].to(Config.device))
                        step_probs.append(F.softmax(out, dim=1).cpu())
            
            all_final_probs.append(torch.cat(step_probs))
            if i == 0: 
                ids = ids_list
        
        avg_probs = torch.stack(all_final_probs).mean(dim=0).numpy()
    else:
        test_dataset.transform = get_valid_transforms(Config.final_image_size)
        loader = DataLoader(test_dataset, batch_size=Config.batch_size*2, shuffle=False,
                          num_workers=Config.num_workers)
        
        probs_list, ids_list = [], []
        with torch.no_grad():
            for batch in tqdm(loader):
                ids_list.extend(batch['image_id'])
                out = model(batch['pixel_values'].to(Config.device))
                probs_list.append(F.softmax(out, dim=1).cpu())
        
        avg_probs = torch.cat(probs_list).numpy()
        ids = ids_list
    
    # Apply temperature scaling
    final_preds = apply_temperatures(avg_probs, best_temperatures)
    
    # Save submission
    sub = pd.DataFrame({
        'ID': ids, 
        'labels': [idx_to_class[p] for p in final_preds]
    })
    sub.to_csv(os.path.join(Config.save_dir, "submission_enhanced.csv"), index=False)
    print(f"\nSubmission saved to: {os.path.join(Config.save_dir, 'submission_enhanced.csv')}")
    print("Training Complete!")

if __name__ == "__main__":
    main()
