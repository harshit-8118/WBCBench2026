"""
Attention Visualization and Model Interpretability
Includes GradCAM, attention maps, and prediction analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import from main training script
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from harshit_corner.train_enhanced import (
    Config, WBCClassifier, WBCDataset,
    get_valid_transforms, set_seed
)

set_seed(42)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (GradCAM)
    Visualizes which parts of the image the model focuses on
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: Layer to compute gradients from
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks
        self.hooks.append(target_layer.register_forward_hook(self.save_activation))
        self.hooks.append(target_layer.register_full_backward_hook(self.save_gradient))
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate CAM for input image
        
        Args:
            input_image: (1, C, H, W) tensor
            target_class: Target class index. If None, uses predicted class
            
        Returns:
            cam: (H, W) heatmap
            prediction: predicted class
        """
        self.model.eval()
        
        # Reset gradients and activations
        self.gradients = None
        self.activations = None
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Check if gradients were captured
        if self.gradients is None or self.activations is None:
            print("Warning: Gradients or activations not captured. Using fallback.")
            # Return uniform heatmap as fallback
            cam = np.ones((14, 14))
            return cam, target_class, F.softmax(output, dim=1)[0].detach().cpu().numpy()
        
        # Compute CAM
        gradients = self.gradients[0]  # (C, H', W') or (N, C)
        activations = self.activations[0]  # (C, H', W') or (N, C)
        
        # Handle different backbone types
        if gradients.dim() == 3:  # CNN (C, H, W)
            # Global average pooling of gradients
            weights = gradients.mean(dim=(1, 2), keepdim=True)
            # Weighted combination
            cam = (weights * activations).sum(dim=0)
        elif gradients.dim() == 2:  # ViT (N, C) - patch tokens
            # Average across patches
            weights = gradients.mean(dim=0)
            cam = (weights * activations).sum(dim=1)
            
            # Reshape to 2D
            num_patches = activations.shape[0]
            patch_size = int(np.sqrt(num_patches))
            if patch_size * patch_size == num_patches:
                cam = cam.reshape(patch_size, patch_size)
            else:
                # Handle non-square patches
                cam = cam.reshape(-1, 1)
                cam = cam.repeat(1, patch_size)[:patch_size, :]
        else:  # (C,) - single token
            cam = activations.mean(dim=0)
            # Create uniform map
            cam = cam.unsqueeze(0).unsqueeze(0)
            if cam.dim() == 2:
                cam = cam.unsqueeze(0).unsqueeze(0)
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        
        return cam.cpu().numpy(), target_class, F.softmax(output, dim=1)[0].detach().cpu().numpy()


class AttentionVisualizer:
    """Visualize attention patterns from ViT models"""
    
    def __init__(self, model):
        self.model = model
        self.attentions = []
        
    def get_attention_maps(self, image, layer_idx=-1):
        """
        Extract attention maps from ViT
        
        Args:
            image: Input image tensor
            layer_idx: Which transformer layer to visualize (-1 for last)
        """
        self.model.eval()
        
        # This requires model modifications to return attention weights
        # For DINOv2, you might need to access internal attention modules
        # This is a placeholder - actual implementation depends on model architecture
        
        with torch.no_grad():
            # You may need to modify your model to return attention weights
            # For now, return None as placeholder
            output = self.model(image)
        
        return None  # Placeholder


def overlay_heatmap_on_image(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on original image
    
    Args:
        image: Original image (H, W, 3) in range [0, 1]
        heatmap: Heatmap (H, W) in range [0, 1]
        alpha: Blending factor
        colormap: OpenCV colormap
    
    Returns:
        overlaid: (H, W, 3) image with heatmap overlay
    """
    # Resize heatmap to match image size
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to RGB
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), 
        colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
    
    # Ensure image is in [0, 1]
    if image.max() > 1.0:
        image = image / 255.0
    
    # Blend
    overlaid = alpha * heatmap_colored + (1 - alpha) * image
    overlaid = np.clip(overlaid, 0, 1)
    
    return overlaid


def visualize_predictions_per_class(model, dataset, class_names, output_dir, 
                                   samples_per_class=5, device='cuda'):
    """
    Visualize model predictions with GradCAM for each class
    Creates subplots showing original image, heatmap, and overlay
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get target layer for GradCAM - with multiple fallback options
    target_layer = None
    layer_name = "unknown"
    
    if hasattr(model.backbone, 'blocks'):  # ViT/DINO
        try:
            # Try different layer options for ViT
            if hasattr(model.backbone.blocks[-1], 'norm1'):
                target_layer = model.backbone.blocks[-1].norm1
                layer_name = "blocks[-1].norm1"
            elif hasattr(model.backbone.blocks[-1], 'norm2'):
                target_layer = model.backbone.blocks[-1].norm2
                layer_name = "blocks[-1].norm2"
            else:
                target_layer = model.backbone.blocks[-1]
                layer_name = "blocks[-1]"
        except:
            target_layer = model.backbone.blocks[-1]
            layer_name = "blocks[-1]"
            
    elif hasattr(model.backbone, 'stages'):  # ConvNeXt
        target_layer = model.backbone.stages[-1]
        layer_name = "stages[-1]"
        
    elif hasattr(model.backbone, 'vision_model'):  # MedSigLIP
        try:
            if hasattr(model.backbone.vision_model.encoder, 'layers'):
                if hasattr(model.backbone.vision_model.encoder.layers[-1], 'layer_norm1'):
                    target_layer = model.backbone.vision_model.encoder.layers[-1].layer_norm1
                    layer_name = "encoder.layers[-1].layer_norm1"
                else:
                    target_layer = model.backbone.vision_model.encoder.layers[-1]
                    layer_name = "encoder.layers[-1]"
        except:
            target_layer = model.backbone.vision_model
            layer_name = "vision_model"
    
    # Final fallback
    if target_layer is None:
        print("Warning: Could not identify optimal target layer. Using model.backbone")
        target_layer = model.backbone
        layer_name = "backbone"
    
    print(f"Using target layer: {layer_name}")
    
    gradcam = GradCAM(model, target_layer)
    
    # Denormalization
    mean = np.array(Config.mean).reshape(1, 1, 3)
    std = np.array(Config.std).reshape(1, 1, 3)
    
    # Group samples by class
    class_samples = {i: [] for i in range(len(class_names))}
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        label = sample['label'].item()
        
        if len(class_samples[label]) < samples_per_class:
            class_samples[label].append(idx)
        
        # Stop if we have enough samples for all classes
        if all(len(samples) >= samples_per_class for samples in class_samples.values()):
            break
    
    # Visualize each class
    for class_idx, class_name in enumerate(class_names):
        print(f"\nProcessing class: {class_name}")
        
        sample_indices = class_samples[class_idx]
        n_samples = len(sample_indices)
        
        if n_samples == 0:
            print(f"No samples found for class {class_name}")
            continue
        
        # Create figure
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for row_idx, sample_idx in enumerate(sample_indices):
            sample = dataset[sample_idx]
            image_tensor = sample['pixel_values'].unsqueeze(0).to(device)
            true_label = sample['label'].item()
            
            # Get original image path for display
            img_name = dataset.image_ids[sample_idx]
            img_dir = dataset.img_dirs[sample_idx]
            img_path = os.path.join(Config.data_root, img_dir, img_name)
            
            try:
                original_img = np.array(Image.open(img_path).convert('RGB'))
            except:
                # Fallback: denormalize tensor
                img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()
                original_img = (img_np * std + mean) * 255
                original_img = original_img.astype(np.uint8)
            
            # Resize to tensor size for consistency
            original_img = cv2.resize(original_img, 
                                     (image_tensor.shape[-1], image_tensor.shape[-2]))
            
            # Generate GradCAM
            cam, pred_class, probs = gradcam.generate_cam(image_tensor, target_class=class_idx)
            
            # Overlay
            overlaid = overlay_heatmap_on_image(
                original_img / 255.0, 
                cam, 
                alpha=0.5
            )
            
            # Plot
            axes[row_idx, 0].imshow(original_img)
            axes[row_idx, 0].set_title(
                f"Original\nTrue: {class_names[true_label]}\nPred: {class_names[pred_class]} ({probs[pred_class]:.2%})",
                fontsize=9
            )
            axes[row_idx, 0].axis('off')
            
            axes[row_idx, 1].imshow(cam, cmap='jet')
            axes[row_idx, 1].set_title("GradCAM Heatmap", fontsize=9)
            axes[row_idx, 1].axis('off')
            
            axes[row_idx, 2].imshow(overlaid)
            axes[row_idx, 2].set_title("Overlay", fontsize=9)
            axes[row_idx, 2].axis('off')
        
        plt.suptitle(f"GradCAM Visualization - Class: {class_name}", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"gradcam_{class_name}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    
    # Clean up hooks
    gradcam.remove_hooks()


def analyze_prediction_confidence(model, dataloader, class_names, output_dir, device='cuda'):
    """
    Analyze prediction confidence across classes
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing predictions"):
            images = batch['pixel_values'].to(device)
            labels = batch['label']
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    # 1. Confidence distribution per class
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    axes = axes.flatten()
    
    for class_idx, class_name in enumerate(class_names):
        mask = all_labels == class_idx
        if mask.sum() == 0:
            continue
        
        class_probs = all_probs[mask, class_idx]
        
        axes[class_idx].hist(class_probs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[class_idx].axvline(class_probs.mean(), color='red', linestyle='--', 
                               label=f'Mean: {class_probs.mean():.3f}')
        axes[class_idx].set_title(f"{class_name}\n(n={mask.sum()})", fontsize=10)
        axes[class_idx].set_xlabel("Confidence", fontsize=8)
        axes[class_idx].set_ylabel("Count", fontsize=8)
        axes[class_idx].legend(fontsize=7)
        axes[class_idx].grid(alpha=0.3)
    
    plt.suptitle("Prediction Confidence Distribution per Class", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_distribution.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion matrix with confidence
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("True Label", fontsize=10)
    axes[0].set_xlabel("Predicted Label", fontsize=10)
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("True Label", fontsize=10)
    axes[1].set_xlabel("Predicted Label", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Per-class metrics
    from sklearn.metrics import classification_report
    
    report = classification_report(all_labels, all_preds, 
                                   target_names=class_names, 
                                   output_dict=True)
    
    # Extract metrics
    classes = class_names
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]
    support = [report[c]['support'] for c in classes]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Precision
    axes[0, 0].barh(classes, precision, color='skyblue')
    axes[0, 0].set_xlabel("Precision", fontsize=10)
    axes[0, 0].set_title("Precision per Class", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Recall
    axes[0, 1].barh(classes, recall, color='lightcoral')
    axes[0, 1].set_xlabel("Recall", fontsize=10)
    axes[0, 1].set_title("Recall per Class", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # F1-Score
    axes[1, 0].barh(classes, f1, color='lightgreen')
    axes[1, 0].set_xlabel("F1-Score", fontsize=10)
    axes[1, 0].set_title("F1-Score per Class", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Support
    axes[1, 1].barh(classes, support, color='plum')
    axes[1, 1].set_xlabel("Number of Samples", fontsize=10)
    axes[1, 1].set_title("Support per Class", fontsize=11, fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_metrics.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


def visualize_misclassifications(model, dataset, class_names, output_dir, 
                                 max_samples=20, device='cuda'):
    """
    Visualize top misclassified samples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    # Get target layer for GradCAM - with fallback options
    target_layer = None
    if hasattr(model.backbone, 'blocks'):
        try:
            if hasattr(model.backbone.blocks[-1], 'norm1'):
                target_layer = model.backbone.blocks[-1].norm1
            else:
                target_layer = model.backbone.blocks[-1]
        except:
            target_layer = model.backbone.blocks[-1]
    elif hasattr(model.backbone, 'stages'):
        target_layer = model.backbone.stages[-1]
    elif hasattr(model.backbone, 'vision_model'):
        try:
            if hasattr(model.backbone.vision_model.encoder, 'layers'):
                target_layer = model.backbone.vision_model.encoder.layers[-1]
        except:
            target_layer = model.backbone.vision_model
    
    if target_layer is None:
        target_layer = model.backbone
    
    gradcam = GradCAM(model, target_layer)
    
    # Find misclassifications
    misclassified = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Finding misclassifications"):
            sample = dataset[idx]
            image = sample['pixel_values'].unsqueeze(0).to(device)
            true_label = sample['label'].item()
            
            output = model(image)
            pred_label = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, pred_label].item()
            
            if pred_label != true_label:
                misclassified.append({
                    'idx': idx,
                    'true': true_label,
                    'pred': pred_label,
                    'confidence': confidence
                })
    
    # Sort by confidence (most confident mistakes first)
    misclassified = sorted(misclassified, key=lambda x: x['confidence'], reverse=True)
    misclassified = misclassified[:max_samples]
    
    print(f"\nFound {len(misclassified)} misclassifications")
    
    # Visualize
    n_samples = min(len(misclassified), max_samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, mis in enumerate(misclassified[:n_samples]):
        sample = dataset[mis['idx']]
        image_tensor = sample['pixel_values'].unsqueeze(0).to(device)
        
        # Get original image
        img_name = dataset.image_ids[mis['idx']]
        img_dir = dataset.img_dirs[mis['idx']]
        img_path = os.path.join(Config.data_root, img_dir, img_name)
        
        try:
            original_img = np.array(Image.open(img_path).convert('RGB'))
        except:
            mean = np.array(Config.mean).reshape(1, 1, 3)
            std = np.array(Config.std).reshape(1, 1, 3)
            img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()
            original_img = (img_np * std + mean) * 255
            original_img = original_img.astype(np.uint8)
        
        original_img = cv2.resize(original_img, 
                                 (image_tensor.shape[-1], image_tensor.shape[-2]))
        
        # Generate GradCAM for predicted class
        cam, _, probs = gradcam.generate_cam(image_tensor, target_class=mis['pred'])
        
        overlaid = overlay_heatmap_on_image(original_img / 255.0, cam, alpha=0.5)
        
        # Plot
        axes[row_idx, 0].imshow(original_img)
        axes[row_idx, 0].set_title(
            f"True: {class_names[mis['true']]}\n"
            f"Pred: {class_names[mis['pred']]} ({mis['confidence']:.2%})",
            fontsize=9, color='red'
        )
        axes[row_idx, 0].axis('off')
        
        axes[row_idx, 1].imshow(cam, cmap='jet')
        axes[row_idx, 1].set_title("GradCAM\n(Predicted Class)", fontsize=9)
        axes[row_idx, 1].axis('off')
        
        axes[row_idx, 2].imshow(overlaid)
        axes[row_idx, 2].set_title("Overlay", fontsize=9)
        axes[row_idx, 2].axis('off')
    
    plt.suptitle("Top Misclassifications (Sorted by Confidence)", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "misclassifications.png"), 
                dpi=200, bbox_inches='tight')
    plt.close()
    
    # Clean up hooks
    gradcam.remove_hooks()


def main():
    """Main visualization pipeline"""
    
    print("="*80)
    print("WBC Classifier - Attention & Prediction Visualization")
    print("="*80)
    
    # Create output directories
    output_base = "visualizations/attention"
    os.makedirs(output_base, exist_ok=True)
    
    gradcam_dir = os.path.join(output_base, "gradcam_per_class")
    analysis_dir = os.path.join(output_base, "analysis")
    misclass_dir = os.path.join(output_base, "misclassifications")
    
    os.makedirs(gradcam_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(misclass_dir, exist_ok=True)
    
    # Load config
    class_to_idx = {cls: idx for idx, cls in enumerate(Config.class_names)}
    
    # Load model
    print("\nLoading model...")
    model = WBCClassifier(
        Config.model_type,
        Config.num_classes,
        config=Config
    ).to(Config.device)
    
    checkpoint_path = os.path.join(Config.save_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=Config.device, weights_only=False)
    
    if 'model_ema' in checkpoint and Config.use_ema:
        print("Using EMA model")
        model.load_state_dict(checkpoint['model_ema'])
    else:
        model.load_state_dict(checkpoint['model'])
    
    print(f"Loaded model with F1: {checkpoint['f1']:.4f}")
    
    # Load evaluation dataset
    print("\nLoading evaluation dataset...")
    eval_dataset = WBCDataset(
        os.path.join(Config.data_root, Config.eval_csv),
        Config.eval_img_dir,
        class_to_idx,
        transform=get_valid_transforms(Config.final_image_size),
        is_test=False
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=Config.batch_size * 2,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(eval_dataset)} samples")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)
    
    # 1. GradCAM per class
    print("\n1. GradCAM visualization per class")
    visualize_predictions_per_class(
        model, eval_dataset, Config.class_names,
        gradcam_dir, samples_per_class=5, device=Config.device
    )
    
    # 2. Prediction confidence analysis
    print("\n2. Analyzing prediction confidence")
    analyze_prediction_confidence(
        model, eval_loader, Config.class_names,
        analysis_dir, device=Config.device
    )
    
    # 3. Misclassification analysis
    print("\n3. Visualizing misclassifications")
    visualize_misclassifications(
        model, eval_dataset, Config.class_names,
        misclass_dir, max_samples=15, device=Config.device
    )
    
    print("\n" + "="*80)
    print(f"All visualizations saved to: {output_base}")
    print(f"  - GradCAM per class: {gradcam_dir}")
    print(f"  - Analysis plots: {analysis_dir}")
    print(f"  - Misclassifications: {misclass_dir}")
    print("="*80)


if __name__ == "__main__":
    main()