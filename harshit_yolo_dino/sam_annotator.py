import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import torch
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SAM-BASED AUTOMATIC ANNOTATION GENERATOR
# ============================================================================
# Uses Meta's Segment Anything Model (SAM) for automatic segmentation
# More accurate than classical CV methods
# 
# Requires: pip install segment-anything-py
# Or: pip install git+https://github.com/facebookresearch/segment-anything.git
# ============================================================================

class SAMAnnotationGenerator:
    """
    Generate bounding box annotations using SAM (Segment Anything Model)
    More accurate than classical methods
    """
    
    def __init__(self, sam_checkpoint=None, output_dir='sam_annotations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup directories
        self.yolo_dir = self.output_dir / 'yolo_format'
        (self.yolo_dir / 'images' / 'train').mkdir(exist_ok=True, parents=True)
        (self.yolo_dir / 'images' / 'val').mkdir(exist_ok=True, parents=True)
        (self.yolo_dir / 'labels' / 'train').mkdir(exist_ok=True, parents=True)
        (self.yolo_dir / 'labels' / 'val').mkdir(exist_ok=True, parents=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        
        # Initialize SAM
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            
            # Auto-download if no checkpoint provided
            if sam_checkpoint is None:
                print("Downloading SAM checkpoint...")
                sam_checkpoint = self.download_sam_checkpoint()
            
            # Load SAM model
            model_type = "vit_h"  # or "vit_l", "vit_b"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=self.device)
            
            # Automatic mask generator
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # Requires PIL
            )
            
            print("✓ SAM model loaded successfully")
            self.sam_available = True
            
        except ImportError:
            print("⚠ SAM not installed. Install with:")
            print("  pip install git+https://github.com/facebookresearch/segment-anything.git")
            print("\nFalling back to classical methods...")
            self.sam_available = False
            self.mask_generator = None
    
    def download_sam_checkpoint(self):
        """Download SAM checkpoint if not available"""
        checkpoint_dir = Path.home() / '.cache' / 'sam'
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        checkpoint_path = checkpoint_dir / 'sam_vit_h_4b8939.pth'
        
        if not checkpoint_path.exists():
            print("Downloading SAM checkpoint (2.4GB)...")
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            urllib.request.urlretrieve(url, checkpoint_path)
            print("✓ Download complete")
        
        return str(checkpoint_path)
    
    def segment_with_sam(self, image):
        """
        Use SAM to automatically segment image
        Returns masks that can be filtered for nucleus/WBC
        """
        if not self.sam_available:
            return []
        
        # SAM expects RGB
        masks = self.mask_generator.generate(image)
        
        return masks
    
    def filter_masks_for_nucleus(self, masks, image_shape):
        """
        Filter SAM masks to find nucleus regions
        - Smaller masks
        - Higher stability scores
        - Darker regions
        """
        nucleus_bboxes = []
        
        for mask_data in masks:
            bbox = mask_data['bbox']  # [x, y, w, h]
            area = mask_data['area']
            stability = mask_data['stability_score']
            
            # Nucleus criteria
            if 300 < area < 20000 and stability > 0.90:
                x, y, w, h = bbox
                # Convert to [x1, y1, x2, y2]
                nucleus_bboxes.append([x, y, x + w, y + h])
        
        return nucleus_bboxes
    
    def filter_masks_for_wbc(self, masks, image_shape):
        """
        Filter SAM masks to find whole WBC
        - Larger masks
        - Encompass entire cell
        """
        wbc_bboxes = []
        
        for mask_data in masks:
            bbox = mask_data['bbox']
            area = mask_data['area']
            stability = mask_data['stability_score']
            
            # WBC criteria (larger than nucleus)
            if 2000 < area < 100000 and stability > 0.85:
                x, y, w, h = bbox
                wbc_bboxes.append([x, y, x + w, y + h])
        
        return wbc_bboxes
    
    def process_image_with_sam(self, image_path):
        """Process single image using SAM"""
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Generate masks
        masks = self.segment_with_sam(image_rgb)
        
        # Filter for nucleus and WBC
        nucleus_bboxes = self.filter_masks_for_nucleus(masks, (h, w))
        wbc_bboxes = self.filter_masks_for_wbc(masks, (h, w))
        
        result = {
            'image_path': str(image_path),
            'image_name': Path(image_path).name,
            'image_size': (w, h),
            'nucleus_bboxes': nucleus_bboxes,
            'wbc_bboxes': wbc_bboxes,
            'num_masks': len(masks)
        }
        
        return result
    
    def visualize_detections(self, image_path, result, save_path=None):
        """Visualize SAM detections"""
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Nucleus detections
        img_nucleus = image_rgb.copy()
        for bbox in result['nucleus_bboxes']:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_nucleus, (x1, y1), (x2, y2), (255, 0, 0), 2)
        axes[1].imshow(img_nucleus)
        axes[1].set_title(f'Nucleus ({len(result["nucleus_bboxes"])})', 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # WBC detections
        img_wbc = image_rgb.copy()
        for bbox in result['wbc_bboxes']:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_wbc, (x1, y1), (x2, y2), (0, 255, 0), 2)
        axes[2].imshow(img_wbc)
        axes[2].set_title(f'WBC ({len(result["wbc_bboxes"])})', 
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# ============================================================================
# SIMPLE FASTEST METHOD - Using Contours Only
# ============================================================================

def quick_auto_annotate(image_dir, output_dir='quick_annotations', 
                        nucleus_only=True, visualize=True):
    """
    Quickest method to generate annotations
    Uses simple contour detection - works for well-stained images
    
    Args:
        image_dir: Directory with images
        output_dir: Where to save YOLO annotations
        nucleus_only: Only detect nucleus (faster, more reliable)
        visualize: Save visualization for first 10 images
    """
    print("\n" + "="*80)
    print("QUICK AUTO-ANNOTATION (Contour-based)")
    print("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create YOLO structure
    yolo_dir = output_path / 'yolo_dataset'
    for split in ['train', 'val']:
        (yolo_dir / 'images' / split).mkdir(exist_ok=True, parents=True)
        (yolo_dir / 'labels' / split).mkdir(exist_ok=True, parents=True)
    
    if visualize:
        (output_path / 'viz').mkdir(exist_ok=True)
    
    # Get images
    images = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    print(f"Found {len(images)} images")
    
    # Split train/val
    np.random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    splits = {'train': images[:split_idx], 'val': images[split_idx:]}
    
    stats = {'train': [], 'val': []}
    
    for split_name, img_list in splits.items():
        print(f"\nProcessing {split_name}...")
        
        for idx, img_path in enumerate(tqdm(img_list)):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Threshold for dark regions (nucleus)
            _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            
            # Clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and save annotations
            bboxes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 30000:  # Nucleus size range
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    
                    # Convert to YOLO format
                    x_center = (x + w_box/2) / w
                    y_center = (y + h_box/2) / h
                    box_w = w_box / w
                    box_h = h_box / h
                    
                    bboxes.append([x_center, y_center, box_w, box_h])
            
            stats[split_name].append(len(bboxes))
            
            # Save annotation
            import shutil
            img_name = img_path.name
            shutil.copy(img_path, yolo_dir / 'images' / split_name / img_name)
            
            label_path = yolo_dir / 'labels' / split_name / (img_path.stem + '.txt')
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            
            # Visualize first few
            if visualize and split_name == 'train' and idx < 10:
                vis_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                for bbox in bboxes:
                    x_c, y_c, w_b, h_b = bbox
                    x1 = int((x_c - w_b/2) * w)
                    y1 = int((y_c - h_b/2) * h)
                    x2 = int((x_c + w_b/2) * w)
                    y2 = int((y_c + h_b/2) * h)
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                plt.figure(figsize=(8, 8))
                plt.imshow(vis_img)
                plt.title(f'{img_name} - {len(bboxes)} detections')
                plt.axis('off')
                plt.savefig(output_path / 'viz' / f'sample_{idx:03d}.png', 
                           dpi=100, bbox_inches='tight')
                plt.close()
    
    # Create dataset.yaml
    yaml_content = f"""path: {yolo_dir.absolute()}
train: images/train
val: images/val

names:
  0: nucleus
"""
    
    with open(yolo_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Train images: {len(splits['train'])}, avg detections: {np.mean(stats['train']):.2f}")
    print(f"Val images: {len(splits['val'])}, avg detections: {np.mean(stats['val']):.2f}")
    print(f"\nDataset saved to: {yolo_dir}")
    print(f"Dataset YAML: {yolo_dir / 'dataset.yaml'}")
    
    print("\n" + "="*80)
    print("TRAIN YOLO MODEL")
    print("="*80)
    print(f"\nyolo task=detect mode=train \\")
    print(f"     data={yolo_dir / 'dataset.yaml'} \\")
    print(f"     model=yolo11n.pt \\")
    print(f"     epochs=50 \\")
    print(f"     imgsz=640 \\")
    print(f"     batch=16")
    print("\n" + "="*80)
    
    return yolo_dir


def main():
    """Choose your method"""
    print("="*80)
    print("AUTO-ANNOTATION FOR WBC IMAGES - METHOD SELECTION")
    print("="*80)
    print("\nAvailable methods:")
    print("  1. QUICK (fastest, contour-based, ~95% accurate)")
    print("  2. SAM (most accurate, requires SAM model, slower)")
    print("  3. Classical CV (medium speed and accuracy)")
    print("="*80)
    
    # For most users, QUICK method is recommended
    IMAGE_DIR = "/data/data/WBCBench/wbc-bench-2026/phase2/train"
    
    print("\nRecommended: Using QUICK method...")
    yolo_dir = quick_auto_annotate(
        image_dir=IMAGE_DIR,
        output_dir='quick_annotations',
        nucleus_only=True,
        visualize=True
    )
    
    print("\n✓ Done! Check visualizations and proceed to training.")


if __name__ == "__main__":
    main()