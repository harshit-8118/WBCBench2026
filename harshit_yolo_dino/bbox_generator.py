import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology, segmentation
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu, gaussian
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# AUTOMATIC BOUNDING BOX GENERATION FOR WBC IMAGES
# ============================================================================
# This script automatically generates bounding box annotations for:
# 1. Nucleus Detection (using color thresholding + morphology)
# 2. WBC Detection (using watershed segmentation)
#
# NO MANUAL LABELING REQUIRED!
# 
# Methods used:
# - Color space analysis (HSV for nucleus detection)
# - Otsu thresholding
# - Morphological operations
# - Watershed segmentation
# - Connected component analysis
# ============================================================================

class AutoBBoxGenerator:
    """Automatically generate bounding boxes for nucleus and WBC detection"""
    
    def __init__(self, output_dir='auto_annotations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create YOLO format directories
        self.yolo_dir = self.output_dir / 'yolo_format'
        (self.yolo_dir / 'images' / 'train').mkdir(exist_ok=True, parents=True)
        (self.yolo_dir / 'images' / 'val').mkdir(exist_ok=True, parents=True)
        (self.yolo_dir / 'labels' / 'train').mkdir(exist_ok=True, parents=True)
        (self.yolo_dir / 'labels' / 'val').mkdir(exist_ok=True, parents=True)
        
        # Visualization directory
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        
        print(f"✓ Output directory created: {self.output_dir}")
    
    def detect_nucleus_color_based(self, image):
        """
        Detect nucleus using color thresholding
        Nucleus typically appears as dark purple/blue regions
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define purple/blue range for nucleus (Giemsa/Wright stain)
        # Adjust these ranges based on your staining
        lower_purple = np.array([120, 30, 30])
        upper_purple = np.array([160, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
        
        # Also try dark regions (alternative)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, dark_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask, dark_mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by area (nucleus should be reasonably sized)
            if 500 < area < 50000:  # Adjust based on image resolution
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by aspect ratio (nucleus should be roughly circular)
                aspect_ratio = w / h if h > 0 else 0
                if 0.5 < aspect_ratio < 2.0:
                    bboxes.append([x, y, x+w, y+h])
        
        return bboxes, combined_mask
    
    def detect_nucleus_watershed(self, image):
        """
        Advanced nucleus detection using watershed segmentation
        More accurate but slightly slower
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Denoise
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu thresholding
        thresh = threshold_otsu(denoised)
        binary = denoised < thresh  # Nucleus is darker
        
        # Remove small objects
        binary = morphology.remove_small_objects(binary, min_size=300)
        
        # Fill holes
        binary = ndimage.binary_fill_holes(binary)
        
        # Distance transform
        distance = ndimage.distance_transform_edt(binary)
        
        # Find peaks (nucleus centers)
        coords = peak_local_max(distance, min_distance=20, labels=binary)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers = ndimage.label(mask)[0]
        
        # Watershed
        labels = segmentation.watershed(-distance, markers, mask=binary)
        
        # Extract bounding boxes
        bboxes = []
        for region in measure.regionprops(labels):
            if region.area > 500:  # Minimum area threshold
                minr, minc, maxr, maxc = region.bbox
                bboxes.append([minc, minr, maxc, maxr])
        
        return bboxes, (labels > 0).astype(np.uint8) * 255
    
    def detect_wbc_cell(self, image):
        """
        Detect entire WBC (cell + cytoplasm)
        Typically larger region encompassing the whole cell
        """
        # Convert to LAB color space (better for cell detection)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 10
        )
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # WBC should be larger than nucleus alone
            if 2000 < area < 100000:  # Adjust based on resolution
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.4 < aspect_ratio < 2.5:
                    bboxes.append([x, y, x+w, y+h])
        
        return bboxes, binary
    
    def detect_wbc_edge_based(self, image):
        """
        Alternative WBC detection using edge detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 30, 100)
        
        # Dilate edges to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 2000 < area < 100000:
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append([x, y, x+w, y+h])
        
        return bboxes, dilated
    
    def process_single_image(self, image_path, method='auto'):
        """
        Process a single image and generate bounding boxes
        
        Args:
            image_path: Path to image
            method: 'auto', 'color', 'watershed', 'edge'
        
        Returns:
            dict with nucleus and WBC bounding boxes
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error loading {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Detect nucleus
        if method == 'watershed':
            nucleus_bboxes, nucleus_mask = self.detect_nucleus_watershed(image_rgb)
        else:
            nucleus_bboxes, nucleus_mask = self.detect_nucleus_color_based(image_rgb)
        
        # Detect WBC
        wbc_bboxes, wbc_mask = self.detect_wbc_cell(image_rgb)
        
        # If no WBC detected, try edge-based method
        if len(wbc_bboxes) == 0:
            wbc_bboxes, wbc_mask = self.detect_wbc_edge_based(image_rgb)
        
        result = {
            'image_path': str(image_path),
            'image_name': Path(image_path).name,
            'image_size': (w, h),
            'nucleus_bboxes': nucleus_bboxes,
            'wbc_bboxes': wbc_bboxes,
            'nucleus_mask': nucleus_mask,
            'wbc_mask': wbc_mask
        }
        
        return result
    
    def visualize_detections(self, image_path, result, save_path=None):
        """Visualize detected bounding boxes"""
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Nucleus mask
        axes[0, 1].imshow(result['nucleus_mask'], cmap='gray')
        axes[0, 1].set_title('Nucleus Mask', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # WBC mask
        axes[0, 2].imshow(result['wbc_mask'], cmap='gray')
        axes[0, 2].set_title('WBC Mask', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Nucleus bboxes
        img_nucleus = image_rgb.copy()
        for bbox in result['nucleus_bboxes']:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_nucleus, (x1, y1), (x2, y2), (255, 0, 0), 2)
        axes[1, 0].imshow(img_nucleus)
        axes[1, 0].set_title(f'Nucleus Detection ({len(result["nucleus_bboxes"])})', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # WBC bboxes
        img_wbc = image_rgb.copy()
        for bbox in result['wbc_bboxes']:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_wbc, (x1, y1), (x2, y2), (0, 255, 0), 2)
        axes[1, 1].imshow(img_wbc)
        axes[1, 1].set_title(f'WBC Detection ({len(result["wbc_bboxes"])})', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Combined
        img_combined = image_rgb.copy()
        for bbox in result['nucleus_bboxes']:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_combined, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for bbox in result['wbc_bboxes']:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
        axes[1, 2].imshow(img_combined)
        axes[1, 2].set_title('Combined (Red=Nucleus, Green=WBC)', 
                            fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def convert_to_yolo_format(self, bbox, image_width, image_height):
        """
        Convert [x1, y1, x2, y2] to YOLO format [x_center, y_center, width, height]
        All values normalized to [0, 1]
        """
        x1, y1, x2, y2 = bbox
        
        x_center = ((x1 + x2) / 2) / image_width
        y_center = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        
        return [x_center, y_center, width, height]
    
    def save_yolo_annotations(self, result, output_path, task='nucleus'):
        """
        Save annotations in YOLO format
        Format: <class> <x_center> <y_center> <width> <height>
        """
        w, h = result['image_size']
        
        bboxes = result['nucleus_bboxes'] if task == 'nucleus' else result['wbc_bboxes']
        
        with open(output_path, 'w') as f:
            for bbox in bboxes:
                yolo_bbox = self.convert_to_yolo_format(bbox, w, h)
                # Class 0 for single-class detection
                line = f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n"
                f.write(line)
    
    def process_dataset(self, image_dir, val_split=0.2, method='auto', 
                       visualize_samples=10, task='both'):
        """
        Process entire dataset and generate YOLO annotations
        
        Args:
            image_dir: Directory containing images
            val_split: Fraction for validation set
            method: Detection method
            visualize_samples: Number of samples to visualize
            task: 'nucleus', 'wbc', or 'both'
        """
        print("\n" + "="*80)
        print("AUTOMATIC BOUNDING BOX GENERATION")
        print("="*80)
        
        # Get all images
        image_paths = list(Path(image_dir).glob("*.jpg")) + \
                     list(Path(image_dir).glob("*.png")) + \
                     list(Path(image_dir).glob("*.jpeg"))
        
        print(f"\nFound {len(image_paths)} images")
        
        if len(image_paths) == 0:
            print("No images found!")
            return
        
        # Split into train/val
        np.random.shuffle(image_paths)
        split_idx = int(len(image_paths) * (1 - val_split))
        train_images = image_paths[:split_idx]
        val_images = image_paths[split_idx:]
        
        print(f"Train: {len(train_images)}, Val: {len(val_images)}")
        
        # Process images
        all_results = []
        
        for split_name, image_list in [('train', train_images), ('val', val_images)]:
            print(f"\nProcessing {split_name} set...")
            
            for img_path in tqdm(image_list):
                # Generate annotations
                result = self.process_single_image(img_path, method=method)
                
                if result is None:
                    continue
                
                all_results.append(result)
                
                # Copy image
                img_name = img_path.name
                dest_img = self.yolo_dir / 'images' / split_name / img_name
                import shutil
                shutil.copy(img_path, dest_img)
                
                # Save annotations
                label_name = img_path.stem + '.txt'
                
                if task in ['nucleus', 'both']:
                    nucleus_label = self.yolo_dir / 'labels' / split_name / label_name
                    self.save_yolo_annotations(result, nucleus_label, task='nucleus')
                
                # For WBC, we'd create a separate dataset
                # Or use different class IDs in the same file
        
        # Visualize samples
        print(f"\nVisualizing {visualize_samples} samples...")
        for i, result in enumerate(all_results[:visualize_samples]):
            save_path = self.output_dir / 'visualizations' / f"sample_{i:03d}.png"
            self.visualize_detections(result['image_path'], result, save_path)
        
        # Create dataset.yaml
        self.create_dataset_yaml(task)
        
        # Save statistics
        self.save_statistics(all_results)
        
        print("\n" + "="*80)
        print("GENERATION COMPLETE")
        print("="*80)
        print(f"\nYOLO dataset saved to: {self.yolo_dir}")
        print(f"Visualizations saved to: {self.output_dir / 'visualizations'}")
        
        return all_results
    
    def create_dataset_yaml(self, task='nucleus'):
        """Create YOLO dataset configuration file"""
        yaml_content = f"""# Auto-generated YOLO dataset configuration
path: {self.yolo_dir.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: {task}

# Image size
imgsz: 640
"""
        
        yaml_path = self.yolo_dir / f'{task}_dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✓ Created dataset YAML: {yaml_path}")
    
    def save_statistics(self, results):
        """Save detection statistics"""
        stats = {
            'total_images': len(results),
            'avg_nucleus_per_image': np.mean([len(r['nucleus_bboxes']) for r in results]),
            'avg_wbc_per_image': np.mean([len(r['wbc_bboxes']) for r in results]),
            'images_with_nucleus': sum(1 for r in results if len(r['nucleus_bboxes']) > 0),
            'images_with_wbc': sum(1 for r in results if len(r['wbc_bboxes']) > 0),
        }
        
        stats_path = self.output_dir / 'statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\nDetection Statistics:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Avg nucleus/image: {stats['avg_nucleus_per_image']:.2f}")
        print(f"  Avg WBC/image: {stats['avg_wbc_per_image']:.2f}")
        print(f"  Images with nucleus: {stats['images_with_nucleus']}")
        print(f"  Images with WBC: {stats['images_with_wbc']}")


def main():
    """Main execution"""
    print("="*80)
    print("AUTOMATIC BOUNDING BOX GENERATOR FOR WBC IMAGES")
    print("="*80)
    print("\nThis script will:")
    print("  1. Automatically detect nucleus and WBC regions")
    print("  2. Generate bounding box annotations")
    print("  3. Create YOLO format dataset")
    print("  4. NO MANUAL LABELING REQUIRED!")
    print("="*80)
    
    # Configuration
    IMAGE_DIR = "/data/data/WBCBench/wbc-bench-2026/phase2/train"
    OUTPUT_DIR = "auto_annotations"
    
    # Initialize generator
    generator = AutoBBoxGenerator(output_dir=OUTPUT_DIR)
    
    # Process dataset
    results = generator.process_dataset(
        image_dir=IMAGE_DIR,
        val_split=0.2,
        method='auto',  # 'auto', 'watershed', 'color', 'edge'
        visualize_samples=10,
        task='nucleus'  # 'nucleus', 'wbc', or 'both'
    )
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Review visualizations in auto_annotations/visualizations/")
    print("2. Check if detection quality is acceptable")
    print("3. Train YOLO model:")
    print(f"\n   yolo task=detect mode=train \\")
    print(f"        data={OUTPUT_DIR}/yolo_format/nucleus_dataset.yaml \\")
    print(f"        model=yolo11n.pt \\")
    print(f"        epochs=100 \\")
    print(f"        imgsz=640")
    print("\n4. Use trained model for analysis!")
    print("="*80)


if __name__ == "__main__":
    main()