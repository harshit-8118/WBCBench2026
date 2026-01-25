#!/usr/bin/env python3
"""
WBC YOLO Training Script
Complete pipeline for training YOLO11 on WBC detection
"""

# Suppress matplotlib/tkinter warnings
import os
os.environ['MPLBACKEND'] = 'Agg'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import cv2
import numpy as np
import pandas as pd

# Set matplotlib backend BEFORE importing pyplot to avoid tkinter errors
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from tqdm import tqdm
import json
import yaml
import shutil
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
import torch

# ============================================================================
# COMPREHENSIVE YOLO TRAINING SCRIPT FOR WBC DETECTION
# ============================================================================
# Features:
# 1. Dataset analysis (image sizes, label distribution)
# 2. Data preprocessing (filter multiple labels, keep max area)
# 3. Visualization of ground truth
# 4. YOLO training with optimal hyperparameters
# 5. Validation with IoU metrics
# 6. Model export and documentation
# ============================================================================

class WBCYOLOTrainer:
    """Complete YOLO training pipeline for WBC detection"""
    
    def __init__(self, dataset_root, output_dir='wbc_yolo_output'):
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.output_dir / 'analysis').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'processed_data').mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        
        # Paths
        self.train_images = self.dataset_root / 'train' / 'images'
        self.train_labels = self.dataset_root / 'train' / 'labels'
        self.val_images = self.dataset_root / 'valid' / 'images'
        self.val_labels = self.dataset_root / 'valid' / 'labels'
        self.data_yaml = self.dataset_root / 'data.yaml'
        
        # Statistics storage
        self.stats = {
            'train': defaultdict(list),
            'val': defaultdict(list)
        }
        
        print("="*80)
        print("WBC YOLO TRAINER INITIALIZED")
        print("="*80)
        print(f"Dataset root: {self.dataset_root}")
        print(f"Output directory: {self.output_dir}")
        print("="*80)

    def analyze_dataset(self):
        """Comprehensive dataset analysis"""
        print("\n" + "="*80)
        print("STEP 1: DATASET ANALYSIS")
        print("="*80)
        
        for split_name, img_dir, label_dir in [
            ('train', self.train_images, self.train_labels),
            ('val', self.val_images, self.val_labels)
        ]:
            print(f"\nAnalyzing {split_name} set...")
            
            # Get all images
            image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            print(f"Found {len(image_files)} images")
            
            for img_path in tqdm(image_files, desc=f"Analyzing {split_name}"):
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                self.stats[split_name]['widths'].append(w)
                self.stats[split_name]['heights'].append(h)
                self.stats[split_name]['areas'].append(w * h)
                
                # Read labels
                label_path = label_dir / (img_path.stem + '.txt')
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        labels = f.readlines()
                    
                    self.stats[split_name]['num_labels'].append(len(labels))
                    
                    # Parse bboxes
                    for label in labels:
                        parts = label.strip().split()
                        if len(parts) == 5:
                            cls, x_c, y_c, w_box, h_box = map(float, parts)
                            
                            if cls == 0:  # WBC class
                                continue

                            # Absolute bbox dimensions
                            abs_w = w_box * w
                            abs_h = h_box * h
                            abs_area = abs_w * abs_h
                            
                            self.stats[split_name]['bbox_widths'].append(abs_w)
                            self.stats[split_name]['bbox_heights'].append(abs_h)
                            self.stats[split_name]['bbox_areas'].append(abs_area)
                            self.stats[split_name]['bbox_aspect_ratios'].append(abs_w / abs_h if abs_h > 0 else 0)
                else:
                    self.stats[split_name]['num_labels'].append(0)
        
        # Print statistics
        self._print_statistics()
        
        # Save statistics
        self._save_statistics()
        
        # Create visualizations
        self._create_analysis_plots()
    
    def _print_statistics(self):
        """Print dataset statistics"""
        print("\n" + "="*80)
        print("DATASET STATISTICS")
        print("="*80)
        
        for split_name in ['train', 'val']:
            print(f"\n{split_name.upper()} SET:")
            print("-" * 40)
            
            stats = self.stats[split_name]
            
            # Image statistics
            print(f"Total Images: {len(stats['widths'])}")
            print(f"\nImage Dimensions:")
            print(f"  Width:  {np.min(stats['widths'])} - {np.max(stats['widths'])} "
                  f"(mean: {np.mean(stats['widths']):.1f})")
            print(f"  Height: {np.min(stats['heights'])} - {np.max(stats['heights'])} "
                  f"(mean: {np.mean(stats['heights']):.1f})")
            
            # Label statistics
            print(f"\nLabels per Image:")
            print(f"  Min: {np.min(stats['num_labels'])}")
            print(f"  Max: {np.max(stats['num_labels'])}")
            print(f"  Mean: {np.mean(stats['num_labels']):.2f}")
            print(f"  Total labels: {sum(stats['num_labels'])}")
            
            # Bounding box statistics
            if stats['bbox_widths']:
                print(f"\nBounding Box Dimensions (pixels):")
                print(f"  Width:  {np.min(stats['bbox_widths']):.1f} - {np.max(stats['bbox_widths']):.1f} "
                      f"(mean: {np.mean(stats['bbox_widths']):.1f})")
                print(f"  Height: {np.min(stats['bbox_heights']):.1f} - {np.max(stats['bbox_heights']):.1f} "
                      f"(mean: {np.mean(stats['bbox_heights']):.1f})")
                print(f"  Area:   {np.min(stats['bbox_areas']):.1f} - {np.max(stats['bbox_areas']):.1f} "
                      f"(mean: {np.mean(stats['bbox_areas']):.1f})")
                print(f"  Aspect Ratio: {np.min(stats['bbox_aspect_ratios']):.2f} - "
                      f"{np.max(stats['bbox_aspect_ratios']):.2f} "
                      f"(mean: {np.mean(stats['bbox_aspect_ratios']):.2f})")
    
    def _save_statistics(self):
        """Save statistics to JSON"""
        stats_dict = {}
        
        for split_name in ['train', 'val']:
            stats_dict[split_name] = {
                'num_images': len(self.stats[split_name]['widths']),
                'image_width_min': int(np.min(self.stats[split_name]['widths'])),
                'image_width_max': int(np.max(self.stats[split_name]['widths'])),
                'image_width_mean': float(np.mean(self.stats[split_name]['widths'])),
                'image_height_min': int(np.min(self.stats[split_name]['heights'])),
                'image_height_max': int(np.max(self.stats[split_name]['heights'])),
                'image_height_mean': float(np.mean(self.stats[split_name]['heights'])),
                'labels_per_image_min': int(np.min(self.stats[split_name]['num_labels'])),
                'labels_per_image_max': int(np.max(self.stats[split_name]['num_labels'])),
                'labels_per_image_mean': float(np.mean(self.stats[split_name]['num_labels'])),
                'total_labels': int(sum(self.stats[split_name]['num_labels'])),
            }
            
            if self.stats[split_name]['bbox_widths']:
                stats_dict[split_name].update({
                    'bbox_width_min': float(np.min(self.stats[split_name]['bbox_widths'])),
                    'bbox_width_max': float(np.max(self.stats[split_name]['bbox_widths'])),
                    'bbox_width_mean': float(np.mean(self.stats[split_name]['bbox_widths'])),
                    'bbox_height_min': float(np.min(self.stats[split_name]['bbox_heights'])),
                    'bbox_height_max': float(np.max(self.stats[split_name]['bbox_heights'])),
                    'bbox_height_mean': float(np.mean(self.stats[split_name]['bbox_heights'])),
                    'bbox_area_min': float(np.min(self.stats[split_name]['bbox_areas'])),
                    'bbox_area_max': float(np.max(self.stats[split_name]['bbox_areas'])),
                    'bbox_area_mean': float(np.mean(self.stats[split_name]['bbox_areas'])),
                })
        
        save_path = self.output_dir / 'analysis' / 'dataset_statistics.json'
        with open(save_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        print(f"\n✓ Statistics saved to: {save_path}")
    
    def _create_analysis_plots(self):
        """Create comprehensive analysis plots"""
        print("\nCreating analysis visualizations...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        colors = {'train': '#2E86AB', 'val': '#A23B72'}
        
        # 1. Image size distribution
        ax1 = fig.add_subplot(gs[0, 0])
        for split_name, color in colors.items():
            ax1.hist(self.stats[split_name]['widths'], bins=30, alpha=0.6, 
                    label=f'{split_name} width', color=color)
        ax1.set_xlabel('Image Width (pixels)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Image Width Distribution')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        for split_name, color in colors.items():
            ax2.hist(self.stats[split_name]['heights'], bins=30, alpha=0.6,
                    label=f'{split_name} height', color=color)
        ax2.set_xlabel('Image Height (pixels)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Image Height Distribution')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 2. Labels per image
        ax3 = fig.add_subplot(gs[0, 2])
        for split_name, color in colors.items():
            ax3.hist(self.stats[split_name]['num_labels'], bins=range(0, 11), 
                    alpha=0.6, label=split_name, color=color)
        ax3.set_xlabel('Labels per Image')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Labels per Image Distribution')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 3. Bbox size distribution
        ax4 = fig.add_subplot(gs[0, 3])
        for split_name, color in colors.items():
            if self.stats[split_name]['bbox_areas']:
                ax4.hist(self.stats[split_name]['bbox_areas'], bins=50, 
                        alpha=0.6, label=split_name, color=color)
        ax4.set_xlabel('Bounding Box Area (pixels²)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Bbox Area Distribution')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 4. Bbox width/height
        ax5 = fig.add_subplot(gs[1, 0])
        for split_name, color in colors.items():
            if self.stats[split_name]['bbox_widths']:
                ax5.hist(self.stats[split_name]['bbox_widths'], bins=50,
                        alpha=0.6, label=f'{split_name} width', color=color)
        ax5.set_xlabel('Bbox Width (pixels)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Bbox Width Distribution')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        ax6 = fig.add_subplot(gs[1, 1])
        for split_name, color in colors.items():
            if self.stats[split_name]['bbox_heights']:
                ax6.hist(self.stats[split_name]['bbox_heights'], bins=50,
                        alpha=0.6, label=f'{split_name} height', color=color)
        ax6.set_xlabel('Bbox Height (pixels)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Bbox Height Distribution')
        ax6.legend()
        ax6.grid(alpha=0.3)
        
        # 5. Aspect ratio
        ax7 = fig.add_subplot(gs[1, 2])
        for split_name, color in colors.items():
            if self.stats[split_name]['bbox_aspect_ratios']:
                ax7.hist(self.stats[split_name]['bbox_aspect_ratios'], bins=50,
                        alpha=0.6, label=split_name, color=color)
        ax7.set_xlabel('Aspect Ratio (W/H)')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Bbox Aspect Ratio Distribution')
        ax7.legend()
        ax7.grid(alpha=0.3)
        
        # 6. Scatter: Width vs Height
        ax8 = fig.add_subplot(gs[1, 3])
        for split_name, color in colors.items():
            if self.stats[split_name]['bbox_widths']:
                ax8.scatter(self.stats[split_name]['bbox_widths'][:1000],
                           self.stats[split_name]['bbox_heights'][:1000],
                           alpha=0.3, s=10, label=split_name, color=color)
        ax8.set_xlabel('Bbox Width (pixels)')
        ax8.set_ylabel('Bbox Height (pixels)')
        ax8.set_title('Bbox Dimensions Scatter')
        ax8.legend()
        ax8.grid(alpha=0.3)
        
        # 7. Box plots
        ax9 = fig.add_subplot(gs[2, 0])
        data_to_plot = [self.stats['train']['widths'], self.stats['val']['widths']]
        bp = ax9.boxplot(data_to_plot, labels=['Train', 'Val'], patch_artist=True)
        for patch, color in zip(bp['boxes'], [colors['train'], colors['val']]):
            patch.set_facecolor(color)
        ax9.set_ylabel('Width (pixels)')
        ax9.set_title('Image Width Box Plot')
        ax9.grid(alpha=0.3)
        
        ax10 = fig.add_subplot(gs[2, 1])
        data_to_plot = [self.stats['train']['heights'], self.stats['val']['heights']]
        bp = ax10.boxplot(data_to_plot, labels=['Train', 'Val'], patch_artist=True)
        for patch, color in zip(bp['boxes'], [colors['train'], colors['val']]):
            patch.set_facecolor(color)
        ax10.set_ylabel('Height (pixels)')
        ax10.set_title('Image Height Box Plot')
        ax10.grid(alpha=0.3)
        
        # 8. Summary table
        ax11 = fig.add_subplot(gs[2, 2:])
        ax11.axis('tight')
        ax11.axis('off')
        
        summary_data = []
        for split_name in ['train', 'val']:
            stats = self.stats[split_name]
            summary_data.append([
                split_name.upper(),
                f"{len(stats['widths'])}",
                f"{np.mean(stats['widths']):.0f}x{np.mean(stats['heights']):.0f}",
                f"{np.mean(stats['num_labels']):.1f}",
                f"{np.mean(stats['bbox_areas']):.0f}" if stats['bbox_areas'] else "N/A"
            ])
        
        table = ax11.table(
            cellText=summary_data,
            colLabels=['Split', 'Images', 'Avg Size', 'Avg Labels', 'Avg Bbox Area'],
            cellLoc='center',
            loc='center',
            colWidths=[0.15, 0.15, 0.2, 0.2, 0.2]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        for i in range(5):
            table[(0, i)].set_facecolor('#E8E8E8')
            table[(0, i)].set_text_props(weight='bold')
        
        ax11.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('Dataset Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
        
        save_path = self.output_dir / 'analysis' / 'dataset_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Analysis plots saved to: {save_path}")
    
    def preprocess_labels(self, keep_max_area_only=True):
        """
        Preprocess labels: keep only the bbox with maximum area per image
        """
        print("\n" + "="*80)
        print("STEP 2: LABEL PREPROCESSING")
        print("="*80)
        
        if not keep_max_area_only:
            print("Skipping preprocessing - using all labels as-is")
            return
        
        print("Filtering: Keeping only bbox with maximum area per image")
        
        for split_name, label_dir in [('train', self.train_labels), ('val', self.val_labels)]:
            print(f"\nProcessing {split_name} labels...")
            
            label_files = list(label_dir.glob("*.txt"))
            filtered_count = 0
            
            for label_path in tqdm(label_files):
                with open(label_path, 'r') as f:
                    labels = f.readlines()
                
                if len(labels) <= 1:
                    continue  # No need to filter
                
                # Parse all boxes
                boxes = []
                for label in labels:
                    parts = label.strip().split()
                    if len(parts) == 5:
                        cls, x_c, y_c, w, h = map(float, parts)
                        if cls==0:
                            continue
                        area = w * h
                        boxes.append((area, label.strip()))
                
                if len(boxes) > 1:
                    # Sort by area and keep largest
                    boxes.sort(reverse=True, key=lambda x: x[0])
                    largest_box = boxes[0][1]
                    
                    # Overwrite with only largest box
                    with open(label_path, 'w') as f:
                        f.write(largest_box + '\n')
                    
                    filtered_count += 1
            
            print(f"✓ Filtered {filtered_count} files in {split_name}")
        
        print("\n✓ Label preprocessing complete")
    
    def visualize_ground_truth(self, num_samples=20):
        """Visualize ground truth annotations"""
        print("\n" + "="*80)
        print("STEP 3: GROUND TRUTH VISUALIZATION")
        print("="*80)
        
        for split_name, img_dir, label_dir in [
            ('train', self.train_images, self.train_labels),
            ('val', self.val_images, self.val_labels)
        ]:
            print(f"\nVisualizing {split_name} samples...")
            
            image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            
            # Randomly sample
            np.random.shuffle(image_files)
            samples = image_files[:num_samples]
            
            for idx, img_path in enumerate(tqdm(samples)):
                img = cv2.imread(str(img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                
                # Read label
                label_path = label_dir / (img_path.stem + '.txt')
                
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        labels = f.readlines()
                    
                    # Draw boxes
                    for label in labels:
                        parts = label.strip().split()
                        if len(parts) == 5:
                            cls, x_c, y_c, w_box, h_box = map(float, parts)
                            if cls == 0:  # WBC class
                                continue
                            
                            # Convert to pixel coordinates
                            x1 = int((x_c - w_box/2) * w)
                            y1 = int((y_c - h_box/2) * h)
                            x2 = int((x_c + w_box/2) * w)
                            y2 = int((y_c + h_box/2) * h)
                            
                            # Draw rectangle
                            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Add label
                            label_text = f"WBC: {w_box*w:.0f}x{h_box*h:.0f}"
                            cv2.putText(img_rgb, label_text, (x1, y1-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Save visualization
                plt.figure(figsize=(10, 10))
                plt.imshow(img_rgb)
                plt.title(f'{split_name.upper()}: {img_path.name} ({w}x{h})', 
                         fontsize=12, fontweight='bold')
                plt.axis('off')
                
                save_path = self.output_dir / 'visualizations' / f'{split_name}_gt_{idx:03d}.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
        
        print(f"\n✓ Ground truth visualizations saved to: {self.output_dir / 'visualizations'}")
    
    def create_optimal_data_yaml(self):
        """Create optimized data.yaml for training"""
        print("\n" + "="*80)
        print("STEP 4: DATA CONFIGURATION")
        print("="*80)
        
        # Calculate optimal image size based on dataset
        avg_width = np.mean(self.stats['train']['widths'])
        avg_height = np.mean(self.stats['train']['heights'])
        
        # Round to nearest multiple of 32 (YOLO requirement)
        optimal_size = int(max(avg_width, avg_height) // 32 * 32)
        
        # Common sizes: 320, 384, 416, 480, 512, 640
        if optimal_size < 384:
            optimal_size = 384
        elif optimal_size > 640:
            optimal_size = 640
        
        print(f"Average image size: {avg_width:.0f}x{avg_height:.0f}")
        print(f"Optimal training size: {optimal_size}x{optimal_size}")
        
        # Create new data.yaml
        data_config = {
            'path': str(self.dataset_root.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'names': {
                0: 'WBC'
            },
            'nc': 1,  # Number of classes
        }
        
        new_yaml_path = self.output_dir / 'processed_data' / 'data.yaml'
        with open(new_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"✓ Created optimized data.yaml: {new_yaml_path}")
        
        return new_yaml_path, optimal_size
    
    def train_yolo_model(self, data_yaml, img_size=384, epochs=100, batch_size=16,
                        model_name='yolo11n.pt', patience=20):
        """
        Train YOLO model with optimal hyperparameters
        """
        print("\n" + "="*80)
        print("STEP 5: YOLO MODEL TRAINING")
        print("="*80)
        
        print(f"\nTraining Configuration:")
        print(f"  Model: {model_name}")
        print(f"  Image size: {img_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Patience: {patience}")
        print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        # Initialize model
        model = YOLO(model_name)
        
        # Training arguments optimized for WBC detection
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            
            # Optimization
            optimizer='AdamW',
            lr0=0.001,  # Initial learning rate
            lrf=0.01,   # Final learning rate
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Augmentation (moderate for medical images)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.3,
            shear=0.0,
            perspective=0.0,
            flipud=0.5,
            fliplr=0.5,
            mosaic=0.5,
            mixup=0.1,
            
            # Loss weights
            box=7.5,
            cls=0.5,
            dfl=1.5,
            
            # Other settings
            patience=patience,
            save=True,
            save_period=10,
            val=True,
            plots=True,
            device=0 if torch.cuda.is_available() else 'cpu',
            workers=8,
            project=str(self.output_dir / 'models'),
            name='wbc_detection',
            exist_ok=True,
            pretrained=True,
            verbose=True,
            
            # Specific for single class
            single_cls=True,
        )
        
        print("\n✓ Training complete!")
        print(f"✓ Results saved to: {self.output_dir / 'models' / 'wbc_detection'}")
        
        return model, results
    
    def validate_model(self, model_path, data_yaml, img_size=384):
        """
        Validate trained model and compute detailed metrics
        """
        print("\n" + "="*80)
        print("STEP 6: MODEL VALIDATION")
        print("="*80)
        
        # Load best model
        model = YOLO(model_path)
        
        # Run validation
        print("\nRunning validation...")
        metrics = model.val(
            data=str(data_yaml),
            imgsz=img_size,
            batch=16,
            conf=0.25,
            iou=0.6,
            plots=True,
            save_json=True,
            project=str(self.output_dir / 'validation'),
            name='val_results',
            exist_ok=True
        )
        
        # Extract metrics
        results = {
            'mAP@0.5': float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0,
            'mAP@0.75': float(metrics.box.map75) if hasattr(metrics.box, 'map75') else 0.0,
            'mAP@0.5:0.95': float(metrics.box.map) if hasattr(metrics.box, 'map') else 0.0,
            'precision': float(metrics.box.p) if hasattr(metrics.box, 'p') else 0.0,
            'recall': float(metrics.box.r) if hasattr(metrics.box, 'r') else 0.0,
        }
        
        # Print results
        print("\n" + "="*80)
        print("VALIDATION RESULTS")
        print("="*80)
        for key, value in results.items():
            print(f"{key:20s}: {value:.4f}")
        print("="*80)
        
        # Save results
        # results_path = self.output_dir / 'validation' / 'metrics.json'
        # with open(results_path, 'w') as f:
        #     json.dump(results, f, indent=2)
        
        # print(f"\n✓ Validation results saved to: {results_path}")
        
        return results
    
    def export_model(self, model_path, formats=['onnx', 'torchscript']):
        """Export model to different formats"""
        print("\n" + "="*80)
        print("STEP 7: MODEL EXPORT")
        print("="*80)
        
        model = YOLO(model_path)
        
        for fmt in formats:
            print(f"\nExporting to {fmt.upper()}...")
            try:
                model.export(format=fmt)
                print(f"✓ Successfully exported to {fmt}")
            except Exception as e:
                print(f"✗ Error exporting to {fmt}: {e}")
    
    def create_inference_examples(self, model_path, num_samples=10):
        """Create inference examples on validation set"""
        print("\n" + "="*80)
        print("STEP 8: INFERENCE EXAMPLES")
        print("="*80)
        
        model = YOLO(model_path)
        
        # Get validation images
        val_images = list(self.val_images.glob("*.jpg")) + list(self.val_images.glob("*.png"))
        np.random.shuffle(val_images)
        samples = val_images[:num_samples]
        
        inference_dir = self.output_dir / 'inference_examples'
        inference_dir.mkdir(exist_ok=True)
        
        print(f"\nRunning inference on {len(samples)} samples...")
        
        for idx, img_path in enumerate(tqdm(samples)):
            # Run inference
            results = model(img_path, conf=0.25, iou=0.6, verbose=False)[0]
            
            # Load original image
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Ground truth
            label_path = self.val_labels / (img_path.stem + '.txt')
            img_gt = img_rgb.copy()
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    labels = f.readlines()
                
                for label in labels:
                    parts = label.strip().split()
                    if len(parts) == 5:
                        cls, x_c, y_c, w_box, h_box = map(float, parts)
                        x1 = int((x_c - w_box/2) * w)
                        y1 = int((y_c - h_box/2) * h)
                        x2 = int((x_c + w_box/2) * w)
                        y2 = int((y_c + h_box/2) * h)
                        cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img_gt, 'GT', (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            axes[0].imshow(img_gt)
            axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # Prediction
            img_pred = img_rgb.copy()
            
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                
                cv2.rectangle(img_pred, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img_pred, f'{conf:.2f}', (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            axes[1].imshow(img_pred)
            axes[1].set_title(f'Prediction ({len(results.boxes)} detections)', 
                             fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            plt.suptitle(f'{img_path.name} ({w}x{h})', fontsize=12)
            plt.tight_layout()
            
            save_path = inference_dir / f'inference_{idx:03d}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"\n✓ Inference examples saved to: {inference_dir}")
    
    def generate_report(self, model_path, validation_results):
        """Generate comprehensive training report"""
        print("\n" + "="*80)
        print("STEP 9: GENERATING REPORT")
        print("="*80)
        
        report = []
        report.append("="*80)
        report.append("WBC YOLO DETECTION - TRAINING REPORT")
        report.append("="*80)
        report.append("")
        
        # Dataset statistics
        report.append("DATASET STATISTICS")
        report.append("-"*80)
        for split_name in ['train', 'val']:
            stats = self.stats[split_name]
            report.append(f"\n{split_name.upper()} SET:")
            report.append(f"  Total images: {len(stats['widths'])}")
            report.append(f"  Image size range: {np.min(stats['widths'])}x{np.min(stats['heights'])} - "
                         f"{np.max(stats['widths'])}x{np.max(stats['heights'])}")
            report.append(f"  Average image size: {np.mean(stats['widths']):.0f}x{np.mean(stats['heights']):.0f}")
            report.append(f"  Total labels: {sum(stats['num_labels'])}")
            report.append(f"  Labels per image: {np.mean(stats['num_labels']):.2f} (min: {np.min(stats['num_labels'])}, "
                         f"max: {np.max(stats['num_labels'])})")
            
            if stats['bbox_areas']:
                report.append(f"  Bbox area range: {np.min(stats['bbox_areas']):.0f} - {np.max(stats['bbox_areas']):.0f}")
                report.append(f"  Average bbox area: {np.mean(stats['bbox_areas']):.0f} pixels²")
        
        report.append("")
        
        # Validation results
        report.append("VALIDATION RESULTS")
        report.append("-"*80)
        for key, value in validation_results.items():
            report.append(f"  {key:20s}: {value:.4f}")
        
        report.append("")
        
        # Model information
        report.append("MODEL INFORMATION")
        report.append("-"*80)
        report.append(f"  Model path: {model_path}")
        report.append(f"  Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        report.append("")
        report.append("="*80)
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.output_dir / 'TRAINING_REPORT.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✓ Report saved to: {report_path}")


def main():
    """Main training pipeline"""
    print("="*80)
    print("WBC YOLO DETECTION - COMPLETE TRAINING PIPELINE")
    print("="*80)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    DATASET_ROOT = "yolo_detection_dataset"  # Your dataset directory
    OUTPUT_DIR = "wbc_yolo_output"
    
    # Training parameters
    MODEL_NAME = 'yolo11n.pt'  # yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
    EPOCHS = 150
    BATCH_SIZE = 64
    PATIENCE = 50
    
    # Data preprocessing
    KEEP_MAX_AREA_ONLY = True  # Keep only largest bbox per image
    
    # Visualization
    NUM_GT_SAMPLES = 20  # Ground truth visualizations
    NUM_INFERENCE_SAMPLES = 100  # Inference examples
    
    # ========================================================================
    
    # Initialize trainer
    trainer = WBCYOLOTrainer(
        dataset_root=DATASET_ROOT,
        output_dir=OUTPUT_DIR
    )
    
    # Step 1: Analyze dataset
    trainer.analyze_dataset()
    
    # Step 2: Preprocess labels
    trainer.preprocess_labels(keep_max_area_only=KEEP_MAX_AREA_ONLY)
    
    # Step 3: Visualize ground truth
    trainer.visualize_ground_truth(num_samples=NUM_GT_SAMPLES)
    
    # Step 4: Create optimal data.yaml
    data_yaml, optimal_img_size = trainer.create_optimal_data_yaml()
    
    # Step 5: Train model
    model, results = trainer.train_yolo_model(
        data_yaml=data_yaml,
        img_size=optimal_img_size,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_name=MODEL_NAME,
        patience=PATIENCE
    )
    
    # Get best model path
    if hasattr(model, 'trainer') and model.trainer is not None:
        best_model = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
    else:
        # Fallback to the path you expected, but ensure it exists
        best_model = trainer.output_dir / 'models' / 'wbc_detection' / 'weights' / 'best.pt'

    print(f"Checking for model at: {best_model}")
    if not best_model.exists():
        # Search for it if the path logic failed
        print("Model not found at primary path, searching in project directory...")
        found_models = list(Path(OUTPUT_DIR).glob("**/best.pt"))
        if found_models:
            best_model = found_models[0]
            print(f"Found model at: {best_model}")
        else:
            raise FileNotFoundError(f"Could not find best.pt. Did the training finish at least one epoch?")
    # best_model = trainer.output_dir / 'models' / 'wbc_detection' / 'weights' / 'best.pt'
    
    # Step 6: Validate model
    validation_results = trainer.validate_model(
        model_path=best_model,
        data_yaml=data_yaml,
        img_size=optimal_img_size
    )
    
    # Step 7: Export model
    trainer.export_model(
        model_path=best_model,
        formats=['onnx', 'torchscript']
    )
    
    # Step 8: Create inference examples
    trainer.create_inference_examples(
        model_path=best_model,
        num_samples=NUM_INFERENCE_SAMPLES
    )
    
    # Step 9: Generate report
    trainer.generate_report(
        model_path=best_model,
        validation_results=validation_results
    )
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print("\nKey files:")
    print(f"  - Best model: {best_model}")
    print(f"  - Analysis: {OUTPUT_DIR}/analysis/")
    print(f"  - Visualizations: {OUTPUT_DIR}/visualizations/")
    print(f"  - Validation: {OUTPUT_DIR}/validation/")
    print(f"  - Inference examples: {OUTPUT_DIR}/inference_examples/")
    print(f"  - Report: {OUTPUT_DIR}/TRAINING_REPORT.txt")
    print("\n" + "="*80)


if __name__ == "__main__":
    # source wbc_env/bin/activate
    main()