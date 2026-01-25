import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# For YOLO11
from ultralytics import YOLO


# ============================================================================
# YOLO11 WBC Detection Analysis & Visualization
# ============================================================================
# CHANGES FROM ORIGINAL:
# - Removed all nucleus detection functionality
# - Simplified to work with WBC detection model only
# - Updated visualizations to show WBC detections without nucleus comparison
# - Removed dual-model comparison features
# - Streamlined class structure and methods
# ============================================================================

class WBCDetectionAnalyzer:
    """
    UPDATED: Analyzer for YOLO11 WBC detection model only
    (Previously handled both nucleus and WBC detection)
    """
    
    def __init__(self,
                 wbc_model_path=None,
                 data_root="/data/data/WBCBench/wbc-bench-2026",
                 output_dir="wbc_yolo_analysis_results"):  # CHANGED: Updated default output directory name
        """
        Initialize the analyzer
        
        Args:
            wbc_model_path: Path to trained WBC detection YOLO model
            data_root: Root directory for dataset
            output_dir: Directory to save analysis results
            
        REMOVED PARAMETERS:
            - nucleus_model_path (no longer needed)
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "detections").mkdir(exist_ok=True)
        # REMOVED: comparisons subdirectory (was for nucleus vs WBC comparison)
        
        # REMOVED: self.nucleus_model initialization
        # Load WBC model only
        self.wbc_model = None
        
        if wbc_model_path and os.path.exists(wbc_model_path):
            print(f"Loading WBC detection model from: {wbc_model_path}")
            self.wbc_model = YOLO(wbc_model_path)
            print("✓ WBC model loaded successfully")
        else:
            print("⚠ WBC model not provided or not found")
            print(f"  Looked for model at: {wbc_model_path}")
        
        # REMOVED: self.nucleus_detections storage
        # Detection storage for WBC only
        self.wbc_detections = []
        
        # WBC class names (update based on your dataset)
        self.wbc_classes = ['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 
                           'PC', 'PLY', 'PMY', 'SNE', 'VLY']
        
    def analyze_single_image(self, image_path, conf_threshold=0.25, 
                            visualize=True, save_viz=True):
        """
        UPDATED: Analyze a single image with WBC detection model only
        (Previously analyzed with both nucleus and WBC models)
        
        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold for detections
            visualize: Whether to create visualizations
            save_viz: Whether to save visualizations
            
        Returns:
            dict with analysis results (WBC detections only)
        """
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # CHANGED: Removed nucleus_detections from results
        results = {
            'image_path': str(image_path),
            'image_name': Path(image_path).name,
            'image_size': (w, h),
            'wbc_detections': []  # Only WBC detections now
        }
        
        # REMOVED: Nucleus Detection section (entire if self.nucleus_model block)
        
        # WBC Detection (unchanged logic, but now the primary focus)
        if self.wbc_model:
            wbc_results = self.wbc_model(image_path, conf=conf_threshold, verbose=False)[0]
            
            for box in wbc_results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                results['wbc_detections'].append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': conf,
                    'class': cls,
                    'class_name': wbc_results.names[cls],
                    'area': float((x2-x1) * (y2-y1)),
                    'center': [float((x1+x2)/2), float((y1+y2)/2)]
                })
        else:
            print(f"⚠ No WBC model loaded, skipping detection for {Path(image_path).name}")
        
        # Visualization
        if visualize:
            self._visualize_detections(image_rgb, results, save_viz)
        
        return results
    
    def _visualize_detections(self, image, results, save_viz=True):
        """
        UPDATED: Create visualization showing original image and WBC detections side-by-side
        (Previously showed original, nucleus, and WBC in 3 panels)
        """
        # CHANGED: From 3 subplots to 2 subplots (removed nucleus panel)
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # REMOVED: Nucleus detection visualization (axes[1] in original code)
        
        # WBC detections
        img_wbc = image.copy()
        for det in results['wbc_detections']:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls_name = det['class_name']
            
            # Draw box with green color (you can change color if desired)
            cv2.rectangle(img_wbc, (int(x1), int(y1)), (int(x2), int(y2)), 
                         (0, 255, 0), 2)
            
            # Label with class name and confidence
            label = f"{cls_name}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_wbc, (int(x1), int(y1)-label_h-5), 
                         (int(x1)+label_w, int(y1)), (0, 255, 0), -1)
            cv2.putText(img_wbc, label, (int(x1), int(y1)-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # CHANGED: Now using axes[1] instead of axes[2]
        axes[1].imshow(img_wbc)
        axes[1].set_title(f'WBC Detection ({len(results["wbc_detections"])} cells detected)', 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_viz:
            save_path = self.output_dir / "visualizations" / f"{results['image_name']}"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization: {save_path}")
        
        plt.close()
    
    def analyze_dataset(self, image_dir, csv_file=None, max_images=None, 
                       conf_threshold=0.25, save_individual_viz=False):
        """
        UPDATED: Analyze entire dataset with WBC detection only
        (Previously analyzed with both nucleus and WBC models)
        
        Args:
            image_dir: Directory containing images
            csv_file: Optional CSV with labels for validation
            max_images: Maximum number of images to process
            conf_threshold: Confidence threshold
            save_individual_viz: Save visualization for each image
        """
        print("\n" + "="*80)
        print("WBC DETECTION DATASET ANALYSIS")  # CHANGED: Updated title
        print("="*80)
        
        image_paths = list(Path(image_dir).glob("*.jpg")) + \
                     list(Path(image_dir).glob("*.png")) + \
                     list(Path(image_dir).glob("*.jpeg"))
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"Found {len(image_paths)} images")
        
        all_results = []
        
        for img_path in tqdm(image_paths, desc="Analyzing images"):
            result = self.analyze_single_image(
                img_path, 
                conf_threshold=conf_threshold,
                visualize=save_individual_viz,
                save_viz=save_individual_viz
            )
            
            if result:
                all_results.append(result)
        
        # Aggregate statistics
        self._compute_aggregate_statistics(all_results)
        
        # Save results
        self._save_detection_results(all_results)
        
        return all_results
    
    def _compute_aggregate_statistics(self, results):
        """
        UPDATED: Compute and visualize aggregate statistics for WBC detections only
        (Previously computed statistics for both nucleus and WBC)
        """
        print("\n" + "="*80)
        print("AGGREGATE STATISTICS - WBC DETECTION")  # CHANGED: Updated title
        print("="*80)
        
        # REMOVED: total_nucleus calculation
        # Detection counts for WBC only
        total_wbc = sum(len(r['wbc_detections']) for r in results)
        
        print(f"\nTotal Images Analyzed: {len(results)}")
        print(f"Total WBC Detections: {total_wbc}")
        # REMOVED: Total Nucleus Detections and Average Nucleus per Image
        print(f"Avg WBC per Image: {total_wbc/len(results):.2f}")
        
        # Check if we have any detections
        if total_wbc == 0:  # CHANGED: Only check WBC (previously checked both)
            print("\n⚠ WARNING: No WBC detections found!")
            print("Please check:")
            print("  1. WBC model path is correct")
            print("  2. Model is properly loaded")
            print("  3. Confidence threshold is not too high")
            print("  4. Images are in correct format")
            print("  5. Model was trained on compatible data")
            return
        
        # REMOVED: nucleus_confs calculation
        # Confidence distributions for WBC only
        wbc_confs = [d['confidence'] for r in results for d in r['wbc_detections']]
        
        # REMOVED: Nucleus confidence statistics section
        
        if wbc_confs:
            print(f"\nWBC Detection Confidence Statistics:")
            print(f"  Mean: {np.mean(wbc_confs):.3f}")
            print(f"  Std Dev: {np.std(wbc_confs):.3f}")
            print(f"  Min: {np.min(wbc_confs):.3f}")
            print(f"  Max: {np.max(wbc_confs):.3f}")
            print(f"  Median: {np.median(wbc_confs):.3f}")
        
        # Class distribution for WBC
        if wbc_confs:
            wbc_class_counts = defaultdict(int)
            for r in results:
                for d in r['wbc_detections']:
                    wbc_class_counts[d['class_name']] += 1
            
            print("\nWBC Class Distribution:")
            for cls, count in sorted(wbc_class_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cls}: {count} ({count/total_wbc*100:.1f}%)")
        
        # Visualizations
        self._plot_statistics(results, wbc_confs)  # CHANGED: Removed nucleus_confs parameter
    
    def _plot_statistics(self, results, wbc_confs):
        """
        UPDATED: Create statistical plots for WBC detections only
        (Previously created plots comparing nucleus and WBC)
        
        REMOVED PARAMETER: nucleus_confs
        """
        # Check if we have any data to plot
        if not wbc_confs:  # CHANGED: Only check WBC (previously checked both)
            print("\n⚠ Skipping visualization - no WBC detections found")
            return
            
        # CHANGED: Adjusted grid layout from 3x3 to 3x2 (fewer plots needed)
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Detection counts per image
        ax1 = fig.add_subplot(gs[0, 0])
        # REMOVED: nucleus_counts calculation
        wbc_counts = [len(r['wbc_detections']) for r in results]
        
        # CHANGED: Only plot WBC histogram (removed nucleus overlay)
        ax1.hist(wbc_counts, bins=20, alpha=0.7, label='WBC', color='green')
        ax1.set_xlabel('WBC Detections per Image', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title('WBC Detection Count Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Confidence distribution
        ax2 = fig.add_subplot(gs[0, 1])
        # REMOVED: nucleus_confs histogram
        ax2.hist(wbc_confs, bins=30, alpha=0.7, label='WBC', color='green')
        ax2.set_xlabel('Confidence Score', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('WBC Confidence Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Box plot of confidences
        ax3 = fig.add_subplot(gs[1, 0])
        # CHANGED: Only plot WBC boxplot (removed nucleus comparison)
        bp = ax3.boxplot([wbc_confs], labels=['WBC'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        ax3.set_ylabel('Confidence Score', fontsize=10)
        ax3.set_title('WBC Confidence Box Plot', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # 4. Bounding box areas
        ax4 = fig.add_subplot(gs[1, 1])
        # REMOVED: nucleus_areas calculation
        wbc_areas = [d['area'] for r in results for d in r['wbc_detections']]
        
        # CHANGED: Only plot WBC areas (removed nucleus overlay)
        ax4.hist(wbc_areas, bins=30, alpha=0.7, label='WBC', color='green')
        ax4.set_xlabel('Bounding Box Area (pixels²)', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.set_title('WBC Detection Area Distribution', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. WBC class distribution (unchanged)
        ax5 = fig.add_subplot(gs[2, 0])
        wbc_class_counts = defaultdict(int)
        for r in results:
            for d in r['wbc_detections']:
                wbc_class_counts[d['class_name']] += 1
        
        if wbc_class_counts:
            classes = list(wbc_class_counts.keys())
            counts = list(wbc_class_counts.values())
            bars = ax5.bar(classes, counts, color='green', alpha=0.7)
            ax5.set_xlabel('WBC Class', fontsize=10)
            ax5.set_ylabel('Count', fontsize=10)
            ax5.set_title('WBC Class Distribution', fontsize=12, fontweight='bold')
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=8)
        
        # REMOVED: Scatter plot of Nucleus vs WBC detections (no longer relevant)
        
        # 6. Detection confidence vs area
        ax6 = fig.add_subplot(gs[2, 1])
        # CHANGED: Only show WBC confidence vs area (removed nucleus plot)
        ax6.scatter(wbc_areas, wbc_confs, alpha=0.5, s=30, color='green')
        ax6.set_xlabel('Bounding Box Area (pixels²)', fontsize=10)
        ax6.set_ylabel('Confidence', fontsize=10)
        ax6.set_title('WBC: Confidence vs Area', fontsize=12, fontweight='bold')
        ax6.grid(alpha=0.3)
        
        # Add correlation coefficient
        if len(wbc_areas) > 1:
            corr = np.corrcoef(wbc_areas, wbc_confs)[0, 1]
            ax6.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # REMOVED: Nucleus confidence vs area plot
        # REMOVED: Summary statistics table with both nucleus and WBC
        
        # CHANGED: Updated main title
        plt.suptitle('WBC YOLO Detection Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
        
        save_path = self.output_dir / "metrics" / "wbc_analysis_dashboard.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved analysis dashboard: {save_path}")
        plt.close()
    
    def _save_detection_results(self, results):
        """
        UPDATED: Save WBC detection results to JSON and CSV
        (Previously saved both nucleus and WBC detections)
        """
        # Save full JSON
        json_path = self.output_dir / "detections" / "wbc_detections.json"  # CHANGED: Renamed file
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved full detections: {json_path}")
        
        # Create CSV summary
        summary_data = []
        for r in results:
            # CHANGED: Removed nucleus-related fields
            summary_data.append({
                'image_name': r['image_name'],
                'wbc_count': len(r['wbc_detections']),
                'wbc_avg_conf': np.mean([d['confidence'] for d in r['wbc_detections']]) if r['wbc_detections'] else 0,
                'wbc_total_area': sum(d['area'] for d in r['wbc_detections']),
                'wbc_classes': ', '.join(set([d['class_name'] for d in r['wbc_detections']])) if r['wbc_detections'] else ''
            })
        
        df = pd.DataFrame(summary_data)
        csv_path = self.output_dir / "detections" / "wbc_detection_summary.csv"  # CHANGED: Renamed file
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved detection summary: {csv_path}")
    
    # REMOVED: create_comparison_visualization() method (was for nucleus vs WBC comparison)
    
    def create_detailed_visualization(self, image_path, save_path=None):
        """
        NEW METHOD: Create detailed visualization showing WBC detections with statistics
        (Replaces the comparison visualization which showed nucleus vs WBC)
        """
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.analyze_single_image(image_path, visualize=False, save_viz=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # WBC detections with detailed annotations
        img_wbc = image_rgb.copy()
        class_colors = {}
        color_idx = 0
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                  (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]
        
        for det in results['wbc_detections']:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            cls_name = det['class_name']
            
            # Assign color per class
            if cls_name not in class_colors:
                class_colors[cls_name] = colors[color_idx % len(colors)]
                color_idx += 1
            
            color = class_colors[cls_name]
            cv2.rectangle(img_wbc, (x1, y1), (x2, y2), color, 2)
            
            # Label with class and confidence
            label = f"{cls_name}: {det['confidence']:.2f}"
            cv2.putText(img_wbc, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        axes[1].imshow(img_wbc)
        axes[1].set_title(f'WBC Detections ({len(results["wbc_detections"])} cells)', 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved detailed visualization: {save_path}")
        else:
            save_path = self.output_dir / "visualizations" / f"detailed_{Path(image_path).name}"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved detailed visualization: {save_path}")
        
        plt.close()
        
        return results
    
    def evaluate_on_validation_set(self, val_images_dir, gt_annotations_path=None):
        """
        Evaluate model performance on validation set with ground truth
        
        Args:
            val_images_dir: Directory with validation images
            gt_annotations_path: Path to ground truth annotations (YOLO format or COCO format)
        """
        # TODO: Implement mAP, precision, recall calculations with ground truth
        # This would require ground truth annotations in a specific format
        print("Validation evaluation requires ground truth annotations")
        print("For full metrics, please provide annotations in YOLO or COCO format")
        

def main():
    """
    UPDATED: Main execution function for WBC detection only
    (Previously handled both nucleus and WBC detection)
    """
    print("="*80)
    print("YOLO11 WBC Detection Analysis")  # CHANGED: Updated title
    print("="*80)
    
    # Configuration
    # REMOVED: NUCLEUS_MODEL_PATH
    WBC_MODEL_PATH = "models/yolo11.pt"  # Update this with your model path
    DATA_ROOT = "/data/data/WBCBench/wbc-bench-2026"
    
    # For demo: using phase2 eval images
    IMAGE_DIR = os.path.join(DATA_ROOT, "phase2/eval")
    
    # CHANGED: Only check WBC model path
    if not os.path.exists(WBC_MODEL_PATH):
        print("\n" + "="*80)
        print("⚠️  MODEL CONFIGURATION REQUIRED")
        print("="*80)
        print("\nPlease update the WBC model path in the main() function:")
        print(f"  WBC_MODEL_PATH = 'your/path/to/wbc_model.pt'")
        print("\nCurrent path:")
        print(f"  WBC: {WBC_MODEL_PATH}")
        print("\n" + "="*80)
        print("\nThe script will continue but NO detections will be made without the model.")
        print("="*80 + "\n")
        
        response = input("Continue without model? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please update model path and run again.")
            return
    
    # CHANGED: Initialize analyzer with WBC model only
    analyzer = WBCDetectionAnalyzer(  # CHANGED: Class name
        wbc_model_path=WBC_MODEL_PATH if os.path.exists(WBC_MODEL_PATH) else None,
        data_root=DATA_ROOT,
        output_dir="wbc_yolo_analysis_results"  # CHANGED: Directory name
    )
    
    # Option 1: Analyze specific images
    print("\n" + "="*80)
    print("ANALYZING SAMPLE IMAGES")
    print("="*80)
    
    # Get sample images
    sample_images = list(Path(IMAGE_DIR).glob("*.jpg"))[:5]  # First 5 images
    
    if not sample_images:
        print(f"⚠ No images found in {IMAGE_DIR}")
        print("Please check the IMAGE_DIR path")
        return
    
    for img_path in sample_images:
        print(f"\nAnalyzing: {img_path.name}")
        # CHANGED: Use new detailed visualization method instead of comparison
        analyzer.create_detailed_visualization(img_path)
    
    # Option 2: Analyze entire dataset
    print("\n" + "="*80)
    print("ANALYZING ENTIRE DATASET")
    print("="*80)
    
    results = analyzer.analyze_dataset(
        image_dir=IMAGE_DIR,
        max_images=100,  # Limit for speed, remove for full dataset
        conf_threshold=0.25,
        save_individual_viz=False  # Set True to save all visualizations
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved in: {analyzer.output_dir}")
    print("\nGenerated files:")
    print("  - visualizations/: Individual WBC detection images")
    print("  - metrics/: Analysis dashboard and statistics")
    print("  - detections/: JSON and CSV detection data")
    # REMOVED: comparisons directory mention
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    if analyzer.wbc_model is None:  # CHANGED: Only check WBC model
        print("❌ No WBC model was loaded!")
        print("\nTo get actual detections:")
        print("1. Train or obtain a YOLO11 model for WBC detection")
        print("2. Update WBC_MODEL_PATH in this script")
        print("3. Run the script again")
    else:
        print("1. Review the wbc_analysis_dashboard.png for overall performance")
        print("2. Check wbc_detection_summary.csv for per-image statistics")
        print("3. Examine visualizations for qualitative assessment")
        print("4. Use wbc_detections.json for downstream processing or integration")
        print("5. Analyze class distribution to identify model biases")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()