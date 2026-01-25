import os
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

from ultralytics import YOLO
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================================
# YOLO Model Validation with Ground Truth
# ============================================================================
# This script computes detailed metrics for YOLO detection models:
# - Precision, Recall, F1 at various IoU thresholds
# - mAP@0.5, mAP@0.75, mAP@0.5:0.95
# - Per-class performance metrics
# - Confusion matrices
# - Precision-Recall curves
# ============================================================================

class YOLOValidator:
    """Validation metrics for YOLO detection models"""
    
    def __init__(self, model_path, model_type='nucleus'):
        """
        Args:
            model_path: Path to YOLO model weights
            model_type: 'nucleus' or 'wbc'
        """
        self.model = YOLO(model_path)
        self.model_type = model_type
        print(f"✓ Loaded {model_type} model from {model_path}")
        
    def calculate_iou(self, box1, box2):
        """
        Calculate IoU between two boxes
        box format: [x1, y1, x2, y2]
        """
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        if x2_min < x1_max or y2_min < y1_max:
            return 0.0
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_predictions_to_ground_truth(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        """
        Match predictions to ground truth using IoU
        
        Returns:
            tp: True positives (matched predictions)
            fp: False positives (unmatched predictions)
            fn: False negatives (unmatched ground truth)
            matched_pairs: List of (pred_idx, gt_idx, iou) tuples
        """
        if len(pred_boxes) == 0:
            return 0, 0, len(gt_boxes), []
        
        if len(gt_boxes) == 0:
            return 0, len(pred_boxes), 0, []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred in enumerate(pred_boxes):
            for j, gt in enumerate(gt_boxes):
                iou_matrix[i, j] = self.calculate_iou(pred, gt)
        
        # Greedy matching: highest IoU first
        matched_pairs = []
        matched_preds = set()
        matched_gts = set()
        
        # Flatten and sort by IoU
        matches = []
        for i in range(len(pred_boxes)):
            for j in range(len(gt_boxes)):
                if iou_matrix[i, j] >= iou_threshold:
                    matches.append((i, j, iou_matrix[i, j]))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        
        for pred_idx, gt_idx, iou in matches:
            if pred_idx not in matched_preds and gt_idx not in matched_gts:
                matched_pairs.append((pred_idx, gt_idx, iou))
                matched_preds.add(pred_idx)
                matched_gts.add(gt_idx)
        
        tp = len(matched_pairs)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        
        return tp, fp, fn, matched_pairs
    
    def compute_metrics_at_iou(self, all_predictions, all_ground_truths, iou_threshold=0.5):
        """
        Compute precision, recall, F1 at specific IoU threshold
        
        Args:
            all_predictions: List of predictions per image
            all_ground_truths: List of ground truth per image
            iou_threshold: IoU threshold for matching
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for preds, gts in zip(all_predictions, all_ground_truths):
            pred_boxes = [p['bbox'] for p in preds]
            gt_boxes = [g['bbox'] for g in gts]
            
            tp, fp, fn, _ = self.match_predictions_to_ground_truth(
                pred_boxes, gt_boxes, iou_threshold
            )
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        }
    
    def compute_map(self, all_predictions, all_ground_truths, iou_thresholds=[0.5, 0.75]):
        """
        Compute mAP at different IoU thresholds
        """
        maps = {}
        
        for iou_thresh in iou_thresholds:
            metrics = self.compute_metrics_at_iou(all_predictions, all_ground_truths, iou_thresh)
            maps[f'mAP@{iou_thresh}'] = metrics['f1']  # Simplified; true mAP needs PR curve integration
        
        # mAP@0.5:0.95 (COCO style)
        iou_range = np.arange(0.5, 1.0, 0.05)
        map_values = []
        for iou_thresh in iou_range:
            metrics = self.compute_metrics_at_iou(all_predictions, all_ground_truths, iou_thresh)
            map_values.append(metrics['f1'])
        
        maps['mAP@0.5:0.95'] = np.mean(map_values)
        
        return maps
    
    def validate_on_dataset(self, val_data_yaml, conf_threshold=0.25):
        """
        Run YOLO validation using ultralytics built-in validation
        
        Args:
            val_data_yaml: Path to YOLO dataset yaml file
            conf_threshold: Confidence threshold
        """
        print(f"\n{'='*80}")
        print(f"Running YOLO Validation for {self.model_type}")
        print(f"{'='*80}\n")
        
        # Run validation
        metrics = self.model.val(
            data=val_data_yaml,
            conf=conf_threshold,
            iou=0.6,
            verbose=True
        )
        
        return metrics
    
    def create_metrics_report(self, metrics, output_dir):
        """Create detailed metrics report with visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Extract metrics
        results = {
            'Model Type': self.model_type,
            'Precision': metrics.box.p if hasattr(metrics.box, 'p') else 0,
            'Recall': metrics.box.r if hasattr(metrics.box, 'r') else 0,
            'mAP@0.5': metrics.box.map50 if hasattr(metrics.box, 'map50') else 0,
            'mAP@0.75': metrics.box.map75 if hasattr(metrics.box, 'map75') else 0,
            'mAP@0.5:0.95': metrics.box.map if hasattr(metrics.box, 'map') else 0,
        }
        
        # Print results
        print(f"\n{'='*80}")
        print(f"VALIDATION RESULTS - {self.model_type.upper()}")
        print(f"{'='*80}")
        for key, value in results.items():
            if key != 'Model Type':
                print(f"{key:20s}: {value:.4f}")
        print(f"{'='*80}\n")
        
        # Save to JSON
        with open(output_dir / f"{self.model_type}_metrics.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualization
        self._plot_metrics_summary(results, output_dir)
        
        return results
    
    def _plot_metrics_summary(self, results, output_dir):
        """Create metrics visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot of main metrics
        metrics_to_plot = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.75', 'mAP@0.5:0.95']
        values = [results[m] for m in metrics_to_plot]
        
        bars = axes[0].bar(range(len(metrics_to_plot)), values, 
                          color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
        axes[0].set_xticks(range(len(metrics_to_plot)))
        axes[0].set_xticklabels(metrics_to_plot, rotation=45, ha='right')
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_title(f'{self.model_type.upper()} Model Performance', 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylim(0, 1.0)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=10)
        
        # Radar chart
        categories = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.75', 'mAP@0.5:0.95']
        values = [results[c] for c in categories]
        values += values[:1]  # Complete the circle
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax = plt.subplot(122, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=2, color='#2E86AB')
        ax.fill(angles, values, alpha=0.25, color='#2E86AB')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9)
        ax.set_ylim(0, 1.0)
        ax.set_title(f'{self.model_type.upper()} Metrics Radar', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{self.model_type}_metrics_summary.png", 
                   dpi=300, bbox_inches='tight')
        print(f"✓ Saved metrics summary: {output_dir / f'{self.model_type}_metrics_summary.png'}")
        plt.close()


class DualModelValidator:
    """Validate both nucleus and WBC detection models together"""
    
    def __init__(self, nucleus_model_path, wbc_model_path, output_dir='validation_results'):
        self.nucleus_validator = YOLOValidator(nucleus_model_path, 'nucleus')
        self.wbc_validator = YOLOValidator(wbc_model_path, 'wbc')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def validate_both_models(self, nucleus_data_yaml, wbc_data_yaml, conf_threshold=0.25):
        """Run validation on both models"""
        print("\n" + "="*80)
        print("DUAL MODEL VALIDATION")
        print("="*80)
        
        # Validate nucleus model
        print("\n[1/2] Validating Nucleus Detection Model...")
        nucleus_metrics = self.nucleus_validator.validate_on_dataset(
            nucleus_data_yaml, conf_threshold
        )
        nucleus_results = self.nucleus_validator.create_metrics_report(
            nucleus_metrics, self.output_dir
        )
        
        # Validate WBC model
        print("\n[2/2] Validating WBC Detection Model...")
        wbc_metrics = self.wbc_validator.validate_on_dataset(
            wbc_data_yaml, conf_threshold
        )
        wbc_results = self.wbc_validator.create_metrics_report(
            wbc_metrics, self.output_dir
        )
        
        # Create comparison
        self._create_comparison_report(nucleus_results, wbc_results)
        
        return nucleus_results, wbc_results
    
    def _create_comparison_report(self, nucleus_results, wbc_results):
        """Create side-by-side comparison of both models"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        metrics = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.75', 'mAP@0.5:0.95']
        
        x = np.arange(len(metrics))
        width = 0.35
        
        nucleus_values = [nucleus_results[m] for m in metrics]
        wbc_values = [wbc_results[m] for m in metrics]
        
        # Grouped bar chart
        bars1 = axes[0].bar(x - width/2, nucleus_values, width, label='Nucleus', 
                           color='#2E86AB', alpha=0.8)
        bars2 = axes[0].bar(x + width/2, wbc_values, width, label='WBC', 
                           color='#A23B72', alpha=0.8)
        
        axes[0].set_xlabel('Metrics', fontsize=12)
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_title('Model Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics, rotation=45, ha='right')
        axes[0].legend()
        axes[0].set_ylim(0, 1.0)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=8)
        
        # Summary table
        axes[1].axis('tight')
        axes[1].axis('off')
        
        table_data = []
        for metric in metrics:
            table_data.append([
                metric,
                f"{nucleus_results[metric]:.4f}",
                f"{wbc_results[metric]:.4f}",
                f"{abs(nucleus_results[metric] - wbc_results[metric]):.4f}"
            ])
        
        table = axes[1].table(
            cellText=table_data,
            colLabels=['Metric', 'Nucleus', 'WBC', 'Diff'],
            cellLoc='center',
            loc='center',
            colWidths=[0.3, 0.2, 0.2, 0.2]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Color code the header
        for i in range(4):
            table[(0, i)].set_facecolor('#E8E8E8')
            table[(0, i)].set_text_props(weight='bold')
        
        axes[1].set_title('Detailed Comparison', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        save_path = self.output_dir / "model_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved model comparison: {save_path}")
        plt.close()


def main():
    """
    Main validation script
    
    Usage:
        Update paths and run:
        python yolo_validation_metrics.py
    """
    print("="*80)
    print("YOLO Model Validation with Metrics")
    print("="*80)
    
    # ========================================================================
    # CONFIGURATION - UPDATE THESE PATHS
    # ========================================================================
    
    # Model paths
    NUCLEUS_MODEL = "path/to/nucleus_model.pt"  # Your trained nucleus model
    WBC_MODEL = "path/to/wbc_model.pt"  # Your trained WBC model
    
    # Dataset YAML files (YOLO format)
    NUCLEUS_DATA_YAML = "path/to/nucleus_dataset.yaml"
    WBC_DATA_YAML = "path/to/wbc_dataset.yaml"
    
    OUTPUT_DIR = "validation_results"
    CONF_THRESHOLD = 0.25
    
    # ========================================================================
    
    # Initialize dual validator
    validator = DualModelValidator(
        nucleus_model_path=NUCLEUS_MODEL,
        wbc_model_path=WBC_MODEL,
        output_dir=OUTPUT_DIR
    )
    
    # Run validation
    nucleus_results, wbc_results = validator.validate_both_models(
        nucleus_data_yaml=NUCLEUS_DATA_YAML,
        wbc_data_yaml=WBC_DATA_YAML,
        conf_threshold=CONF_THRESHOLD
    )
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nResults saved in: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - nucleus_metrics.json: Nucleus model metrics")
    print("  - wbc_metrics.json: WBC model metrics")
    print("  - nucleus_metrics_summary.png: Nucleus visualizations")
    print("  - wbc_metrics_summary.png: WBC visualizations")
    print("  - model_comparison.png: Side-by-side comparison")
    
    # Print summary
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    print("\nNucleus Model:")
    print(f"  Precision: {nucleus_results['Precision']:.4f}")
    print(f"  Recall: {nucleus_results['Recall']:.4f}")
    print(f"  mAP@0.5: {nucleus_results['mAP@0.5']:.4f}")
    
    print("\nWBC Model:")
    print(f"  Precision: {wbc_results['Precision']:.4f}")
    print(f"  Recall: {wbc_results['Recall']:.4f}")
    print(f"  mAP@0.5: {wbc_results['mAP@0.5']:.4f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()