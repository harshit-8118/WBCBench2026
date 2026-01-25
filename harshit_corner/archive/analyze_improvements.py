"""
Analysis script to compare base model vs ensemble performance
Specifically focuses on confused class pairs
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from harshit_corner.train_enhanced import Config, WBCClassifier, WBCDataset, get_valid_transforms
from harshit_corner.archive.train_ensemble import FeatureExtractor, EnsembleClassifiers, EnsembleConfig

def analyze_confusion_improvement(
    base_model_path,
    ensemble_dir,
    eval_csv_path,
    eval_img_dir,
    save_dir="analysis"
):
    """
    Compare base model vs ensemble on confused class pairs
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*80)
    print("CONFUSION ANALYSIS: Base Model vs Ensemble")
    print("="*80)
    
    # Setup
    class_to_idx = {cls: idx for idx, cls in enumerate(Config.class_names)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    # Load validation data
    eval_dataset = WBCDataset(
        eval_csv_path,
        eval_img_dir,
        class_to_idx,
        transform=get_valid_transforms(Config.final_image_size)
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8
    )
    
    # Get ground truth
    y_true = np.array([class_to_idx[label] for label in eval_dataset.labels])
    
    # =========================================================================
    # 1. Get base model predictions
    # =========================================================================
    print("\n[1] Getting base model predictions...")
    checkpoint = torch.load(base_model_path, map_location=Config.device)
    base_model = WBCClassifier(Config.model_type, Config.num_classes, config=Config).to(Config.device)
    base_model.load_state_dict(checkpoint.get('model_ema', checkpoint['model']))
    base_model.eval()
    
    base_preds = []
    with torch.no_grad():
        for batch in eval_loader:
            pixel_values = batch['pixel_values'].to(Config.device)
            logits = base_model(pixel_values)
            preds = torch.argmax(logits, dim=1)
            base_preds.extend(preds.cpu().numpy())
    
    base_preds = np.array(base_preds)
    base_f1 = f1_score(y_true, base_preds, average='macro')
    print(f"✓ Base Model F1: {base_f1:.4f}")
    
    # =========================================================================
    # 2. Get ensemble predictions
    # =========================================================================
    print("\n[2] Getting ensemble predictions...")
    
    # Extract features
    extractor = FeatureExtractor(base_model, Config.device, Config)
    X_val, _, _ = extractor.extract_features(eval_loader, "Extracting features")
    
    # Load ensemble
    ensemble = EnsembleClassifiers(EnsembleConfig, Config.class_names)
    ensemble.load(ensemble_dir)
    
    # Get predictions from all methods
    ensemble_results = {}
    for method_name in ensemble.classifiers.keys():
        preds, _ = ensemble.predict(X_val, method=method_name)
        f1 = f1_score(y_true, preds, average='macro')
        ensemble_results[method_name] = {
            'predictions': preds,
            'f1': f1
        }
        print(f"  {method_name:20s}: F1 = {f1:.4f}")
    
    # Weighted ensemble
    if hasattr(ensemble, 'ensemble_weights'):
        preds, _ = ensemble.predict(X_val, method='weighted')
        f1 = f1_score(y_true, preds, average='macro')
        ensemble_results['weighted'] = {
            'predictions': preds,
            'f1': f1
        }
        print(f"  {'weighted':20s}: F1 = {f1:.4f}")
    
    # =========================================================================
    # 3. Analyze confused class pairs
    # =========================================================================
    print("\n" + "="*80)
    print("CONFUSED CLASS PAIR ANALYSIS")
    print("="*80)
    
    confused_pairs = [
        ('VLY', 'LY'),
        ('BNE', 'MMY'),
        ('BNE', 'SNE'),
        ('MMY', 'SNE')
    ]
    
    analysis_results = []
    
    for cls1, cls2 in confused_pairs:
        if cls1 not in class_to_idx or cls2 not in class_to_idx:
            continue
        
        print(f"\n{cls1} vs {cls2}")
        print("-" * 40)
        
        idx1 = class_to_idx[cls1]
        idx2 = class_to_idx[cls2]
        
        # Get samples of these two classes
        mask = (y_true == idx1) | (y_true == idx2)
        y_subset = y_true[mask]
        
        # Base model confusion
        base_subset = base_preds[mask]
        base_acc = (y_subset == base_subset).mean()
        base_confused = ((y_subset == idx1) & (base_subset == idx2)).sum()
        base_confused += ((y_subset == idx2) & (base_subset == idx1)).sum()
        
        print(f"Base Model:")
        print(f"  Accuracy: {base_acc:.3f}")
        print(f"  Confused: {base_confused}/{mask.sum()} ({base_confused/mask.sum()*100:.1f}%)")
        
        # Ensemble confusion
        best_ensemble = None
        best_improvement = 0
        
        for method_name, result in ensemble_results.items():
            ens_subset = result['predictions'][mask]
            ens_acc = (y_subset == ens_subset).mean()
            ens_confused = ((y_subset == idx1) & (ens_subset == idx2)).sum()
            ens_confused += ((y_subset == idx2) & (ens_subset == idx1)).sum()
            
            improvement = (base_confused - ens_confused) / base_confused * 100 if base_confused > 0 else 0
            
            print(f"\n{method_name}:")
            print(f"  Accuracy: {ens_acc:.3f} ({'+' if ens_acc > base_acc else ''}{(ens_acc-base_acc):.3f})")
            print(f"  Confused: {ens_confused}/{mask.sum()} ({ens_confused/mask.sum()*100:.1f}%)")
            print(f"  Improvement: {improvement:+.1f}%")
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_ensemble = method_name
        
        analysis_results.append({
            'pair': f"{cls1}/{cls2}",
            'base_accuracy': base_acc,
            'base_confused': base_confused,
            'base_confused_pct': base_confused/mask.sum()*100,
            'best_method': best_ensemble,
            'best_improvement': best_improvement,
            'total_samples': mask.sum()
        })
    
    # =========================================================================
    # 4. Visualize improvements
    # =========================================================================
    print("\n[4] Creating visualizations...")
    
    # Plot 1: F1 Score Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Subplot 1: Overall F1 scores
    methods = ['Base Model'] + list(ensemble_results.keys())
    f1_scores = [base_f1] + [r['f1'] for r in ensemble_results.values()]
    
    colors = ['#d62728' if i == 0 else '#2ca02c' for i in range(len(methods))]
    
    axes[0, 0].barh(methods, f1_scores, color=colors)
    axes[0, 0].set_xlabel('Macro F1 Score')
    axes[0, 0].set_title('Overall Performance Comparison')
    axes[0, 0].axvline(base_f1, color='red', linestyle='--', alpha=0.5, label='Base Model')
    axes[0, 0].legend()
    
    for i, (method, score) in enumerate(zip(methods, f1_scores)):
        axes[0, 0].text(score, i, f' {score:.4f}', va='center')
    
    # Subplot 2: Confusion reduction for each pair
    pairs = [r['pair'] for r in analysis_results]
    improvements = [r['best_improvement'] for r in analysis_results]
    
    axes[0, 1].barh(pairs, improvements, color='#1f77b4')
    axes[0, 1].set_xlabel('Confusion Reduction (%)')
    axes[0, 1].set_title('Confusion Reduction by Class Pair')
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.5)
    
    for i, (pair, imp) in enumerate(zip(pairs, improvements)):
        axes[0, 1].text(imp, i, f' {imp:+.1f}%', va='center')
    
    # Subplot 3: Confusion matrices comparison
    # Base model confusion matrix
    cm_base = confusion_matrix(y_true, base_preds, labels=range(Config.num_classes))
    cm_base_norm = cm_base.astype('float') / cm_base.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_base_norm, annot=False, fmt='.2f', cmap='RdYlGn_r',
                xticklabels=Config.class_names, yticklabels=Config.class_names,
                ax=axes[1, 0], cbar_kws={'label': 'Proportion'})
    axes[1, 0].set_title('Base Model Confusion Matrix')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    
    # Best ensemble confusion matrix
    best_overall = max(ensemble_results.items(), key=lambda x: x[1]['f1'])
    cm_ens = confusion_matrix(y_true, best_overall[1]['predictions'], labels=range(Config.num_classes))
    cm_ens_norm = cm_ens.astype('float') / cm_ens.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_ens_norm, annot=False, fmt='.2f', cmap='RdYlGn_r',
                xticklabels=Config.class_names, yticklabels=Config.class_names,
                ax=axes[1, 1], cbar_kws={'label': 'Proportion'})
    axes[1, 1].set_title(f'Best Ensemble ({best_overall[0]}) Confusion Matrix')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {os.path.join(save_dir, 'performance_comparison.png')}")
    
    # =========================================================================
    # 5. Detailed classification reports
    # =========================================================================
    print("\n[5] Generating classification reports...")
    
    # Base model report
    base_report = classification_report(
        y_true, base_preds,
        target_names=Config.class_names,
        output_dict=True
    )
    
    # Best ensemble report
    ens_report = classification_report(
        y_true, best_overall[1]['predictions'],
        target_names=Config.class_names,
        output_dict=True
    )
    
    # Create comparison DataFrame
    comparison_data = []
    for cls in Config.class_names:
        comparison_data.append({
            'Class': cls,
            'Base_F1': base_report[cls]['f1-score'],
            'Ensemble_F1': ens_report[cls]['f1-score'],
            'Improvement': ens_report[cls]['f1-score'] - base_report[cls]['f1-score'],
            'Base_Precision': base_report[cls]['precision'],
            'Ensemble_Precision': ens_report[cls]['precision'],
            'Base_Recall': base_report[cls]['recall'],
            'Ensemble_Recall': ens_report[cls]['recall'],
            'Support': base_report[cls]['support']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Improvement', ascending=False)
    
    # Save to CSV
    comparison_df.to_csv(os.path.join(save_dir, 'class_wise_comparison.csv'), index=False)
    print(f"✓ Saved to {os.path.join(save_dir, 'class_wise_comparison.csv')}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nOverall Performance:")
    print(f"  Base Model F1:    {base_f1:.4f}")
    print(f"  Best Ensemble F1: {best_overall[1]['f1']:.4f} ({best_overall[0]})")
    print(f"  Improvement:      {'+' if best_overall[1]['f1'] > base_f1 else ''}{(best_overall[1]['f1'] - base_f1):.4f}")
    
    print(f"\nConfused Class Improvements:")
    for result in analysis_results:
        print(f"  {result['pair']:10s}: {result['best_improvement']:+.1f}% reduction (method: {result['best_method']})")
    
    print(f"\nClasses with Largest F1 Improvement:")
    for _, row in comparison_df.head(5).iterrows():
        print(f"  {row['Class']:5s}: {row['Improvement']:+.4f} ({row['Base_F1']:.4f} → {row['Ensemble_F1']:.4f})")
    
    print(f"\nClasses with Largest F1 Drop:")
    for _, row in comparison_df.tail(3).iterrows():
        if row['Improvement'] < 0:
            print(f"  {row['Class']:5s}: {row['Improvement']:+.4f} ({row['Base_F1']:.4f} → {row['Ensemble_F1']:.4f})")
    
    print("\n" + "="*80)
    print(f"Analysis complete! Results saved to {save_dir}/")
    print("="*80)
    
    return comparison_df, analysis_results

if __name__ == "__main__":
    # Run analysis
    comparison_df, analysis_results = analyze_confusion_improvement(
        base_model_path="checkpoints_enhanced/best_model.pth",
        ensemble_dir="ensemble_models",
        eval_csv_path=os.path.join(Config.data_root, Config.eval_csv),
        eval_img_dir=Config.eval_img_dir,
        save_dir="analysis"
    )