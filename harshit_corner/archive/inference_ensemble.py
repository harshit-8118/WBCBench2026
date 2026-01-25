"""
Inference script for ensemble classifiers
Use this to make predictions with trained ensemble models
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from harshit_corner.train_enhanced import Config, WBCClassifier, WBCDataset, get_valid_transforms
from harshit_corner.archive.train_ensemble import FeatureExtractor, EnsembleClassifiers, EnsembleConfig

def load_ensemble_model(ensemble_dir, method='weighted'):
    """
    Load trained ensemble model
    
    Args:
        ensemble_dir: Directory containing saved ensemble models
        method: Which classifier to use ('weighted', 'xgboost', 'lightgbm', etc.)
    
    Returns:
        ensemble: Loaded EnsembleClassifiers object
    """
    print(f"Loading ensemble from {ensemble_dir}...")
    ensemble = EnsembleClassifiers(EnsembleConfig, Config.class_names)
    ensemble.load(ensemble_dir)
    print(f"✓ Loaded {len(ensemble.classifiers)} classifiers")
    return ensemble

def predict_with_ensemble(
    base_model_path,
    ensemble_dir,
    test_csv_path,
    test_img_dir,
    output_path,
    method='weighted',
    use_ema=True
):
    """
    Make predictions using ensemble model
    
    Args:
        base_model_path: Path to best_model.pth (feature extractor)
        ensemble_dir: Directory with ensemble models
        test_csv_path: Path to test CSV
        test_img_dir: Test image directory
        output_path: Where to save predictions
        method: Ensemble method to use
        use_ema: Use EMA weights from base model
    """
    
    print("="*80)
    print("ENSEMBLE INFERENCE")
    print("="*80)
    
    # 1. Load base model for feature extraction
    print("\n[1] Loading feature extractor...")
    checkpoint = torch.load(base_model_path, map_location=Config.device)
    model = WBCClassifier(Config.model_type, Config.num_classes, config=Config).to(Config.device)
    
    if 'model_ema' in checkpoint and use_ema:
        print("Using EMA weights")
        model.load_state_dict(checkpoint['model_ema'])
    else:
        model.load_state_dict(checkpoint['model'])
    
    # 2. Load test data
    print("\n[2] Loading test data...")
    class_to_idx = {cls: idx for idx, cls in enumerate(Config.class_names)}
    
    test_dataset = WBCDataset(
        test_csv_path,
        test_img_dir,
        class_to_idx,
        transform=get_valid_transforms(Config.final_image_size),
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8
    )
    
    # 3. Extract features
    print("\n[3] Extracting features...")
    extractor = FeatureExtractor(model, Config.device, Config)
    X_test, _, test_ids = extractor.extract_features(test_loader)
    print(f"✓ Extracted features: {X_test.shape}")
    
    # 4. Load ensemble and predict
    print(f"\n[4] Making predictions with {method}...")
    ensemble = load_ensemble_model(ensemble_dir, method)
    predictions, probabilities = ensemble.predict(X_test, method=method)
    
    # 5. Create submission
    print("\n[5] Creating submission...")
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    submission = pd.DataFrame({
        'ID': test_ids,
        'labels': [idx_to_class[p] for p in predictions]
    })
    
    submission.to_csv(output_path, index=False)
    print(f"✓ Submission saved to {output_path}")
    
    # Print prediction distribution
    print("\nPrediction Distribution:")
    pred_dist = pd.Series([idx_to_class[p] for p in predictions]).value_counts()
    for cls, count in pred_dist.items():
        print(f"  {cls:5s}: {count:4d} ({count/len(predictions)*100:.1f}%)")
    
    return submission, probabilities

def compare_ensemble_methods(base_model_path, ensemble_dir, test_csv_path, test_img_dir):
    """
    Compare predictions from all available ensemble methods
    """
    print("="*80)
    print("COMPARING ALL ENSEMBLE METHODS")
    print("="*80)
    
    # Load ensemble
    ensemble = load_ensemble_model(ensemble_dir)
    
    # Load base model and extract features
    checkpoint = torch.load(base_model_path, map_location=Config.device)
    model = WBCClassifier(Config.model_type, Config.num_classes, config=Config).to(Config.device)
    model.load_state_dict(checkpoint.get('model_ema', checkpoint['model']))
    
    class_to_idx = {cls: idx for idx, cls in enumerate(Config.class_names)}
    test_dataset = WBCDataset(
        test_csv_path, test_img_dir, class_to_idx,
        transform=get_valid_transforms(Config.final_image_size), is_test=True
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
    
    extractor = FeatureExtractor(model, Config.device, Config)
    X_test, _, test_ids = extractor.extract_features(test_loader)
    
    # Get predictions from all methods
    all_predictions = {}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    methods = list(ensemble.classifiers.keys())
    if hasattr(ensemble, 'ensemble_weights'):
        methods.append('weighted')
    
    for method in methods:
        predictions, _ = ensemble.predict(X_test, method=method)
        all_predictions[method] = [idx_to_class[int(p)] for p in predictions]
    
    # Create comparison DataFrame
    comparison = pd.DataFrame(all_predictions)
    comparison.insert(0, 'ID', test_ids)
    
    # Save comparison
    comparison_path = os.path.join(ensemble_dir, 'method_comparison.csv')
    comparison.to_csv(comparison_path, index=False)
    print(f"\n✓ Method comparison saved to {comparison_path}")
    
    # Show agreement statistics
    print("\nMethod Agreement:")
    methods = [m for m in methods if m != 'weighted']
    
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            agreement = (comparison[method1] == comparison[method2]).mean()
            print(f"  {method1:15s} vs {method2:15s}: {agreement*100:.1f}%")
    
    return comparison

# Example usage
if __name__ == "__main__":
    # Configuration
    BASE_MODEL_PATH = "checkpoints_enhanced/best_model.pth"
    ENSEMBLE_DIR = "ensemble_models"
    TEST_CSV = os.path.join(Config.data_root, Config.test_csv)
    TEST_IMG_DIR = Config.test_img_dir
    
    # Method 1: Single prediction with best method
    print("\n" + "="*80)
    print("METHOD 1: Single Prediction")
    print("="*80)
    submission, probs = predict_with_ensemble(
        base_model_path=BASE_MODEL_PATH,
        ensemble_dir=ENSEMBLE_DIR,
        test_csv_path=TEST_CSV,
        test_img_dir=TEST_IMG_DIR,
        output_path=os.path.join(ENSEMBLE_DIR, "final_submission.csv"),
        method='weighted'  # or 'xgboost', 'lightgbm', 'catboost', etc.
    )
    
    # Method 2: Compare all methods
    print("\n" + "="*80)
    print("METHOD 2: Compare All Methods")
    print("="*80)
    comparison = compare_ensemble_methods(
        base_model_path=BASE_MODEL_PATH,
        ensemble_dir=ENSEMBLE_DIR,
        test_csv_path=TEST_CSV,
        test_img_dir=TEST_IMG_DIR
    )
    
    print("\n✓ Inference complete!")