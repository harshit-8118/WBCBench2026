import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Import from your training script
from harshit_corner.train_enhanced import (
    Config, WBCClassifier, WBCDataset, 
    get_valid_transforms, set_seed
)

set_seed(42)

# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================
class EnsembleConfig:
    # Which ensemble methods to use
    use_xgboost = True
    use_lightgbm = False
    use_catboost = False
    use_random_forest = True
    use_gradient_boosting = False  # Slower, can skip
    use_logistic = False
    use_svm = False  # Very slow for large datasets
    use_stacking = False  # Meta-learner combining all
    
    # Paths
    checkpoint_path = "checkpoints_enhanced/best_model.pth"
    save_dir = "ensemble_models"
    
    # Feature extraction settings
    use_ema_model = True  # Use EMA weights if available
    batch_size = 64
    num_workers = 8
    
    # Ensemble-specific parameters
    n_estimators = 2000  # For tree-based methods
    max_depth = 15
    learning_rate = 0.05
    
    # Cross-validation for stacking
    cv_folds = 5
    
    # Class weights for imbalanced classes
    use_class_weights = True
    
    # Focus on confused classes (VLY/LY, BNE/MMY/SNE)
    boost_confused_classes = True
    confused_pairs = [
        ('VLY', 'BL'),
        ('VLY', 'LY'),
        ('BNE', 'MMY'),
        ('BNE', 'SNE'),
        ('MO', 'SNE'),
        ('MMY', 'SNE'),
        ('EO', 'SNE'),
        ('LY', 'BL')
    ]

# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================
class FeatureExtractor:
    """Extract features from trained model"""
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.model.eval()
    
    @torch.no_grad()
    def extract_features(self, dataloader, desc="Extracting Features"):
        """Extract deep features from model"""
        all_features = []
        all_labels = []
        all_ids = []
        is_test = False
        
        for batch in tqdm(dataloader, desc=desc):
            pixel_values = batch['pixel_values'].to(self.device)
            
            if 'label' in batch:
                labels = batch['label'].cpu().numpy()
                all_labels.extend(labels)
            else:
                is_test = True
                all_ids.extend(batch['image_id'])
            
            with autocast(enabled=Config.use_amp):
                # Extract features before classifier
                features = self.model.get_features(pixel_values)
                all_features.append(features.cpu().numpy())
        
        features = np.vstack(all_features)
        
        if is_test:
            return features, None, all_ids
        else:
            labels = np.array(all_labels)
            return features, labels, None

# =============================================================================
# ENSEMBLE CLASSIFIERS
# =============================================================================
class EnsembleClassifiers:
    """Collection of ensemble classifiers"""
    
    def __init__(self, config, class_names):
        self.config = config
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.classifiers = {}
        self.scaler = StandardScaler()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        
    def calculate_class_weights(self, y_train):
        """Calculate class weights for imbalanced data"""
        counter = Counter(y_train)
        total = len(y_train)
        
        # Inverse frequency weights
        weights = {}
        for cls_idx in range(self.num_classes):
            count = counter.get(cls_idx, 1)
            weights[cls_idx] = total / (self.num_classes * count)
        
        # Boost confused classes
        if self.config.boost_confused_classes:
            idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
            for cls1, cls2 in self.config.confused_pairs:
                if cls1 in self.class_to_idx:
                    idx1 = self.class_to_idx[cls1]
                    weights[idx1] *= 1.5  # Increase weight by 50%
                if cls2 in self.class_to_idx:
                    idx2 = self.class_to_idx[cls2]
                    weights[idx2] *= 1.5
        
        return weights
    
    def get_sample_weights(self, y_train, class_weights):
        """Get sample weights based on class weights"""
        return np.array([class_weights[y] for y in y_train])
    
    def build_xgboost(self, class_weights):
        """XGBoost with custom parameters for confused classes"""
        params = {
            'objective': 'multi:softprob',
            'num_class': self.num_classes,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'n_estimators': self.config.n_estimators,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 7,
            'gamma': 0.1,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1
        }
        return xgb.XGBClassifier(**params)
    
    def build_lightgbm(self, class_weights):
        """LightGBM - faster alternative to XGBoost"""
        params = {
            'objective': 'multiclass',
            'num_class': self.num_classes,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'n_estimators': self.config.n_estimators,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.05,
            'reg_lambda': 1.0,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'random_state': 42,
            'n_jobs': -1,
            'boosting_type': 'gbdt'
        }
        return lgb.LGBMClassifier(**params)
    
    def build_catboost(self, class_weights):
        """CatBoost - handles categorical features well"""
        params = {
            'iterations': self.config.n_estimators,
            'depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'loss_function': 'MultiClass',
            'eval_metric': 'TotalF1',
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1,
            'l2_leaf_reg': 3.0,
            'bagging_temperature': 0.5,
            'random_strength': 1.0
        }
        
        if self.config.use_class_weights:
            # CatBoost uses class_weights parameter
            params['class_weights'] = list(class_weights.values())
        
        return CatBoostClassifier(**params)
    
    def build_random_forest(self, class_weights):
        """Random Forest with balanced class weights"""
        params = {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        }
        
        if self.config.use_class_weights:
            params['class_weight'] = class_weights
        
        return RandomForestClassifier(**params)
    
    def build_gradient_boosting(self, class_weights):
        """Gradient Boosting (sklearn)"""
        return GradientBoostingClassifier(
            n_estimators=300,  # Fewer for speed
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
    
    def build_logistic(self, class_weights):
        """Logistic Regression with L2 regularization"""
        params = {
            'C': 0.1,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'multi_class': 'multinomial',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        }
        
        if self.config.use_class_weights:
            params['class_weight'] = class_weights
        
        return LogisticRegression(**params)
    
    def build_svm(self, class_weights):
        """SVM with RBF kernel (slow for large datasets)"""
        params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': 42,
            'verbose': False
        }
        
        if self.config.use_class_weights:
            params['class_weight'] = class_weights
        
        return SVC(**params)
    
    def train_base_classifiers(self, X_train, y_train):
        """Train all enabled base classifiers"""
        print("\n" + "="*80)
        print("Training Base Classifiers")
        print("="*80)
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(y_train)
        sample_weights = self.get_sample_weights(y_train, class_weights)
        
        print(f"Class weights: {class_weights}")
        
        # Train each enabled classifier
        if self.config.use_xgboost:
            print("\n[1/7] Training XGBoost...")
            self.classifiers['xgboost'] = self.build_xgboost(class_weights)
            self.classifiers['xgboost'].fit(
                X_train_scaled, y_train,
                sample_weight=sample_weights if self.config.use_class_weights else None
            )
            print(f"✓ XGBoost trained")
        
        if self.config.use_lightgbm:
            print("\n[2/7] Training LightGBM...")
            self.classifiers['lightgbm'] = self.build_lightgbm(class_weights)
            self.classifiers['lightgbm'].fit(
                X_train_scaled, y_train,
                sample_weight=sample_weights if self.config.use_class_weights else None
            )
            print(f"✓ LightGBM trained")
        
        if self.config.use_catboost:
            print("\n[3/7] Training CatBoost...")
            self.classifiers['catboost'] = self.build_catboost(class_weights)
            self.classifiers['catboost'].fit(X_train_scaled, y_train, verbose=False)
            print(f"✓ CatBoost trained")
        
        if self.config.use_random_forest:
            print("\n[4/7] Training Random Forest...")
            self.classifiers['random_forest'] = self.build_random_forest(class_weights)
            self.classifiers['random_forest'].fit(X_train_scaled, y_train)
            if hasattr(self.classifiers['random_forest'], 'oob_score_'):
                print(f"  OOB Score: {self.classifiers['random_forest'].oob_score_:.4f}")
            print(f"✓ Random Forest trained")
        
        if self.config.use_gradient_boosting:
            print("\n[5/7] Training Gradient Boosting...")
            self.classifiers['gradient_boosting'] = self.build_gradient_boosting(class_weights)
            self.classifiers['gradient_boosting'].fit(X_train_scaled, y_train)
            print(f"✓ Gradient Boosting trained")
        
        if self.config.use_logistic:
            print("\n[6/7] Training Logistic Regression...")
            self.classifiers['logistic'] = self.build_logistic(class_weights)
            self.classifiers['logistic'].fit(X_train_scaled, y_train)
            print(f"✓ Logistic Regression trained")
        
        if self.config.use_svm:
            print("\n[7/7] Training SVM...")
            print("  (Warning: This may take a long time...)")
            self.classifiers['svm'] = self.build_svm(class_weights)
            self.classifiers['svm'].fit(X_train_scaled, y_train)
            print(f"✓ SVM trained")
        
        print("\n" + "="*80)
        print(f"Trained {len(self.classifiers)} base classifiers")
        print("="*80)
    
    def train_stacking_classifier(self, X_train, y_train):
        """Train meta-learner using stacking"""
        if not self.config.use_stacking or len(self.classifiers) < 2:
            print("\nSkipping stacking (need at least 2 base classifiers)")
            return
        
        print("\n" + "="*80)
        print("Training Stacking Ensemble (Meta-Learner)")
        print("="*80)
        
        X_train_scaled = self.scaler.transform(X_train)
        
        # Prepare base estimators for VotingClassifier
        estimators = [(name, clf) for name, clf in self.classifiers.items() 
                     if name != 'stacking']
        
        # Use soft voting (averaging probabilities)
        self.classifiers['stacking'] = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1,
            verbose=False
        )
        
        print(f"Stacking {len(estimators)} classifiers with soft voting...")
        self.classifiers['stacking'].fit(X_train_scaled, y_train)
        print("✓ Stacking ensemble trained")
    
    def train_weighted_ensemble(self, X_train, y_train, X_val, y_val):
        """Train weighted ensemble based on validation performance"""
        print("\n" + "="*80)
        print("Training Weighted Ensemble")
        print("="*80)
        
        X_val_scaled = self.scaler.transform(X_val)
        
        # Get predictions from each classifier
        weights = {}
        for name, clf in self.classifiers.items():
            if name in ['stacking', 'weighted']:
                continue
            
            y_pred = clf.predict(X_val_scaled)
            f1 = f1_score(y_val, y_pred, average='macro')
            weights[name] = f1
            print(f"{name:20s}: F1 = {f1:.4f}")
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        print("\nNormalized Weights:")
        for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name:20s}: {weight:.4f}")
        
        self.ensemble_weights = weights
    
    def evaluate(self, X_test, y_test, split_name="Validation"):
        """Evaluate all classifiers"""
        print("\n" + "="*80)
        print(f"Evaluating on {split_name} Set")
        print("="*80)
        
        X_test_scaled = self.scaler.transform(X_test)
        results = {}
        
        for name, clf in self.classifiers.items():
            y_pred = clf.predict(X_test_scaled)
            y_proba = clf.predict_proba(X_test_scaled) if hasattr(clf, 'predict_proba') else None
            
            f1 = f1_score(y_test, y_pred, average='macro')
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            
            results[name] = {
                'f1': f1,
                'balanced_accuracy': bal_acc,
                'predictions': y_pred,
                'probabilities': y_proba
            }
            
            print(f"\n{name.upper()}")
            print(f"  Macro F1: {f1:.4f}")
            print(f"  Balanced Accuracy: {bal_acc:.4f}")
            
            # Check performance on confused classes
            if name != 'weighted':
                self._evaluate_confused_classes(y_test, y_pred)
        
        # Weighted ensemble prediction
        if hasattr(self, 'ensemble_weights'):
            weighted_proba = np.zeros((len(X_test_scaled), self.num_classes))
            
            for name, weight in self.ensemble_weights.items():
                if name in self.classifiers:
                    clf = self.classifiers[name]
                    proba = clf.predict_proba(X_test_scaled)
                    weighted_proba += weight * proba
            
            y_pred_weighted = np.argmax(weighted_proba, axis=1)
            f1 = f1_score(y_test, y_pred_weighted, average='macro')
            bal_acc = balanced_accuracy_score(y_test, y_pred_weighted)
            
            results['weighted'] = {
                'f1': f1,
                'balanced_accuracy': bal_acc,
                'predictions': y_pred_weighted,
                'probabilities': weighted_proba
            }
            
            print(f"\nWEIGHTED ENSEMBLE")
            print(f"  Macro F1: {f1:.4f}")
            print(f"  Balanced Accuracy: {bal_acc:.4f}")
            self._evaluate_confused_classes(y_test, y_pred_weighted)
        
        return results
    
    def _evaluate_confused_classes(self, y_true, y_pred):
        """Evaluate performance on confused class pairs"""
        idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        for cls1, cls2 in self.config.confused_pairs:
            if cls1 not in self.class_to_idx or cls2 not in self.class_to_idx:
                continue
            
            idx1 = self.class_to_idx[cls1]
            idx2 = self.class_to_idx[cls2]
            
            # Find samples of these classes
            mask = (y_true == idx1) | (y_true == idx2)
            if mask.sum() == 0:
                continue
            
            y_true_subset = y_true[mask]
            y_pred_subset = y_pred[mask]
            
            # Calculate accuracy for this pair
            acc = (y_true_subset == y_pred_subset).mean()
            
            # Count confusions
            confused = ((y_true_subset == idx1) & (y_pred_subset == idx2)).sum()
            confused += ((y_true_subset == idx2) & (y_pred_subset == idx1)).sum()
            
            print(f"  {cls1}/{cls2}: Acc={acc:.3f}, Confused={confused}/{mask.sum()}")
    
    def predict(self, X_test, method='weighted'):
        """Make predictions using specified method"""
        X_test_scaled = self.scaler.transform(X_test)
        
        if method == 'weighted' and hasattr(self, 'ensemble_weights'):
            # Weighted ensemble
            weighted_proba = np.zeros((len(X_test_scaled), self.num_classes))
            for name, weight in self.ensemble_weights.items():
                if name in self.classifiers:
                    proba = self.classifiers[name].predict_proba(X_test_scaled)
                    weighted_proba += weight * proba
            return np.argmax(weighted_proba, axis=1), weighted_proba
        
        elif method in self.classifiers:
            # Single classifier
            clf = self.classifiers[method]
            preds = clf.predict(X_test_scaled)
            proba = clf.predict_proba(X_test_scaled) if hasattr(clf, 'predict_proba') else None
            return preds, proba
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def save(self, save_dir):
        """Save all classifiers and scaler"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save scaler
        with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save each classifier
        for name, clf in self.classifiers.items():
            path = os.path.join(save_dir, f'{name}_classifier.pkl')
            with open(path, 'wb') as f:
                pickle.dump(clf, f)
            print(f"Saved {name} to {path}")
        
        # Save weights
        if hasattr(self, 'ensemble_weights'):
            with open(os.path.join(save_dir, 'ensemble_weights.json'), 'w') as f:
                json.dump(self.ensemble_weights, f, indent=4)
        
        print(f"\nAll models saved to {save_dir}")
    
    def load(self, save_dir):
        """Load all classifiers and scaler"""
        # Load scaler
        with open(os.path.join(save_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load classifiers
        for filename in os.listdir(save_dir):
            if filename.endswith('_classifier.pkl'):
                name = filename.replace('_classifier.pkl', '')
                path = os.path.join(save_dir, filename)
                with open(path, 'rb') as f:
                    self.classifiers[name] = pickle.load(f)
                print(f"Loaded {name}")
        
        # Load weights
        weights_path = os.path.join(save_dir, 'ensemble_weights.json')
        if os.path.exists(weights_path):
            with open(weights_path, 'r') as f:
                self.ensemble_weights = json.load(f)

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================
def main():
    print("="*80)
    print("ENSEMBLE CLASSIFIER TRAINING")
    print("="*80)
    print(f"Checkpoint: {EnsembleConfig.checkpoint_path}")
    print(f"Save Directory: {EnsembleConfig.save_dir}")
    print("\nEnabled Classifiers:")
    if EnsembleConfig.use_xgboost: print("  ✓ XGBoost")
    if EnsembleConfig.use_lightgbm: print("  ✓ LightGBM")
    if EnsembleConfig.use_catboost: print("  ✓ CatBoost")
    if EnsembleConfig.use_random_forest: print("  ✓ Random Forest")
    if EnsembleConfig.use_gradient_boosting: print("  ✓ Gradient Boosting")
    if EnsembleConfig.use_logistic: print("  ✓ Logistic Regression")
    if EnsembleConfig.use_svm: print("  ✓ SVM")
    if EnsembleConfig.use_stacking: print("  ✓ Stacking Ensemble")
    print("="*80)
    
    os.makedirs(EnsembleConfig.save_dir, exist_ok=True)
    
    # =========================================================================
    # 1. Load trained model
    # =========================================================================
    print("\n[STEP 1] Loading trained model...")
    checkpoint = torch.load(EnsembleConfig.checkpoint_path, map_location=Config.device)
    
    model = WBCClassifier(Config.model_type, Config.num_classes, config=Config).to(Config.device)
    
    # Use EMA weights if available
    if 'model_ema' in checkpoint and EnsembleConfig.use_ema_model:
        print("Using EMA model weights")
        model.load_state_dict(checkpoint['model_ema'])
    else:
        model.load_state_dict(checkpoint['model'])
    
    print(f"✓ Model loaded (F1: {checkpoint.get('f1', 'N/A')})")
    
    # =========================================================================
    # 2. Setup datasets
    # =========================================================================
    print("\n[STEP 2] Setting up datasets...")
    class_to_idx = {cls: idx for idx, cls in enumerate(Config.class_names)}
    
    # Load combined training data
    combined_csv = os.path.join(Config.save_dir, "combined_train.csv")
    if not os.path.exists(combined_csv):
        # Recreate if needed
        train_df = pd.read_csv(os.path.join(Config.data_root, Config.train_csv))
        train_df['img_dir'] = Config.train_img_dir
        
        if Config.use_phase1:
            p1_df = pd.read_csv(os.path.join(Config.data_root, Config.phase1_csv))
            p1_df = p1_df[['ID', 'labels']].copy()
            p1_df['img_dir'] = Config.phase1_img_dir
            train_df = pd.concat([train_df, p1_df], ignore_index=True)
        
        train_df.to_csv(combined_csv, index=False)
    
    train_dataset = WBCDataset(
        combined_csv,
        Config.train_img_dir,
        class_to_idx,
        transform=get_valid_transforms(Config.final_image_size)  # No augmentation for feature extraction
    )
    
    eval_dataset = WBCDataset(
        os.path.join(Config.data_root, Config.eval_csv),
        Config.eval_img_dir,
        class_to_idx,
        transform=get_valid_transforms(Config.final_image_size)
    )
    
    test_dataset = WBCDataset(
        os.path.join(Config.data_root, Config.test_csv),
        Config.test_img_dir,
        class_to_idx,
        transform=get_valid_transforms(Config.final_image_size),
        is_test=True
    )
    
    print(f"✓ Train: {len(train_dataset)}, Val: {len(eval_dataset)}, Test: {len(test_dataset)}")
    
    # =========================================================================
    # 3. Extract features
    # =========================================================================
    print("\n[STEP 3] Extracting features from trained model...")
    extractor = FeatureExtractor(model, Config.device, Config)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=EnsembleConfig.batch_size,
        shuffle=False,
        num_workers=EnsembleConfig.num_workers,
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=EnsembleConfig.batch_size,
        shuffle=False,
        num_workers=EnsembleConfig.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=EnsembleConfig.batch_size,
        shuffle=False,
        num_workers=EnsembleConfig.num_workers,
        pin_memory=True
    )
    
    X_train, y_train, _ = extractor.extract_features(train_loader, "Extracting Train Features")
    X_val, y_val, _ = extractor.extract_features(eval_loader, "Extracting Val Features")
    X_test, _, test_ids = extractor.extract_features(test_loader, "Extracting Test Features")
    
    print(f"\n✓ Feature Extraction Complete")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Save features for future use
    np.save(os.path.join(EnsembleConfig.save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(EnsembleConfig.save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(EnsembleConfig.save_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(EnsembleConfig.save_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(EnsembleConfig.save_dir, 'X_test.npy'), X_test)
    print(f"✓ Features saved to {EnsembleConfig.save_dir}")
    
    # =========================================================================
    # 4. Train ensemble classifiers
    # =========================================================================
    ensemble = EnsembleClassifiers(EnsembleConfig, Config.class_names)
    
    # Train base classifiers
    ensemble.train_base_classifiers(X_train, y_train)
    
    # Train stacking ensemble
    if EnsembleConfig.use_stacking:
        ensemble.train_stacking_classifier(X_train, y_train)
    
    # Train weighted ensemble
    ensemble.train_weighted_ensemble(X_train, y_train, X_val, y_val)
    
    # =========================================================================
    # 5. Evaluate on validation set
    # =========================================================================
    results = ensemble.evaluate(X_val, y_val, split_name="Validation")
    
    # Find best performing method
    best_method = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"\n{'='*80}")
    print(f"BEST METHOD: {best_method[0].upper()}")
    print(f"  Macro F1: {best_method[1]['f1']:.4f}")
    print(f"  Balanced Accuracy: {best_method[1]['balanced_accuracy']:.4f}")
    print(f"{'='*80}")
    
    # =========================================================================
    # 6. Save models
    # =========================================================================
    print("\n[STEP 6] Saving ensemble models...")
    ensemble.save(EnsembleConfig.save_dir)
    
    # Save results
    results_summary = {
        name: {
            'f1': float(r['f1']),
            'balanced_accuracy': float(r['balanced_accuracy'])
        }
        for name, r in results.items()
    }
    
    with open(os.path.join(EnsembleConfig.save_dir, 'validation_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    # =========================================================================
    # 7. Generate predictions for test set
    # =========================================================================
    print("\n[STEP 7] Generating test predictions...")
    
    # Use best method or weighted ensemble
    use_method = 'weighted' if 'weighted' in results else best_method[0]
    print(f"Using method: {use_method}")
    
    test_preds, test_proba = ensemble.predict(X_test, method=use_method)
    
    # Create submission
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    submission = pd.DataFrame({
        'ID': test_ids,
        'labels': [idx_to_class[p] for p in test_preds]
    })
    
    submission_path = os.path.join(EnsembleConfig.save_dir, f'submission_{use_method}.csv')
    submission.to_csv(submission_path, index=False)
    print(f"✓ Submission saved to {submission_path}")
    
    # Save all methods' predictions for ensembling
    for method_name in results.keys():
        if method_name == 'weighted':
            continue
        preds, proba = ensemble.predict(X_test, method=method_name)
        sub = pd.DataFrame({
            'ID': test_ids,
            'labels': [idx_to_class[int(p)] for p in preds]
        })
        sub.to_csv(os.path.join(EnsembleConfig.save_dir, f'submission_{method_name}.csv'), index=False)
    
    print("\n" + "="*80)
    print("ENSEMBLE TRAINING COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {EnsembleConfig.save_dir}")
    print(f"Best validation F1: {best_method[1]['f1']:.4f} ({best_method[0]})")

if __name__ == "__main__":
    main()