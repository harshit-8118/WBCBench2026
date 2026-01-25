"""
t-SNE Visualization of Class Token Embeddings
Visualizes the learned feature space of the WBC classifier
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import from main training script
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# You'll need to import these from your main script
# Adjust the import statement based on your main script name
from harshit_corner.train_enhanced import (
    Config, WBCClassifier, WBCDataset, 
    get_valid_transforms, set_seed
)

set_seed(42)


class FeatureExtractor:
    """Extract features from model for visualization"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def extract_features(self, dataloader, max_samples=None):
        """
        Extract features and labels from dataloader
        
        Returns:
            features: numpy array of shape (N, feature_dim)
            labels: numpy array of shape (N,)
            image_ids: list of image IDs
        """
        all_features = []
        all_labels = []
        all_ids = []
        
        sample_count = 0
        
        for batch in tqdm(dataloader, desc="Extracting features"):
            pixel_values = batch['pixel_values'].to(self.device)
            
            # Extract features before classification head
            features = self.model.get_features(pixel_values)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(batch['label'].cpu().numpy())
            
            if 'image_id' in batch:
                all_ids.extend(batch['image_id'])
            
            sample_count += len(pixel_values)
            if max_samples and sample_count >= max_samples:
                break
        
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        
        return features, labels, all_ids


def plot_tsne_2d(features, labels, class_names, save_path, 
                 perplexity=30, n_iter=1000, random_state=42):
    """
    Create 2D t-SNE visualization
    
    Args:
        features: (N, D) feature array
        labels: (N,) label array
        class_names: list of class names
        save_path: path to save the plot
        perplexity: t-SNE perplexity parameter
        n_iter: number of iterations
    """
    print(f"Computing t-SNE with perplexity={perplexity}...")
    
    # Handle different scikit-learn versions
    try:
        tsne = TSNE(
            n_components=2, 
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state,
            verbose=1
        )
    except TypeError:
        # Newer scikit-learn versions use max_iter instead of n_iter
        tsne = TSNE(
            n_components=2, 
            perplexity=perplexity,
            max_iter=n_iter,
            random_state=random_state,
            verbose=1
        )
    
    embeddings_2d = tsne.fit_transform(features)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color palette
    colors = sns.color_palette("husl", len(class_names))
    
    # Plot each class
    for idx, class_name in enumerate(class_names):
        mask = labels == idx
        if mask.sum() > 0:
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[idx]],
                label=f"{class_name} (n={mask.sum()})",
                alpha=0.6,
                s=50,
                edgecolors='white',
                linewidth=0.5
            )
    
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title(f"t-SNE Visualization of WBC Features (perplexity={perplexity})", 
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE plot to {save_path}")
    plt.close()
    
    return embeddings_2d


def plot_tsne_grid(features, labels, class_names, save_path, 
                   perplexities=[5, 15, 30, 50]):
    """
    Create grid of t-SNE plots with different perplexities
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    colors = sns.color_palette("husl", len(class_names))
    
    for idx, perplexity in enumerate(perplexities):
        print(f"Computing t-SNE with perplexity={perplexity}...")
        
        try:
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                n_iter=1000,
                random_state=42
            )
        except TypeError:
            # Newer scikit-learn versions
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                max_iter=1000,
                random_state=42
            )
        
        embeddings = tsne.fit_transform(features)
        
        ax = axes[idx]
        
        for class_idx, class_name in enumerate(class_names):
            mask = labels == class_idx
            if mask.sum() > 0:
                ax.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c=[colors[class_idx]],
                    label=class_name if idx == 0 else "",
                    alpha=0.6,
                    s=30,
                    edgecolors='white',
                    linewidth=0.3
                )
        
        ax.set_title(f"Perplexity = {perplexity}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Dimension 1", fontsize=10)
        ax.set_ylabel("Dimension 2", fontsize=10)
    
    # Add legend to the first subplot
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, 
              loc='center left', 
              bbox_to_anchor=(1.0, 0.5),
              fontsize=9)
    
    plt.suptitle("t-SNE with Different Perplexities", 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 0.95, 0.99])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE grid to {save_path}")
    plt.close()


def plot_pca_tsne_comparison(features, labels, class_names, save_path):
    """
    Compare PCA and t-SNE visualizations side by side
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    colors = sns.color_palette("husl", len(class_names))
    
    # PCA
    print("Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_embeddings = pca.fit_transform(features)
    
    ax = axes[0]
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        if mask.sum() > 0:
            ax.scatter(
                pca_embeddings[mask, 0],
                pca_embeddings[mask, 1],
                c=[colors[class_idx]],
                label=class_name,
                alpha=0.6,
                s=50,
                edgecolors='white',
                linewidth=0.5
            )
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})", fontsize=11)
    ax.set_title("PCA Projection", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # t-SNE
    print("Computing t-SNE...")
    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    tsne_embeddings = tsne.fit_transform(features)
    
    ax = axes[1]
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        if mask.sum() > 0:
            ax.scatter(
                tsne_embeddings[mask, 0],
                tsne_embeddings[mask, 1],
                c=[colors[class_idx]],
                label=class_name,
                alpha=0.6,
                s=50,
                edgecolors='white',
                linewidth=0.5
            )
    
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.set_title("t-SNE Projection", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Shared legend
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend,
              loc='center left',
              bbox_to_anchor=(1.0, 0.5),
              fontsize=9)
    
    plt.suptitle("PCA vs t-SNE Feature Visualization", 
                fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {save_path}")
    plt.close()


def plot_class_distributions(features, labels, class_names, save_path):
    """
    Plot class-wise feature distributions and statistics
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Class counts
    ax = axes[0, 0]
    unique, counts = np.unique(labels, return_counts=True)
    colors = sns.color_palette("husl", len(class_names))
    ax.bar([class_names[i] for i in unique], counts, color=colors)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Class Distribution in Dataset", fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Feature norm distribution per class
    ax = axes[0, 1]
    feature_norms = np.linalg.norm(features, axis=1)
    
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        if mask.sum() > 0:
            ax.hist(feature_norms[mask], bins=30, alpha=0.5, 
                   label=class_name, color=colors[class_idx])
    
    ax.set_xlabel("Feature Norm", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("Feature Magnitude Distribution", fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    
    # 3. Inter-class distances (mean)
    ax = axes[1, 0]
    n_classes = len(class_names)
    distance_matrix = np.zeros((n_classes, n_classes))
    
    for i in range(n_classes):
        for j in range(n_classes):
            feat_i = features[labels == i]
            feat_j = features[labels == j]
            if len(feat_i) > 0 and len(feat_j) > 0:
                # Mean Euclidean distance between class centroids
                centroid_i = feat_i.mean(axis=0)
                centroid_j = feat_j.mean(axis=0)
                distance_matrix[i, j] = np.linalg.norm(centroid_i - centroid_j)
    
    im = ax.imshow(distance_matrix, cmap='viridis')
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_title("Inter-Class Centroid Distances", fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # 4. Intra-class variance
    ax = axes[1, 1]
    variances = []
    for class_idx in range(n_classes):
        mask = labels == class_idx
        if mask.sum() > 1:
            class_features = features[mask]
            variance = np.var(class_features, axis=0).mean()
            variances.append(variance)
        else:
            variances.append(0)
    
    ax.bar(class_names, variances, color=colors)
    ax.set_ylabel("Average Variance", fontsize=10)
    ax.set_title("Intra-Class Feature Variance", fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved class distribution analysis to {save_path}")
    plt.close()


def main():
    """Main visualization pipeline"""
    
    print("="*80)
    print("WBC Classifier - t-SNE Feature Visualization")
    print("="*80)
    
    # Create output directory
    output_dir = "visualizations/tsne"
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Load EMA model if available
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
    
    # Extract features
    print("\nExtracting features...")
    extractor = FeatureExtractor(model, Config.device)
    features, labels, image_ids = extractor.extract_features(eval_loader)
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Save features for later use
    np.savez(
        os.path.join(output_dir, "features.npz"),
        features=features,
        labels=labels,
        image_ids=image_ids
    )
    print(f"Saved features to {output_dir}/features.npz")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)
    
    # 1. Standard t-SNE
    print("\n1. Standard t-SNE (perplexity=30)")
    plot_tsne_2d(
        features, labels, Config.class_names,
        os.path.join(output_dir, "tsne_standard.png"),
        perplexity=30
    )
    
    # 2. t-SNE grid with different perplexities
    print("\n2. t-SNE with multiple perplexities")
    plot_tsne_grid(
        features, labels, Config.class_names,
        os.path.join(output_dir, "tsne_grid.png"),
        perplexities=[5, 15, 30, 50]
    )
    
    # 3. PCA vs t-SNE comparison
    print("\n3. PCA vs t-SNE comparison")
    plot_pca_tsne_comparison(
        features, labels, Config.class_names,
        os.path.join(output_dir, "pca_vs_tsne.png")
    )
    
    # 4. Class distribution analysis
    print("\n4. Class distribution analysis")
    plot_class_distributions(
        features, labels, Config.class_names,
        os.path.join(output_dir, "class_distributions.png")
    )
    
    # 5. High perplexity t-SNE for global structure
    print("\n5. High perplexity t-SNE (global structure)")
    plot_tsne_2d(
        features, labels, Config.class_names,
        os.path.join(output_dir, "tsne_high_perplexity.png"),
        perplexity=50
    )
    
    print("\n" + "="*80)
    print(f"All visualizations saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()