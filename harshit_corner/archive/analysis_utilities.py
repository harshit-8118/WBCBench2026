"""
Additional Analysis Utilities
Extra tools for model diagnostics and feature analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import entropy
import torch
import torch.nn.functional as F
from tqdm import tqdm

from harshit_corner.train_enhanced import Config


def analyze_feature_similarity(features, labels, class_names, output_dir):
    """
    Analyze feature similarity within and between classes
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_classes = len(class_names)
    
    # Compute class centroids
    centroids = np.zeros((n_classes, features.shape[1]))
    for i in range(n_classes):
        mask = labels == i
        if mask.sum() > 0:
            centroids[i] = features[mask].mean(axis=0)
    
    # 1. Within-class cohesion (average distance to centroid)
    cohesion = []
    for i in range(n_classes):
        mask = labels == i
        if mask.sum() > 1:
            class_features = features[mask]
            distances = cdist(class_features, centroids[i:i+1], metric='euclidean')
            cohesion.append(distances.mean())
        else:
            cohesion.append(0)
    
    # 2. Between-class separation (distance between centroids)
    separation = cdist(centroids, centroids, metric='euclidean')
    
    # 3. Silhouette-like score (separation - cohesion)
    quality_scores = []
    for i in range(n_classes):
        # Average distance to other centroids
        other_distances = separation[i, [j for j in range(n_classes) if j != i]]
        avg_separation = other_distances.mean() if len(other_distances) > 0 else 0
        quality_scores.append(avg_separation - cohesion[i])
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Cohesion (lower is better)
    ax = axes[0, 0]
    colors = sns.color_palette("husl", n_classes)
    ax.barh(class_names, cohesion, color=colors)
    ax.set_xlabel("Average Distance to Centroid", fontsize=10)
    ax.set_title("Within-Class Cohesion (Lower = More Compact)", 
                fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 2: Separation heatmap
    ax = axes[0, 1]
    im = ax.imshow(separation, cmap='RdYlGn')
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_title("Between-Class Separation (Higher = Better)", 
                fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Euclidean Distance')
    
    # Plot 3: Quality score (separation - cohesion)
    ax = axes[1, 0]
    ax.barh(class_names, quality_scores, color=colors)
    ax.set_xlabel("Quality Score (Separation - Cohesion)", fontsize=10)
    ax.set_title("Class Discriminability (Higher = Better Separated)", 
                fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    
    # Plot 4: Cohesion vs Separation scatter
    ax = axes[1, 1]
    avg_separations = []
    for i in range(n_classes):
        other_dist = separation[i, [j for j in range(n_classes) if j != i]]
        avg_separations.append(other_dist.mean() if len(other_dist) > 0 else 0)
    
    for i, (name, c, s) in enumerate(zip(class_names, cohesion, avg_separations)):
        ax.scatter(c, s, s=100, c=[colors[i]], label=name, alpha=0.7)
    
    ax.set_xlabel("Within-Class Cohesion", fontsize=10)
    ax.set_ylabel("Between-Class Separation", fontsize=10)
    ax.set_title("Cohesion vs Separation\n(Upper-left is ideal)", 
                fontsize=11, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_similarity_analysis.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Cohesion': cohesion,
        'Avg_Separation': avg_separations,
        'Quality_Score': quality_scores
    })
    metrics_df.to_csv(os.path.join(output_dir, "similarity_metrics.csv"), index=False)
    
    print(f"\nFeature Similarity Analysis:")
    print(metrics_df.to_string())
    print(f"\nSaved to {output_dir}")


def analyze_prediction_entropy(probs, labels, class_names, output_dir):
    """
    Analyze prediction entropy (uncertainty)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate entropy for each prediction
    entropies = entropy(probs.T)  # Shannon entropy
    
    # Normalize entropy (0 = certain, 1 = maximum uncertainty)
    max_entropy = np.log(len(class_names))
    normalized_entropy = entropies / max_entropy
    
    # Separate correct and incorrect predictions
    preds = probs.argmax(axis=1)
    correct_mask = (preds == labels)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Overall entropy distribution
    ax = axes[0, 0]
    ax.hist(normalized_entropy, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(normalized_entropy.mean(), color='red', linestyle='--', 
              label=f'Mean: {normalized_entropy.mean():.3f}')
    ax.set_xlabel("Normalized Entropy", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("Prediction Uncertainty Distribution", fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Correct vs Incorrect entropy
    ax = axes[0, 1]
    ax.hist(normalized_entropy[correct_mask], bins=30, alpha=0.6, 
           label=f'Correct (n={correct_mask.sum()})', color='green')
    ax.hist(normalized_entropy[~correct_mask], bins=30, alpha=0.6,
           label=f'Incorrect (n={(~correct_mask).sum()})', color='red')
    ax.set_xlabel("Normalized Entropy", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("Entropy: Correct vs Incorrect Predictions", fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Entropy per class
    ax = axes[1, 0]
    class_entropies = []
    for i in range(len(class_names)):
        mask = labels == i
        if mask.sum() > 0:
            class_entropies.append(normalized_entropy[mask].mean())
        else:
            class_entropies.append(0)
    
    colors = sns.color_palette("husl", len(class_names))
    ax.barh(class_names, class_entropies, color=colors)
    ax.set_xlabel("Average Normalized Entropy", fontsize=10)
    ax.set_title("Prediction Uncertainty per Class", fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 4: Confidence vs Correctness
    ax = axes[1, 1]
    max_probs = probs.max(axis=1)
    
    ax.scatter(max_probs[correct_mask], normalized_entropy[correct_mask], 
              alpha=0.3, s=10, c='green', label='Correct')
    ax.scatter(max_probs[~correct_mask], normalized_entropy[~correct_mask],
              alpha=0.5, s=20, c='red', label='Incorrect', marker='x')
    ax.set_xlabel("Max Probability (Confidence)", fontsize=10)
    ax.set_ylabel("Normalized Entropy", fontsize=10)
    ax.set_title("Confidence vs Uncertainty\n(Lower-right = confident mistakes)", 
                fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_entropy_analysis.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary statistics
    print("\nPrediction Entropy Analysis:")
    print(f"Overall mean entropy: {normalized_entropy.mean():.4f}")
    print(f"Correct predictions entropy: {normalized_entropy[correct_mask].mean():.4f}")
    print(f"Incorrect predictions entropy: {normalized_entropy[~correct_mask].mean():.4f}")
    print(f"\nSaved to {output_dir}")


def analyze_calibration(probs, labels, class_names, output_dir, n_bins=10):
    """
    Analyze model calibration (reliability diagram)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate accuracy per bin
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (max_probs >= bins[i]) & (max_probs < bins[i+1])
        if i == n_bins - 1:  # Include 1.0 in last bin
            mask = (max_probs >= bins[i]) & (max_probs <= bins[i+1])
        
        if mask.sum() > 0:
            bin_accuracies.append(correct[mask].mean())
            bin_confidences.append(max_probs[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(0)
            bin_confidences.append(bin_centers[i])
            bin_counts.append(0)
    
    # Calculate ECE (Expected Calibration Error)
    total_samples = len(correct)
    ece = sum([
        (count / total_samples) * abs(acc - conf)
        for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
    ])
    
    # Plot reliability diagram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Reliability diagram
    ax = axes[0]
    ax.bar(bin_centers, bin_accuracies, width=1/n_bins, alpha=0.7, 
          edgecolor='black', label='Accuracy')
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    ax.plot(bin_confidences, bin_accuracies, 'bo-', markersize=8, 
           label='Model Calibration')
    
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(f"Reliability Diagram\nECE = {ece:.4f}", 
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Plot 2: Confidence histogram
    ax = axes[1]
    ax.hist(max_probs, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Confidence Distribution", fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add text with calibration info
    text = f"Total Samples: {total_samples}\n"
    text += f"Mean Confidence: {max_probs.mean():.4f}\n"
    text += f"Accuracy: {correct.mean():.4f}\n"
    text += f"ECE: {ece:.4f}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_analysis.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nCalibration Analysis:")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Mean Confidence: {max_probs.mean():.4f}")
    print(f"Overall Accuracy: {correct.mean():.4f}")
    print(f"Confidence - Accuracy Gap: {max_probs.mean() - correct.mean():.4f}")
    print(f"Saved to {output_dir}")


def generate_comprehensive_report(features, probs, labels, class_names, output_dir):
    """
    Generate a comprehensive analysis report
    """
    print("\n" + "="*80)
    print("Generating Comprehensive Analysis Report")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Feature similarity
    print("\n1. Analyzing feature similarity...")
    analyze_feature_similarity(features, labels, class_names, output_dir)
    
    # 2. Prediction entropy
    print("\n2. Analyzing prediction entropy...")
    analyze_prediction_entropy(probs, labels, class_names, output_dir)
    
    # 3. Calibration
    print("\n3. Analyzing model calibration...")
    analyze_calibration(probs, labels, class_names, output_dir)
    
    print("\n" + "="*80)
    print(f"Comprehensive report saved to: {output_dir}")
    print("="*80)


def main():
    """
    Run additional analysis on saved features
    """
    print("="*80)
    print("WBC Classifier - Additional Analysis Utilities")
    print("="*80)
    
    # Load saved features
    features_path = "visualizations/tsne/features.npz"
    if not os.path.exists(features_path):
        print(f"\nError: Features not found at {features_path}")
        print("Please run tsne_visualization.py first to extract features")
        return
    
    print(f"\nLoading features from {features_path}...")
    data = np.load(features_path)
    features = data['features']
    labels = data['labels']
    
    print(f"Loaded {len(features)} samples")
    
    # Load predictions (you need to save these during validation)
    # This is a placeholder - you should save predictions in your main script
    # For now, we'll create dummy probabilities
    print("\nNote: Using dummy probabilities. Please save actual predictions for accurate analysis.")
    
    # Generate random probabilities for demonstration
    # In practice, load actual predictions from validation
    probs = np.random.dirichlet(np.ones(len(Config.class_names)), size=len(features))
    
    # Adjust so argmax matches labels most of the time
    for i in range(len(probs)):
        probs[i, labels[i]] = np.random.uniform(0.6, 0.95)
        probs[i] = probs[i] / probs[i].sum()
    
    # Create output directory
    output_dir = "visualizations/analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comprehensive report
    generate_comprehensive_report(
        features, probs, labels, Config.class_names, output_dir
    )


if __name__ == "__main__":
    main()