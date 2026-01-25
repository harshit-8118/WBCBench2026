"""
Master Visualization Script
Runs all visualization and analysis tools in one go
"""

import os
import sys
import argparse
from datetime import datetime

# Import visualization modules
from harshit_corner.archive.tsne_visualization import main as tsne_main
from attention_visualization import main as attention_main


def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def run_all_visualizations(skip_tsne=False, skip_attention=False):
    """Run all visualization pipelines"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print_banner("WBC Classifier - Comprehensive Visualization Suite")
    print(f"Timestamp: {timestamp}\n")
    
    try:
        if not skip_tsne:
            print_banner("Phase 1: t-SNE Feature Visualization")
            tsne_main()
            print("\n✓ t-SNE visualization completed successfully")
        else:
            print("\n⊗ Skipping t-SNE visualization")
        
        if not skip_attention:
            print_banner("Phase 2: Attention & Prediction Analysis")
            attention_main()
            print("\n✓ Attention visualization completed successfully")
        else:
            print("\n⊗ Skipping attention visualization")
        
        print_banner("All Visualizations Completed Successfully!")
        print("Output directories:")
        print("  - visualizations/tsne/")
        print("  - visualizations/attention/gradcam_per_class/")
        print("  - visualizations/attention/analysis/")
        print("  - visualizations/attention/misclassifications/")
        
    except Exception as e:
        print(f"\n❌ Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive visualizations for WBC classifier"
    )
    parser.add_argument(
        '--skip-tsne', 
        action='store_true',
        help='Skip t-SNE visualization'
    )
    parser.add_argument(
        '--skip-attention',
        action='store_true', 
        help='Skip attention/GradCAM visualization'
    )
    parser.add_argument(
        '--tsne-only',
        action='store_true',
        help='Run only t-SNE visualization'
    )
    parser.add_argument(
        '--attention-only',
        action='store_true',
        help='Run only attention visualization'
    )
    
    args = parser.parse_args()
    
    # Handle exclusive flags
    skip_tsne = args.skip_tsne or args.attention_only
    skip_attention = args.skip_attention or args.tsne_only
    
    run_all_visualizations(
        skip_tsne=skip_tsne,
        skip_attention=skip_attention
    )


if __name__ == "__main__":
    main()