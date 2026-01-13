import torch
from torchvision import transforms
from PIL import Image
import os
from glob import glob

def calculate_mean_std_torch(folder_paths):
    # Collect all image paths
    image_paths = []
    for folder in folder_paths:
        image_paths.extend(glob(os.path.join(folder, "*.jpg")))
        image_paths.extend(glob(os.path.join(folder, "*.jpeg")))

    if not image_paths:
        print("No images found.")
        return None

    # Transform to convert PIL Image to Tensor (scales pixels to [0, 1])
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Accumulators for mean and std calculation
    # We track: sum(x) and sum(x^2) per channel (R, G, B)
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    total_pixels = 0

    print(f"Processing {len(image_paths)} images...")

    for path in image_paths:
        try:
            # Load with Pillow
            img = Image.open(path).convert('RGB')
            
            # Convert to Tensor: Shape [C, H, W]
            img_tensor = transform(img)
            
            # Calculate sum and squared sum across H and W (dims 1 and 2)
            psum += img_tensor.sum(dim=[1, 2])
            psum_sq += (img_tensor ** 2).sum(dim=[1, 2])
            
            # Count pixels
            total_pixels += img_tensor.shape[1] * img_tensor.shape[2]
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

    # Final calculations
    # Mean = E[X]
    mean = psum / total_pixels
    
    # Variance = E[X^2] - (E[X])^2
    var = (psum_sq / total_pixels) - (mean ** 2)
    std = torch.sqrt(var)

    return mean, std

# --- Usage ---
folders = ['/data/data/WBCBench/wbc-bench-2026/phase1', '/data/data/WBCBench/wbc-bench-2026/phase2/train']
stats = calculate_mean_std_torch(folders)

if stats:
    mean, std = stats
    print(f"\nResults for normalization:")
    print(f"Mean: {mean.tolist()}")
    print(f"Std:  {std.tolist()}")