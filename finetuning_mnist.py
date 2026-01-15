import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from torchvision import datasets, transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# =========================
# SETUP
# =========================
OUTPUT_DIR = Path("vit_finetuning_results")
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 10
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("VISION TRANSFORMER FINE-TUNING ON MNIST")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LR}")
print(f"Epochs: {EPOCHS}")
print(f"Output Directory: {OUTPUT_DIR}")
print("=" * 60)

# =========================
# TRANSFORMS
# =========================
print("\n[1/7] Setting up data transforms...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
print("✓ Transforms configured: MNIST (28x28, grayscale) → ViT input (224x224, RGB)")

# =========================
# DATASET
# =========================
print("\n[2/7] Loading MNIST dataset...")
train_ds = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_ds = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

NUM_CLASSES = 10
print(f"✓ Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
print(f"✓ Classes: {NUM_CLASSES} (digits 0-9)")

# =========================
# MODEL ARCHITECTURE INSPECTION
# =========================
print("\n[3/7] Loading pre-trained Vision Transformer...")
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights)

print("\n" + "=" * 60)
print("MODEL ARCHITECTURE OVERVIEW")
print("=" * 60)
print(model)

print("\n" + "-" * 60)
print("KEY COMPONENTS:")
print("-" * 60)
print(f"• Conv Projection: {model.conv_proj}")
print(f"  - Converts image patches to embeddings")
print(f"  - Patch size: 16x16 → (224/16)² = 196 patches")
print(f"\n• Encoder: {len(model.encoder.layers)} transformer blocks")
print(f"  - Each block has: Multi-Head Attention + MLP")
print(f"  - Hidden dimension: {model.encoder.layers[0].mlp[0].in_features}")
print(f"\n• Classification Head (original): {model.heads.head}")

# Freeze backbone
print("\n" + "-" * 60)
print("FREEZING BACKBONE PARAMETERS...")
print("-" * 60)
total_params = 0
frozen_params = 0

for name, param in model.named_parameters():
    total_params += param.numel()
    param.requires_grad = False
    frozen_params += param.numel()

print(f"✓ Frozen {frozen_params:,} parameters")

# Replace head
in_features = model.heads.head.in_features
model.heads.head = nn.Linear(in_features, NUM_CLASSES)
print(f"\n✓ Replaced classification head: {in_features} → {NUM_CLASSES} classes")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nParameter Summary:")
print(f"  • Total: {total_params:,}")
print(f"  • Frozen: {frozen_params:,}")
print(f"  • Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

model = model.to(DEVICE)

# =========================
# VISUALIZE MODEL STRUCTURE
# =========================
print("\n[4/7] Creating model visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Vision Transformer Architecture Analysis', fontsize=16, fontweight='bold')

# Plot 1: Parameter distribution
ax1 = axes[0, 0]
layer_names = ['Conv Proj', 'Encoder', 'Head (frozen)', 'Head (new)']
param_counts = [
    sum(p.numel() for p in model.conv_proj.parameters()),
    sum(p.numel() for p in model.encoder.parameters()),
    in_features * 1000,  # original head
    trainable_params
]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
ax1.barh(layer_names, param_counts, color=colors)
ax1.set_xlabel('Parameters (count)')
ax1.set_title('Parameter Distribution by Layer')
for i, v in enumerate(param_counts):
    ax1.text(v, i, f' {v:,}', va='center')

# Plot 2: Patch embedding visualization
ax2 = axes[0, 1]
sample_img, _ = train_ds[0]
sample_np = sample_img.permute(1, 2, 0).numpy()
sample_np = (sample_np - sample_np.min()) / (sample_np.max() - sample_np.min())
ax2.imshow(sample_np)
ax2.set_title(f'Sample MNIST Digit (Resized to 224×224)\nDivided into {14}×{14} patches of 16×16 pixels')
for i in range(0, 224, 16):
    ax2.axhline(i, color='red', linewidth=0.5, alpha=0.3)
    ax2.axvline(i, color='red', linewidth=0.5, alpha=0.3)
ax2.axis('off')

# Plot 3: Training strategy
ax3 = axes[1, 0]
ax3.text(0.5, 0.8, 'Fine-Tuning Strategy', ha='center', fontsize=14, fontweight='bold')
ax3.text(0.1, 0.6, '✓ Freeze: Pre-trained backbone', fontsize=11)
ax3.text(0.1, 0.5, f'   ({frozen_params:,} params)', fontsize=9, style='italic')
ax3.text(0.1, 0.35, '✓ Train: Classification head only', fontsize=11)
ax3.text(0.1, 0.25, f'   ({trainable_params:,} params)', fontsize=9, style='italic')
ax3.text(0.1, 0.1, f'✓ Transfer Learning: ImageNet → MNIST', fontsize=11)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')

# Plot 4: Architecture flow
ax4 = axes[1, 1]
ax4.text(0.5, 0.95, 'Data Flow Through ViT', ha='center', fontsize=12, fontweight='bold')
flow_text = [
    '1. Input: 224×224×3 image',
    '2. Patch Embed: 196 patches → 768-dim',
    '3. Add Position Embeddings',
    '4. Transformer Encoder (12 blocks)',
    '   • Multi-Head Attention',
    '   • Layer Norm + MLP',
    '5. Extract [CLS] token',
    '6. Classification Head → 10 classes'
]
for i, text in enumerate(flow_text):
    y = 0.85 - i * 0.1
    ax4.text(0.05, y, text, fontsize=10, family='monospace')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'model_architecture.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'model_architecture.png'}")
plt.close()

# =========================
# LOSS & OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.heads.parameters(), lr=LR)

# =========================
# TRAINING LOOP WITH MONITORING
# =========================
print("\n[5/7] Starting training...")
print("=" * 60)

train_losses = []
train_accs = []
epoch_train_losses = []
epoch_train_accs = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', leave=True)
    
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    epoch_train_losses.append(epoch_loss)
    epoch_train_accs.append(epoch_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

print("=" * 60)
print("✓ Training completed!")

# =========================
# EVALUATION WITH DETAILED METRICS
# =========================
print("\n[6/7] Evaluating model...")
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc='Testing'):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

test_acc = 100 * (all_preds == all_labels).sum() / len(all_labels)
print(f"\n✓ Test Accuracy: {test_acc:.2f}%")

# Classification Report
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(all_labels, all_preds, 
                          target_names=[f'Digit {i}' for i in range(10)]))

# =========================
# VISUALIZATION & ANALYSIS
# =========================
print("\n[7/7] Generating evaluation plots...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Training curves
ax1 = fig.add_subplot(gs[0, :2])
ax1_twin = ax1.twinx()
epochs_range = range(1, EPOCHS + 1)
ax1.plot(epochs_range, epoch_train_losses, 'b-o', label='Loss', linewidth=2)
ax1_twin.plot(epochs_range, epoch_train_accs, 'r-s', label='Accuracy', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='b')
ax1_twin.set_ylabel('Accuracy (%)', color='r')
ax1.tick_params(axis='y', labelcolor='b')
ax1_twin.tick_params(axis='y', labelcolor='r')
ax1.set_title('Training Progress', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')

# Plot 2: Per-class accuracy
ax2 = fig.add_subplot(gs[0, 2])
class_accs = []
for i in range(10):
    mask = all_labels == i
    class_acc = 100 * (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
    class_accs.append(class_acc)
bars = ax2.bar(range(10), class_accs, color=plt.cm.viridis(np.linspace(0, 1, 10)))
ax2.set_xlabel('Digit')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Per-Class Accuracy', fontweight='bold')
ax2.set_xticks(range(10))
ax2.axhline(test_acc, color='r', linestyle='--', label=f'Overall: {test_acc:.1f}%')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Confusion Matrix
ax3 = fig.add_subplot(gs[1, :])
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
            xticklabels=range(10), yticklabels=range(10),
            cbar_kws={'label': 'Count'})
ax3.set_xlabel('Predicted Digit')
ax3.set_ylabel('True Digit')
ax3.set_title('Confusion Matrix', fontweight='bold', fontsize=12)

# Plot 4: Confidence distribution
ax4 = fig.add_subplot(gs[2, 0])
max_probs = all_probs.max(axis=1)
correct_mask = all_preds == all_labels
ax4.hist([max_probs[correct_mask], max_probs[~correct_mask]], 
         bins=30, label=['Correct', 'Incorrect'], alpha=0.7, color=['green', 'red'])
ax4.set_xlabel('Prediction Confidence')
ax4.set_ylabel('Count')
ax4.set_title('Confidence Distribution', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Error analysis - most confused pairs
ax5 = fig.add_subplot(gs[2, 1])
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
np.fill_diagonal(cm_norm, 0)
confused_pairs = []
for i in range(10):
    for j in range(10):
        if i != j:
            confused_pairs.append((cm_norm[i, j], f'{i}→{j}'))
confused_pairs.sort(reverse=True)
top_confusions = confused_pairs[:10]
pairs = [p[1] for p in top_confusions]
rates = [p[0] * 100 for p in top_confusions]
ax5.barh(pairs, rates, color='coral')
ax5.set_xlabel('Confusion Rate (%)')
ax5.set_title('Top 10 Confused Digit Pairs', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

# Plot 6: Sample predictions
ax6 = fig.add_subplot(gs[2, 2])
wrong_indices = np.where(all_preds != all_labels)[0]
if len(wrong_indices) > 0:
    sample_idx = np.random.choice(wrong_indices)
    sample_img, true_label = test_ds[sample_idx]
    pred_label = all_preds[sample_idx]
    confidence = all_probs[sample_idx][pred_label]
    
    img_display = sample_img.permute(1, 2, 0).cpu().numpy()
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
    
    ax6.imshow(img_display)
    ax6.set_title(f'Misclassification Example\nTrue: {true_label}, Pred: {pred_label} ({confidence*100:.1f}%)', 
                  fontweight='bold', color='red')
    ax6.axis('off')
else:
    ax6.text(0.5, 0.5, 'Perfect!\nNo errors found', 
             ha='center', va='center', fontsize=14, fontweight='bold')
    ax6.axis('off')

plt.suptitle(f'Vision Transformer Fine-Tuning Results\nTest Accuracy: {test_acc:.2f}%', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(OUTPUT_DIR / 'evaluation_results.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'evaluation_results.png'}")
plt.close()

# =========================
# SAVE DETAILED SUMMARY
# =========================
summary_path = OUTPUT_DIR / 'summary.txt'
with open(summary_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("VISION TRANSFORMER FINE-TUNING SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model: ViT-B/16 (pre-trained on ImageNet)\n")
    f.write(f"Dataset: MNIST (60k train, 10k test)\n")
    f.write(f"Device: {DEVICE}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Learning Rate: {LR}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n\n")
    f.write(f"Parameters:\n")
    f.write(f"  Total: {total_params:,}\n")
    f.write(f"  Frozen: {frozen_params:,}\n")
    f.write(f"  Trainable: {trainable_params:,}\n\n")
    f.write(f"Final Test Accuracy: {test_acc:.2f}%\n\n")
    f.write("Per-Class Accuracy:\n")
    for i, acc in enumerate(class_accs):
        f.write(f"  Digit {i}: {acc:.2f}%\n")
    f.write("\n" + classification_report(all_labels, all_preds, 
                                         target_names=[f'Digit {i}' for i in range(10)]))

print(f"✓ Saved: {summary_path}")

print("\n" + "=" * 60)
print("ALL DONE! 🎉")
print("=" * 60)
print(f"Results saved in: {OUTPUT_DIR}/")
print(f"  • model_architecture.png - Architecture visualization")
print(f"  • evaluation_results.png - Training & evaluation metrics")
print(f"  • summary.txt - Detailed text summary")
print("=" * 60)