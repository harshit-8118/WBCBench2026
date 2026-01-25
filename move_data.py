import os
import shutil
import random

# ===================== CONFIG =====================
DATASET_DIR = "yolo_cell_detection_dataset"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")

TRAIN_RATIO = 0.7
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
SEED = 42
# ==================================================

OUTPUT_DIR = "yolo_detection_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)

# Create output directories
for split in ["train", "valid"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

# Collect image files
images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(IMAGE_EXTS)]
images.sort()
random.shuffle(images)

# Split
split_idx = int(len(images) * TRAIN_RATIO)
train_images = images[:split_idx]
val_images = images[split_idx:]

def move_files(image_list, split):
    for img in image_list:
        name, _ = os.path.splitext(img)
        img_src = os.path.join(IMAGE_DIR, img)
        lbl_src = os.path.join(LABEL_DIR, name + ".txt")

        img_dst = os.path.join(OUTPUT_DIR, split, "images", img)
        lbl_dst = os.path.join(OUTPUT_DIR, split, "labels", name + ".txt")

        shutil.copy2(img_src, img_dst)

        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, lbl_dst)
        else:
            print(f"⚠️ Missing label for {img}")

move_files(train_images, "train")
move_files(val_images, "valid")

print(f"✅ Done!")
print(f"Train images: {len(train_images)}")
print(f"Valid images: {len(val_images)}")
