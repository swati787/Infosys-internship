import os
import shutil
import random

source_dir = "../data/frames"
target_dir = "../data/dataset"

classes = ["walking", "running", "cycling"]
split_ratio = 0.8  # 80% train, 20% test

for cls in classes:
    images = os.listdir(os.path.join(source_dir, cls))
    random.shuffle(images)

    split_point = int(len(images) * split_ratio)
    train_images = images[:split_point]
    test_images = images[split_point:]

    for img in train_images:
        src = os.path.join(source_dir, cls, img)
        dst = os.path.join(target_dir, "train", cls, img)
        shutil.copy(src, dst)

    for img in test_images:
        src = os.path.join(source_dir, cls, img)
        dst = os.path.join(target_dir, "test", cls, img)
        shutil.copy(src, dst)

print("✅ Dataset split completed successfully")
