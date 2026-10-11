#!/usr/bin/env python3
"""
Debug script to test dataset loading
"""

from pathlib import Path

from core.training.improved_training import ImprovedInriaDataset

# Test dataset initialization
print("Testing dataset initialization...")

images_dir = Path('./dataset/train/images')
gt_dir = Path('./dataset/train/gt')

print(f"Images dir exists: {images_dir.exists()}")
print(f"GT dir exists: {gt_dir.exists()}")

if images_dir.exists():
    image_files = sorted(list(images_dir.glob('*.tif')))
    print(f"Found {len(image_files)} image files")
    if image_files:
        print(f"First few: {[f.name for f in image_files[:3]]}")

if gt_dir.exists():
    gt_files = sorted(list(gt_dir.glob('*.tif')))
    print(f"Found {len(gt_files)} GT files")
    if gt_files:
        print(f"First few: {[f.name for f in gt_files[:3]]}")

# Try creating dataset
try:
    dataset = ImprovedInriaDataset(
        images_dir,
        gt_dir,
        image_size=512,
        mode='train',
        crop_patches=True
    )
    print(f"Dataset created successfully with {len(dataset)} samples")
except Exception as e:
    print(f"Error creating dataset: {e}")
    import traceback
    traceback.print_exc()