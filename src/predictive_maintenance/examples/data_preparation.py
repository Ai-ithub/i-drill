#!/usr/bin/env python3
"""
Data Preparation Examples for Wellbore Image Generation System

This script demonstrates how to prepare and preprocess wellbore images
for training the GAN model.
"""

import os
import sys
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Tuple

# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocessing import ImagePreprocessor
from utils.file_utils import ensure_dir_exists, get_files_by_extension
from utils.image_utils import load_image, save_image, resize_image, normalize_image


def organize_dataset(source_dir: str, output_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    Organize images into train/val/test splits
    
    Args:
        source_dir: Directory containing all images
        output_dir: Output directory for organized dataset
        train_ratio: Ratio of images for training
        val_ratio: Ratio of images for validation
    """
    print(f"Organizing dataset from {source_dir} to {output_dir}")
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    
    for dir_path in [train_dir, val_dir, test_dir]:
        ensure_dir_exists(dir_path)
    
    # Get all image files
    image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(get_files_by_extension(source_dir, ext))
    
    print(f"Found {len(all_images)} images")
    
    # Shuffle images for random split
    np.random.shuffle(all_images)
    
    # Calculate split indices
    total_images = len(all_images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)
    
    # Split images
    train_images = all_images[:train_end]
    val_images = all_images[train_end:val_end]
    test_images = all_images[val_end:]
    
    # Copy images to respective directories
    def copy_images(image_list: List[str], target_dir: str, split_name: str):
        print(f"Copying {len(image_list)} images to {split_name} set...")
        for i, img_path in enumerate(image_list):
            filename = os.path.basename(img_path)
            target_path = os.path.join(target_dir, filename)
            shutil.copy2(img_path, target_path)
            
            if (i + 1) % 100 == 0:
                print(f"  Copied {i + 1}/{len(image_list)} images")
    
    copy_images(train_images, train_dir, "train")
    copy_images(val_images, val_dir, "validation")
    copy_images(test_images, test_dir, "test")
    
    print(f"\nDataset organization completed:")
    print(f"  Training: {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")


def preprocess_images(input_dir: str, output_dir: str, target_size: Tuple[int, int] = (256, 256)):
    """
    Preprocess images (resize, normalize, etc.)
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory for preprocessed images
        target_size: Target image size (width, height)
    """
    print(f"Preprocessing images from {input_dir} to {output_dir}")
    print(f"Target size: {target_size}")
    
    ensure_dir_exists(output_dir)
    
    # Get all image files
    image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(get_files_by_extension(input_dir, ext))
    
    print(f"Found {len(all_images)} images to preprocess")
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(
        target_size=target_size,
        normalize=True,
        augment=False  # No augmentation for preprocessing
    )
    
    processed_count = 0
    skipped_count = 0
    
    for i, img_path in enumerate(all_images):
        try:
            # Load image
            image = load_image(img_path)
            
            # Check if image is valid
            if image is None:
                print(f"  Skipping invalid image: {img_path}")
                skipped_count += 1
                continue
            
            # Check minimum size
            if min(image.size) < 64:
                print(f"  Skipping too small image: {img_path} (size: {image.size})")
                skipped_count += 1
                continue
            
            # Preprocess image
            processed_image = preprocessor.preprocess_single(image)
            
            # Convert back to PIL for saving
            if isinstance(processed_image, np.ndarray):
                # Convert from numpy array to PIL Image
                if processed_image.dtype != np.uint8:
                    processed_image = (processed_image * 255).astype(np.uint8)
                
                if len(processed_image.shape) == 3:
                    processed_image = Image.fromarray(processed_image)
                else:
                    processed_image = Image.fromarray(processed_image, mode='L')
            
            # Save processed image
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}.png")  # Save as PNG
            
            processed_image.save(output_path, "PNG")
            processed_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(all_images)} images")
        
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")
            skipped_count += 1
    
    print(f"\nPreprocessing completed:")
    print(f"  Processed: {processed_count} images")
    print(f"  Skipped: {skipped_count} images")


def analyze_dataset(data_dir: str):
    """
    Analyze dataset statistics
    
    Args:
        data_dir: Directory containing images
    """
    print(f"Analyzing dataset: {data_dir}")
    
    # Get all image files
    image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(get_files_by_extension(data_dir, ext))
    
    if not all_images:
        print("No images found!")
        return
    
    print(f"Found {len(all_images)} images")
    
    # Analyze image properties
    sizes = []
    aspects = []
    formats = []
    
    for img_path in all_images[:1000]:  # Sample first 1000 images
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                sizes.append((width, height))
                aspects.append(width / height)
                formats.append(img.format)
        except Exception as e:
            print(f"Error analyzing {img_path}: {e}")
    
    # Calculate statistics
    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    
    print(f"\nDataset Statistics (based on {len(sizes)} samples):")
    print(f"  Image count: {len(all_images)}")
    print(f"  Width - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.1f}")
    print(f"  Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.1f}")
    print(f"  Aspect ratio - Min: {min(aspects):.2f}, Max: {max(aspects):.2f}, Mean: {np.mean(aspects):.2f}")
    
    # Format distribution
    format_counts = {}
    for fmt in formats:
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
    
    print(f"  Formats: {format_counts}")
    
    # Size distribution
    size_ranges = {
        "< 128": 0,
        "128-256": 0,
        "256-512": 0,
        "512-1024": 0,
        "> 1024": 0
    }
    
    for w, h in sizes:
        min_dim = min(w, h)
        if min_dim < 128:
            size_ranges["< 128"] += 1
        elif min_dim < 256:
            size_ranges["128-256"] += 1
        elif min_dim < 512:
            size_ranges["256-512"] += 1
        elif min_dim < 1024:
            size_ranges["512-1024"] += 1
        else:
            size_ranges["> 1024"] += 1
    
    print(f"  Size distribution (by minimum dimension): {size_ranges}")


def clean_dataset(data_dir: str, min_size: int = 64, max_size: int = 2048):
    """
    Clean dataset by removing invalid or problematic images
    
    Args:
        data_dir: Directory containing images
        min_size: Minimum image dimension
        max_size: Maximum image dimension
    """
    print(f"Cleaning dataset: {data_dir}")
    print(f"Size constraints: {min_size} <= min_dimension <= {max_size}")
    
    # Get all image files
    image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(get_files_by_extension(data_dir, ext))
    
    print(f"Found {len(all_images)} images")
    
    removed_count = 0
    reasons = {
        "corrupted": 0,
        "too_small": 0,
        "too_large": 0,
        "wrong_format": 0
    }
    
    for img_path in all_images:
        should_remove = False
        reason = None
        
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                min_dim = min(width, height)
                max_dim = max(width, height)
                
                # Check size constraints
                if min_dim < min_size:
                    should_remove = True
                    reason = "too_small"
                elif max_dim > max_size:
                    should_remove = True
                    reason = "too_large"
                
                # Check if image can be loaded properly
                img.verify()
        
        except Exception as e:
            should_remove = True
            reason = "corrupted"
            print(f"  Corrupted image: {img_path} - {e}")
        
        if should_remove:
            try:
                os.remove(img_path)
                removed_count += 1
                reasons[reason] += 1
                
                if removed_count % 100 == 0:
                    print(f"  Removed {removed_count} images so far...")
            
            except Exception as e:
                print(f"  Error removing {img_path}: {e}")
    
    print(f"\nCleaning completed:")
    print(f"  Removed: {removed_count} images")
    print(f"  Remaining: {len(all_images) - removed_count} images")
    print(f"  Removal reasons: {reasons}")


def main():
    """
    Main function to run data preparation examples
    """
    print("Wellbore Image Generation - Data Preparation")
    print("============================================\n")
    
    # Example paths (update these according to your setup)
    raw_data_dir = "raw_data"  # Directory with all your raw images
    organized_data_dir = "data"  # Output directory for organized dataset
    preprocessed_data_dir = "data_preprocessed"  # Output for preprocessed images
    
    print("Available operations:")
    print("1. Organize dataset into train/val/test splits")
    print("2. Preprocess images (resize, normalize)")
    print("3. Analyze dataset statistics")
    print("4. Clean dataset (remove invalid images)")
    print("\nUpdate the paths in this script and uncomment the operations you need.\n")
    
    # Example usage (uncomment as needed):
    
    # 1. Organize raw images into train/val/test splits
    # if os.path.exists(raw_data_dir):
    #     organize_dataset(raw_data_dir, organized_data_dir)
    # else:
    #     print(f"Raw data directory not found: {raw_data_dir}")
    
    # 2. Analyze dataset
    # for split in ["train", "val", "test"]:
    #     split_dir = os.path.join(organized_data_dir, split)
    #     if os.path.exists(split_dir):
    #         print(f"\n--- Analyzing {split} set ---")
    #         analyze_dataset(split_dir)
    
    # 3. Clean dataset
    # for split in ["train", "val", "test"]:
    #     split_dir = os.path.join(organized_data_dir, split)
    #     if os.path.exists(split_dir):
    #         print(f"\n--- Cleaning {split} set ---")
    #         clean_dataset(split_dir)
    
    # 4. Preprocess images
    # for split in ["train", "val", "test"]:
    #     input_dir = os.path.join(organized_data_dir, split)
    #     output_dir = os.path.join(preprocessed_data_dir, split)
    #     
    #     if os.path.exists(input_dir):
    #         print(f"\n--- Preprocessing {split} set ---")
    #         preprocess_images(input_dir, output_dir, target_size=(256, 256))
    
    print("Data preparation script ready!")
    print("\nTo use this script:")
    print("1. Update the directory paths above")
    print("2. Uncomment the operations you want to perform")
    print("3. Run the script")


if __name__ == "__main__":
    main()