#!/usr/bin/env python3
"""Data preparation script for wellbore image generation"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data import (
    WellboreImageDataset, ImagePreprocessor, DatasetAnalyzer,
    split_dataset, validate_dataset
)
from utils import setup_logging, ensure_dir

def prepare_raw_data(input_dir: str, output_dir: str, image_size: int = 256) -> Dict[str, Any]:
    """Prepare raw wellbore images for training"""
    logging.info(f"Preparing raw data from {input_dir} to {output_dir}")
    
    # Validate input directory
    validation_result = validate_dataset(input_dir)
    if not validation_result['valid']:
        raise ValueError(f"Invalid dataset: {validation_result['error']}")
    
    logging.info(f"Dataset validation passed: {validation_result['total_images']} images found")
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=image_size)
    
    # Preprocess images
    result = preprocessor.batch_preprocess(input_dir, output_dir)
    
    logging.info(f"Preprocessing completed: {result}")
    
    return result

def analyze_dataset(data_path: str) -> Dict[str, Any]:
    """Analyze dataset and print statistics"""
    logging.info(f"Analyzing dataset: {data_path}")
    
    # Create dataset
    dataset = WellboreImageDataset(
        data_path=data_path,
        image_size=256,
        augment=False  # No augmentation for analysis
    )
    
    # Analyze dataset
    analyzer = DatasetAnalyzer(dataset)
    stats = analyzer.analyze()
    
    # Print analysis
    analyzer.print_analysis()
    
    return stats

def create_train_val_test_splits(data_path: str, output_dir: str,
                                train_ratio: float = 0.8,
                                val_ratio: float = 0.1,
                                test_ratio: float = 0.1) -> Dict[str, List[str]]:
    """Create train/validation/test splits"""
    logging.info(f"Creating dataset splits: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # Create splits
    splits = split_dataset(
        data_path=data_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        output_dir=output_dir
    )
    
    # Create symbolic links or copy files to split directories
    create_split_directories(data_path, output_dir, splits)
    
    return splits

def create_split_directories(data_path: str, output_dir: str, 
                           splits: Dict[str, List[str]]) -> None:
    """Create directory structure for splits"""
    logging.info("Creating split directories...")
    
    data_path = Path(data_path)
    output_path = Path(output_dir)
    
    for split_name, file_list in splits.items():
        split_dir = output_path / split_name
        ensure_dir(str(split_dir))
        
        logging.info(f"Creating {split_name} directory with {len(file_list)} images")
        
        # Copy or create symbolic links
        for i, file_path in enumerate(file_list):
            src_path = Path(file_path)
            dst_path = split_dir / f"{i:06d}_{src_path.name}"
            
            try:
                # Try to create symbolic link first (faster)
                if os.name != 'nt':  # Unix-like systems
                    dst_path.symlink_to(src_path.absolute())
                else:  # Windows - copy file
                    import shutil
                    shutil.copy2(src_path, dst_path)
            except Exception as e:
                logging.warning(f"Could not link/copy {src_path}: {e}")
        
        logging.info(f"Created {split_name} directory: {split_dir}")

def clean_dataset(data_path: str, output_dir: str, 
                 min_size: int = 64, max_size: int = 4096) -> Dict[str, Any]:
    """Clean dataset by removing corrupted or invalid images"""
    logging.info(f"Cleaning dataset: {data_path}")
    
    data_path = Path(data_path)
    output_path = Path(output_dir)
    ensure_dir(str(output_path))
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(data_path.rglob(f'*{ext}'))
        image_files.extend(data_path.rglob(f'*{ext.upper()}'))
    
    logging.info(f"Found {len(image_files)} images to check")
    
    # Check each image
    valid_images = []
    corrupted_images = []
    invalid_size_images = []
    
    for image_file in image_files:
        try:
            from PIL import Image
            
            with Image.open(image_file) as img:
                # Check if image can be loaded
                img.verify()
                
                # Reopen for size check (verify() closes the image)
                with Image.open(image_file) as img:
                    width, height = img.size
                    
                    # Check size constraints
                    if (min_size <= width <= max_size and 
                        min_size <= height <= max_size):
                        valid_images.append(image_file)
                    else:
                        invalid_size_images.append(image_file)
                        
        except Exception as e:
            logging.warning(f"Corrupted image {image_file}: {e}")
            corrupted_images.append(image_file)
    
    # Copy valid images to output directory
    logging.info(f"Copying {len(valid_images)} valid images...")
    
    for i, image_file in enumerate(valid_images):
        relative_path = image_file.relative_to(data_path)
        output_file = output_path / relative_path
        
        # Create parent directories
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        import shutil
        shutil.copy2(image_file, output_file)
        
        if (i + 1) % 1000 == 0:
            logging.info(f"Copied {i + 1}/{len(valid_images)} images")
    
    # Save cleaning report
    report = {
        'total_images': len(image_files),
        'valid_images': len(valid_images),
        'corrupted_images': len(corrupted_images),
        'invalid_size_images': len(invalid_size_images)
    }
    
    report_file = output_path / 'cleaning_report.txt'
    with open(report_file, 'w') as f:
        f.write("Dataset Cleaning Report\n")
        f.write("=" * 30 + "\n\n")
        
        for key, value in report.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nCorrupted Images:\n")
        for img in corrupted_images:
            f.write(f"  {img}\n")
        
        f.write("\nInvalid Size Images:\n")
        for img in invalid_size_images:
            f.write(f"  {img}\n")
    
    logging.info(f"Cleaning completed. Report saved: {report_file}")
    
    return report

def create_sample_dataset(output_dir: str, num_samples: int = 1000,
                         image_size: int = 256) -> None:
    """Create a sample synthetic dataset for testing"""
    logging.info(f"Creating sample dataset with {num_samples} images")
    
    import numpy as np
    from PIL import Image, ImageDraw
    import random
    
    output_path = Path(output_dir)
    ensure_dir(str(output_path))
    
    # Create sample images with different patterns
    patterns = ['circles', 'lines', 'noise', 'gradients']
    
    for i in range(num_samples):
        # Choose random pattern
        pattern = random.choice(patterns)
        
        # Create image
        img = Image.new('RGB', (image_size, image_size), 'white')
        draw = ImageDraw.Draw(img)
        
        if pattern == 'circles':
            # Draw random circles
            for _ in range(random.randint(5, 15)):
                x = random.randint(0, image_size)
                y = random.randint(0, image_size)
                r = random.randint(10, 50)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
        
        elif pattern == 'lines':
            # Draw random lines
            for _ in range(random.randint(10, 30)):
                x1, y1 = random.randint(0, image_size), random.randint(0, image_size)
                x2, y2 = random.randint(0, image_size), random.randint(0, image_size)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                width = random.randint(1, 5)
                draw.line([x1, y1, x2, y2], fill=color, width=width)
        
        elif pattern == 'noise':
            # Create noise pattern
            pixels = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
            img = Image.fromarray(pixels)
        
        elif pattern == 'gradients':
            # Create gradient pattern
            pixels = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            for x in range(image_size):
                for y in range(image_size):
                    pixels[y, x] = [
                        int(255 * x / image_size),
                        int(255 * y / image_size),
                        int(255 * (x + y) / (2 * image_size))
                    ]
            img = Image.fromarray(pixels)
        
        # Save image
        img_path = output_path / f'sample_{i:06d}_{pattern}.png'
        img.save(img_path)
        
        if (i + 1) % 100 == 0:
            logging.info(f"Created {i + 1}/{num_samples} sample images")
    
    logging.info(f"Sample dataset created: {output_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Data preparation for wellbore image generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--action',
        type=str,
        choices=['preprocess', 'analyze', 'split', 'clean', 'sample'],
        required=True,
        help='Action to perform'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Input directory containing raw images'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for processed data'
    )
    
    parser.add_argument(
        '--image-size',
        type=int,
        default=256,
        help='Target image size (square)'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio for splitting'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio for splitting'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test set ratio for splitting'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of sample images to create'
    )
    
    parser.add_argument(
        '--min-size',
        type=int,
        default=64,
        help='Minimum image size for cleaning'
    )
    
    parser.add_argument(
        '--max-size',
        type=int,
        default=4096,
        help='Maximum image size for cleaning'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    logging.info(f"Starting data preparation: {args.action}")
    
    try:
        if args.action == 'preprocess':
            if not args.input_dir:
                raise ValueError("Input directory required for preprocessing")
            
            result = prepare_raw_data(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                image_size=args.image_size
            )
            
            print(f"\nPreprocessing Results:")
            print(f"Total images: {result['total']}")
            print(f"Processed: {result['processed']}")
            print(f"Failed: {result['failed']}")
        
        elif args.action == 'analyze':
            if not args.input_dir:
                raise ValueError("Input directory required for analysis")
            
            stats = analyze_dataset(args.input_dir)
        
        elif args.action == 'split':
            if not args.input_dir:
                raise ValueError("Input directory required for splitting")
            
            splits = create_train_val_test_splits(
                data_path=args.input_dir,
                output_dir=args.output_dir,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio
            )
            
            print(f"\nDataset Splits:")
            for split_name, file_list in splits.items():
                print(f"{split_name}: {len(file_list)} images")
        
        elif args.action == 'clean':
            if not args.input_dir:
                raise ValueError("Input directory required for cleaning")
            
            report = clean_dataset(
                data_path=args.input_dir,
                output_dir=args.output_dir,
                min_size=args.min_size,
                max_size=args.max_size
            )
            
            print(f"\nCleaning Results:")
            for key, value in report.items():
                print(f"{key}: {value}")
        
        elif args.action == 'sample':
            create_sample_dataset(
                output_dir=args.output_dir,
                num_samples=args.num_samples,
                image_size=args.image_size
            )
            
            print(f"\nSample dataset created with {args.num_samples} images")
        
        logging.info(f"Data preparation completed successfully")
        
    except Exception as e:
        logging.error(f"Error during data preparation: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()