#!/usr/bin/env python3
"""Data handling and preprocessing for wellbore image generation"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class WellboreImageDataset(Dataset):
    """Dataset class for wellbore images"""
    
    def __init__(self, data_path: str, image_size: int = 256, 
                 augment: bool = True, normalize: bool = True):
        """
        Initialize dataset
        
        Args:
            data_path: Path to directory containing images
            image_size: Target image size (square)
            augment: Whether to apply data augmentation
            normalize: Whether to normalize images to [-1, 1]
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.normalize = normalize
        
        # Find all image files
        self.image_paths = self._find_images()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_path}")
        
        logging.info(f"Found {len(self.image_paths)} images in {data_path}")
        
        # Setup transforms
        self.transform = self._create_transforms(augment)
        
    def _find_images(self) -> List[Path]:
        """Find all image files in the data directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(self.data_path.rglob(f'*{ext}'))
            image_paths.extend(self.data_path.rglob(f'*{ext.upper()}'))
        
        return sorted(image_paths)
    
    def _create_transforms(self, augment: bool) -> transforms.Compose:
        """Create image transforms"""
        transform_list = []
        
        # Resize to target size
        transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        
        # Data augmentation
        if augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                )
            ])
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize to [-1, 1] if requested
        if self.normalize:
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get image by index"""
        image_path = self.image_paths[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            image = self.transform(image)
            
            return image
            
        except Exception as e:
            logging.warning(f"Error loading image {image_path}: {e}")
            # Return a random image instead
            return self.__getitem__((idx + 1) % len(self.image_paths))
    
    def get_image_info(self, idx: int) -> Dict[str, Any]:
        """Get information about an image"""
        image_path = self.image_paths[idx]
        
        try:
            with Image.open(image_path) as img:
                return {
                    'path': str(image_path),
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format
                }
        except Exception as e:
            return {
                'path': str(image_path),
                'error': str(e)
            }

class ImagePreprocessor:
    """Image preprocessing utilities"""
    
    def __init__(self, target_size: int = 256):
        self.target_size = target_size
    
    def preprocess_image(self, image_path: str, output_path: str) -> bool:
        """Preprocess a single image"""
        try:
            # Load image
            with Image.open(image_path) as img:
                # Convert to RGB
                img = img.convert('RGB')
                
                # Resize maintaining aspect ratio
                img = self._resize_with_padding(img)
                
                # Save processed image
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                img.save(output_path, 'PNG', quality=95)
                
                return True
                
        except Exception as e:
            logging.error(f"Error preprocessing {image_path}: {e}")
            return False
    
    def _resize_with_padding(self, image: Image.Image) -> Image.Image:
        """Resize image with padding to maintain aspect ratio"""
        # Calculate scaling factor
        scale = min(self.target_size / image.width, self.target_size / image.height)
        
        # Calculate new size
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        
        # Resize image
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with padding
        new_image = Image.new('RGB', (self.target_size, self.target_size), (255, 255, 255))
        
        # Calculate padding
        x_offset = (self.target_size - new_width) // 2
        y_offset = (self.target_size - new_height) // 2
        
        # Paste resized image onto padded image
        new_image.paste(image, (x_offset, y_offset))
        
        return new_image
    
    def batch_preprocess(self, input_dir: str, output_dir: str) -> Dict[str, int]:
        """Preprocess all images in a directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.rglob(f'*{ext}'))
            image_files.extend(input_path.rglob(f'*{ext.upper()}'))
        
        logging.info(f"Found {len(image_files)} images to preprocess")
        
        # Process images
        processed = 0
        failed = 0
        
        for image_file in image_files:
            # Create output path
            relative_path = image_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix('.png')
            
            # Preprocess image
            if self.preprocess_image(str(image_file), str(output_file)):
                processed += 1
            else:
                failed += 1
            
            if (processed + failed) % 100 == 0:
                logging.info(f"Processed {processed + failed}/{len(image_files)} images")
        
        logging.info(f"Preprocessing complete: {processed} successful, {failed} failed")
        
        return {
            'total': len(image_files),
            'processed': processed,
            'failed': failed
        }

class DatasetAnalyzer:
    """Analyze dataset statistics"""
    
    def __init__(self, dataset: WellboreImageDataset):
        self.dataset = dataset
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze dataset and return statistics"""
        logging.info("Analyzing dataset...")
        
        stats = {
            'total_images': len(self.dataset),
            'image_sizes': [],
            'image_modes': {},
            'image_formats': {},
            'corrupted_images': 0
        }
        
        # Analyze each image
        for i in range(len(self.dataset)):
            info = self.dataset.get_image_info(i)
            
            if 'error' in info:
                stats['corrupted_images'] += 1
                continue
            
            # Collect size information
            stats['image_sizes'].append(info['size'])
            
            # Count modes
            mode = info['mode']
            stats['image_modes'][mode] = stats['image_modes'].get(mode, 0) + 1
            
            # Count formats
            format_name = info['format']
            stats['image_formats'][format_name] = stats['image_formats'].get(format_name, 0) + 1
        
        # Calculate size statistics
        if stats['image_sizes']:
            widths = [size[0] for size in stats['image_sizes']]
            heights = [size[1] for size in stats['image_sizes']]
            
            stats['size_stats'] = {
                'width_min': min(widths),
                'width_max': max(widths),
                'width_mean': np.mean(widths),
                'height_min': min(heights),
                'height_max': max(heights),
                'height_mean': np.mean(heights)
            }
        
        return stats
    
    def print_analysis(self) -> None:
        """Print dataset analysis"""
        stats = self.analyze()
        
        print("\n" + "="*50)
        print("DATASET ANALYSIS")
        print("="*50)
        
        print(f"Total images: {stats['total_images']}")
        print(f"Corrupted images: {stats['corrupted_images']}")
        
        if 'size_stats' in stats:
            size_stats = stats['size_stats']
            print(f"\nImage sizes:")
            print(f"  Width: {size_stats['width_min']}-{size_stats['width_max']} (avg: {size_stats['width_mean']:.1f})")
            print(f"  Height: {size_stats['height_min']}-{size_stats['height_max']} (avg: {size_stats['height_mean']:.1f})")
        
        print(f"\nImage modes:")
        for mode, count in stats['image_modes'].items():
            print(f"  {mode}: {count} ({count/stats['total_images']*100:.1f}%)")
        
        print(f"\nImage formats:")
        for format_name, count in stats['image_formats'].items():
            print(f"  {format_name}: {count} ({count/stats['total_images']*100:.1f}%)")
        
        print("="*50)

def create_data_loader(data_path: str, batch_size: int = 32, image_size: int = 256,
                      augment: bool = True, num_workers: int = 4, 
                      shuffle: bool = True) -> DataLoader:
    """Create data loader for training"""
    dataset = WellboreImageDataset(
        data_path=data_path,
        image_size=image_size,
        augment=augment
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader

def split_dataset(data_path: str, train_ratio: float = 0.8, 
                 val_ratio: float = 0.1, test_ratio: float = 0.1,
                 output_dir: str = None) -> Dict[str, List[str]]:
    """Split dataset into train/validation/test sets"""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # Find all images
    data_path = Path(data_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(data_path.rglob(f'*{ext}'))
        image_files.extend(data_path.rglob(f'*{ext.upper()}'))
    
    # Shuffle files
    np.random.shuffle(image_files)
    
    # Calculate split indices
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split files
    splits = {
        'train': [str(f) for f in image_files[:train_end]],
        'val': [str(f) for f in image_files[train_end:val_end]],
        'test': [str(f) for f in image_files[val_end:]]
    }
    
    logging.info(f"Dataset split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    
    # Save split information if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, file_list in splits.items():
            split_file = output_path / f'{split_name}_files.txt'
            with open(split_file, 'w') as f:
                for file_path in file_list:
                    f.write(f'{file_path}\n')
            logging.info(f"Saved {split_name} split to {split_file}")
    
    return splits

def validate_dataset(data_path: str) -> Dict[str, Any]:
    """Validate dataset and return validation results"""
    logging.info(f"Validating dataset: {data_path}")
    
    data_path = Path(data_path)
    if not data_path.exists():
        return {'valid': False, 'error': f'Path does not exist: {data_path}'}
    
    # Find images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(data_path.rglob(f'*{ext}'))
        image_files.extend(data_path.rglob(f'*{ext.upper()}'))
    
    if len(image_files) == 0:
        return {'valid': False, 'error': 'No image files found'}
    
    # Test loading a few images
    test_count = min(10, len(image_files))
    corrupted = 0
    
    for i in range(test_count):
        try:
            with Image.open(image_files[i]) as img:
                img.verify()
        except Exception:
            corrupted += 1
    
    return {
        'valid': True,
        'total_images': len(image_files),
        'tested_images': test_count,
        'corrupted_images': corrupted,
        'corruption_rate': corrupted / test_count if test_count > 0 else 0
    }

if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset utilities')
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--action', type=str, choices=['analyze', 'validate', 'preprocess', 'split'],
                       required=True, help='Action to perform')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--image-size', type=int, default=256, help='Target image size')
    
    args = parser.parse_args()
    
    if args.action == 'analyze':
        dataset = WellboreImageDataset(args.data_path, image_size=args.image_size, augment=False)
        analyzer = DatasetAnalyzer(dataset)
        analyzer.print_analysis()
    
    elif args.action == 'validate':
        result = validate_dataset(args.data_path)
        print(f"Validation result: {result}")
    
    elif args.action == 'preprocess':
        if not args.output_dir:
            raise ValueError("Output directory required for preprocessing")
        preprocessor = ImagePreprocessor(args.image_size)
        result = preprocessor.batch_preprocess(args.data_path, args.output_dir)
        print(f"Preprocessing result: {result}")
    
    elif args.action == 'split':
        splits = split_dataset(args.data_path, output_dir=args.output_dir)
        print(f"Dataset splits: {splits}")