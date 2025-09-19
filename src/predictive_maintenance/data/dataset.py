"""Dataset class for wellbore images"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
import json
from pathlib import Path

class WellboreImageDataset(Dataset):
    """Dataset for wellbore images with failure type annotations"""
    
    def __init__(self, 
                 data_dir: str,
                 image_size: int = 256,
                 transform: Optional[Callable] = None,
                 failure_types: Optional[List[str]] = None,
                 split: str = 'train',
                 load_annotations: bool = True):
        """
        Args:
            data_dir: Directory containing wellbore images
            image_size: Target image size for resizing
            transform: Optional transform to be applied on images
            failure_types: List of failure types to include
            split: Dataset split ('train', 'val', 'test')
            load_annotations: Whether to load failure type annotations
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.transform = transform
        self.split = split
        self.load_annotations = load_annotations
        
        # Default failure types
        if failure_types is None:
            self.failure_types = [
                'normal',
                'breakouts', 
                'drilling_induced_fractures',
                'washouts',
                'casing_corrosion'
            ]
        else:
            self.failure_types = failure_types
        
        # Create label mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(self.failure_types)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Load image paths and annotations
        self.image_paths, self.annotations = self._load_data()
        
        print(f"Loaded {len(self.image_paths)} images for {split} split")
        if self.load_annotations:
            print(f"Failure type distribution: {self._get_class_distribution()}")
    
    def _load_data(self) -> Tuple[List[str], List[Dict]]:
        """Load image paths and annotations"""
        image_paths = []
        annotations = []
        
        # Look for images in subdirectories organized by failure type
        for failure_type in self.failure_types:
            failure_dir = self.data_dir / failure_type / self.split
            
            if failure_dir.exists():
                # Load images from failure type directory
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                    for img_path in failure_dir.glob(ext):
                        image_paths.append(str(img_path))
                        
                        if self.load_annotations:
                            annotations.append({
                                'failure_type': failure_type,
                                'label': self.label_to_idx[failure_type],
                                'image_path': str(img_path)
                            })
        
        # If no organized structure, load from annotations file
        if not image_paths:
            annotations_file = self.data_dir / f"{self.split}_annotations.json"
            if annotations_file.exists() and self.load_annotations:
                with open(annotations_file, 'r') as f:
                    annotation_data = json.load(f)
                
                for item in annotation_data:
                    img_path = self.data_dir / item['image_path']
                    if img_path.exists():
                        image_paths.append(str(img_path))
                        annotations.append({
                            'failure_type': item['failure_type'],
                            'label': self.label_to_idx.get(item['failure_type'], 0),
                            'image_path': str(img_path),
                            **item  # Include any additional metadata
                        })
            else:
                # Load all images without annotations
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                    for img_path in self.data_dir.glob(f"**/{ext}"):
                        image_paths.append(str(img_path))
                        if self.load_annotations:
                            annotations.append({
                                'failure_type': 'unknown',
                                'label': -1,
                                'image_path': str(img_path)
                            })
        
        return image_paths, annotations
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of failure types in dataset"""
        distribution = {failure_type: 0 for failure_type in self.failure_types}
        
        for annotation in self.annotations:
            failure_type = annotation['failure_type']
            if failure_type in distribution:
                distribution[failure_type] += 1
        
        return distribution
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get image and label at index"""
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Resize image
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Convert to tensor
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        
        # Normalize to [-1, 1] for GAN training
        image = (image / 127.5) - 1.0
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        if self.load_annotations and idx < len(self.annotations):
            label = torch.tensor(self.annotations[idx]['label'], dtype=torch.long)
        else:
            label = torch.tensor(-1, dtype=torch.long)  # Unknown label
        
        return image, label
    
    def get_image_info(self, idx: int) -> Dict:
        """Get detailed information about an image"""
        info = {
            'image_path': self.image_paths[idx],
            'index': idx
        }
        
        if self.load_annotations and idx < len(self.annotations):
            info.update(self.annotations[idx])
        
        return info
    
    def filter_by_failure_type(self, failure_types: List[str]) -> 'WellboreImageDataset':
        """Create a filtered dataset with specific failure types"""
        filtered_paths = []
        filtered_annotations = []
        
        for i, annotation in enumerate(self.annotations):
            if annotation['failure_type'] in failure_types:
                filtered_paths.append(self.image_paths[i])
                filtered_annotations.append(annotation)
        
        # Create new dataset instance
        new_dataset = WellboreImageDataset.__new__(WellboreImageDataset)
        new_dataset.data_dir = self.data_dir
        new_dataset.image_size = self.image_size
        new_dataset.transform = self.transform
        new_dataset.split = self.split
        new_dataset.load_annotations = self.load_annotations
        new_dataset.failure_types = failure_types
        new_dataset.label_to_idx = {label: idx for idx, label in enumerate(failure_types)}
        new_dataset.idx_to_label = {idx: label for label, idx in new_dataset.label_to_idx.items()}
        new_dataset.image_paths = filtered_paths
        new_dataset.annotations = filtered_annotations
        
        return new_dataset
    
    @staticmethod
    def create_train_val_split(data_dir: str, val_ratio: float = 0.2, 
                              random_seed: int = 42) -> Tuple['WellboreImageDataset', 'WellboreImageDataset']:
        """Create train/validation split from a single directory"""
        import random
        random.seed(random_seed)
        
        # Load all data
        full_dataset = WellboreImageDataset(data_dir, split='all')
        
        # Create indices for split
        indices = list(range(len(full_dataset)))
        random.shuffle(indices)
        
        val_size = int(len(indices) * val_ratio)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        # Create train dataset
        train_dataset = WellboreImageDataset.__new__(WellboreImageDataset)
        train_dataset.__dict__.update(full_dataset.__dict__)
        train_dataset.split = 'train'
        train_dataset.image_paths = [full_dataset.image_paths[i] for i in train_indices]
        train_dataset.annotations = [full_dataset.annotations[i] for i in train_indices]
        
        # Create validation dataset
        val_dataset = WellboreImageDataset.__new__(WellboreImageDataset)
        val_dataset.__dict__.update(full_dataset.__dict__)
        val_dataset.split = 'val'
        val_dataset.image_paths = [full_dataset.image_paths[i] for i in val_indices]
        val_dataset.annotations = [full_dataset.annotations[i] for i in val_indices]
        
        return train_dataset, val_dataset

class SyntheticWellboreDataset(Dataset):
    """Dataset for generating synthetic wellbore images on-the-fly"""
    
    def __init__(self, 
                 generator_model: torch.nn.Module,
                 num_samples: int = 10000,
                 latent_dim: int = 512,
                 device: torch.device = torch.device('cpu'),
                 failure_type: Optional[str] = None):
        """
        Args:
            generator_model: Trained StyleGAN2 generator
            num_samples: Number of synthetic samples to generate
            latent_dim: Dimension of latent space
            device: Device to run generation on
            failure_type: Specific failure type to generate (if conditional)
        """
        self.generator = generator_model
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.device = device
        self.failure_type = failure_type
        
        # Pre-generate latent codes for consistency
        self.latent_codes = torch.randn(num_samples, latent_dim, device=device)
        
        # Set generator to evaluation mode
        self.generator.eval()
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic image at index"""
        with torch.no_grad():
            # Get latent code
            latent_code = self.latent_codes[idx:idx+1]
            
            # Generate noise inputs
            noise_inputs = self.generator.generate_noise(1, self.device)
            
            # Generate image
            synthetic_image = self.generator(latent_code, noise_inputs=noise_inputs)
            
            # Remove batch dimension
            synthetic_image = synthetic_image.squeeze(0)
            
            # Create dummy label (for compatibility)
            label = torch.tensor(0, dtype=torch.long)
        
        return synthetic_image, label
    
    def generate_batch(self, batch_size: int) -> torch.Tensor:
        """Generate a batch of synthetic images"""
        with torch.no_grad():
            # Sample random latent codes
            latent_codes = torch.randn(batch_size, self.latent_dim, device=self.device)
            
            # Generate noise inputs
            noise_inputs = self.generator.generate_noise(batch_size, self.device)
            
            # Generate images
            synthetic_images = self.generator(latent_codes, noise_inputs=noise_inputs)
        
        return synthetic_images