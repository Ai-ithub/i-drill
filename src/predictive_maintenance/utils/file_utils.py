#!/usr/bin/env python3
"""File utilities for wellbore image generation system"""

import os
import shutil
import json
import pickle
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import hashlib
import zipfile
import tarfile
from datetime import datetime
import glob
from PIL import Image
import cv2

def ensure_dir(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't
    
    Args:
        directory: Directory path
        
    Returns:
        Path object of the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path)

def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """Calculate file hash
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        File hash string
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

def copy_file(src: Union[str, Path], dst: Union[str, Path], 
             create_dirs: bool = True) -> bool:
    """Copy file from source to destination
    
    Args:
        src: Source file path
        dst: Destination file path
        create_dirs: Whether to create destination directories
        
    Returns:
        True if successful, False otherwise
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)
        
        if create_dirs:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(src_path, dst_path)
        return True
    except Exception as e:
        logging.error(f"Failed to copy file {src} to {dst}: {e}")
        return False

def move_file(src: Union[str, Path], dst: Union[str, Path], 
             create_dirs: bool = True) -> bool:
    """Move file from source to destination
    
    Args:
        src: Source file path
        dst: Destination file path
        create_dirs: Whether to create destination directories
        
    Returns:
        True if successful, False otherwise
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)
        
        if create_dirs:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.move(str(src_path), str(dst_path))
        return True
    except Exception as e:
        logging.error(f"Failed to move file {src} to {dst}: {e}")
        return False

def delete_file(file_path: Union[str, Path]) -> bool:
    """Delete file
    
    Args:
        file_path: Path to file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.remove(file_path)
        return True
    except Exception as e:
        logging.error(f"Failed to delete file {file_path}: {e}")
        return False

def delete_directory(dir_path: Union[str, Path], recursive: bool = True) -> bool:
    """Delete directory
    
    Args:
        dir_path: Path to directory
        recursive: Whether to delete recursively
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if recursive:
            shutil.rmtree(dir_path)
        else:
            os.rmdir(dir_path)
        return True
    except Exception as e:
        logging.error(f"Failed to delete directory {dir_path}: {e}")
        return False

def list_files(directory: Union[str, Path], pattern: str = '*', 
              recursive: bool = False) -> List[Path]:
    """List files in directory
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        recursive: Whether to search recursively
        
    Returns:
        List of file paths
    """
    path = Path(directory)
    
    if recursive:
        return list(path.rglob(pattern))
    else:
        return list(path.glob(pattern))

def find_files_by_extension(directory: Union[str, Path], 
                          extensions: Union[str, List[str]], 
                          recursive: bool = True) -> List[Path]:
    """Find files by extension
    
    Args:
        directory: Directory to search
        extensions: File extension(s) to search for
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    if isinstance(extensions, str):
        extensions = [extensions]
    
    # Ensure extensions start with dot
    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
    
    files = []
    for ext in extensions:
        pattern = f'*{ext}'
        files.extend(list_files(directory, pattern, recursive))
    
    return files

def save_json(data: Dict[str, Any], file_path: Union[str, Path], 
             indent: int = 2, create_dirs: bool = True) -> bool:
    """Save data to JSON file
    
    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation
        create_dirs: Whether to create directories
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        return True
    except Exception as e:
        logging.error(f"Failed to save JSON to {file_path}: {e}")
        return False

def load_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load data from JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON from {file_path}: {e}")
        return None

def save_yaml(data: Dict[str, Any], file_path: Union[str, Path], 
             create_dirs: bool = True) -> bool:
    """Save data to YAML file
    
    Args:
        data: Data to save
        file_path: Path to save file
        create_dirs: Whether to create directories
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        return True
    except Exception as e:
        logging.error(f"Failed to save YAML to {file_path}: {e}")
        return False

def load_yaml(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load data from YAML file
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load YAML from {file_path}: {e}")
        return None

def save_pickle(data: Any, file_path: Union[str, Path], 
               create_dirs: bool = True) -> bool:
    """Save data to pickle file
    
    Args:
        data: Data to save
        file_path: Path to save file
        create_dirs: Whether to create directories
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        return True
    except Exception as e:
        logging.error(f"Failed to save pickle to {file_path}: {e}")
        return False

def load_pickle(file_path: Union[str, Path]) -> Optional[Any]:
    """Load data from pickle file
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Failed to load pickle from {file_path}: {e}")
        return None

def save_torch_model(model: torch.nn.Module, file_path: Union[str, Path], 
                    create_dirs: bool = True) -> bool:
    """Save PyTorch model
    
    Args:
        model: PyTorch model
        file_path: Path to save file
        create_dirs: Whether to create directories
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(model.state_dict(), path)
        return True
    except Exception as e:
        logging.error(f"Failed to save model to {file_path}: {e}")
        return False

def load_torch_model(model: torch.nn.Module, file_path: Union[str, Path], 
                    device: str = 'cpu') -> bool:
    """Load PyTorch model
    
    Args:
        model: PyTorch model to load state into
        file_path: Path to model file
        device: Device to load model on
        
    Returns:
        True if successful, False otherwise
    """
    try:
        state_dict = torch.load(file_path, map_location=device)
        model.load_state_dict(state_dict)
        return True
    except Exception as e:
        logging.error(f"Failed to load model from {file_path}: {e}")
        return False

def save_checkpoint(checkpoint_data: Dict[str, Any], file_path: Union[str, Path], 
                   create_dirs: bool = True) -> bool:
    """Save training checkpoint
    
    Args:
        checkpoint_data: Checkpoint data dictionary
        file_path: Path to save checkpoint
        create_dirs: Whether to create directories
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint_data, path)
        return True
    except Exception as e:
        logging.error(f"Failed to save checkpoint to {file_path}: {e}")
        return False

def load_checkpoint(file_path: Union[str, Path], 
                   device: str = 'cpu') -> Optional[Dict[str, Any]]:
    """Load training checkpoint
    
    Args:
        file_path: Path to checkpoint file
        device: Device to load checkpoint on
        
    Returns:
        Checkpoint data or None if failed
    """
    try:
        return torch.load(file_path, map_location=device)
    except Exception as e:
        logging.error(f"Failed to load checkpoint from {file_path}: {e}")
        return None

def create_archive(source_dir: Union[str, Path], archive_path: Union[str, Path], 
                  format: str = 'zip', create_dirs: bool = True) -> bool:
    """Create archive from directory
    
    Args:
        source_dir: Directory to archive
        archive_path: Path for archive file
        format: Archive format ('zip', 'tar', 'tar.gz')
        create_dirs: Whether to create directories
        
    Returns:
        True if successful, False otherwise
    """
    try:
        source_path = Path(source_dir)
        archive_path = Path(archive_path)
        
        if create_dirs:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'zip':
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in source_path.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(source_path)
                        zipf.write(file_path, arcname)
        
        elif format in ['tar', 'tar.gz']:
            mode = 'w:gz' if format == 'tar.gz' else 'w'
            with tarfile.open(archive_path, mode) as tarf:
                tarf.add(source_path, arcname=source_path.name)
        
        else:
            raise ValueError(f"Unsupported archive format: {format}")
        
        return True
    except Exception as e:
        logging.error(f"Failed to create archive {archive_path}: {e}")
        return False

def extract_archive(archive_path: Union[str, Path], extract_dir: Union[str, Path], 
                   create_dirs: bool = True) -> bool:
    """Extract archive to directory
    
    Args:
        archive_path: Path to archive file
        extract_dir: Directory to extract to
        create_dirs: Whether to create directories
        
    Returns:
        True if successful, False otherwise
    """
    try:
        archive_path = Path(archive_path)
        extract_path = Path(extract_dir)
        
        if create_dirs:
            extract_path.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(extract_path)
        
        elif archive_path.suffix in ['.tar', '.gz']:
            with tarfile.open(archive_path, 'r:*') as tarf:
                tarf.extractall(extract_path)
        
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
        
        return True
    except Exception as e:
        logging.error(f"Failed to extract archive {archive_path}: {e}")
        return False

def get_directory_size(directory: Union[str, Path]) -> int:
    """Get total size of directory in bytes
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
    
    return total_size

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"

def clean_directory(directory: Union[str, Path], older_than_days: int = 7, 
                   pattern: str = '*', dry_run: bool = False) -> List[Path]:
    """Clean old files from directory
    
    Args:
        directory: Directory to clean
        older_than_days: Delete files older than this many days
        pattern: File pattern to match
        dry_run: If True, only return files that would be deleted
        
    Returns:
        List of deleted (or would be deleted) files
    """
    import time
    
    cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
    deleted_files = []
    
    for file_path in list_files(directory, pattern, recursive=True):
        if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
            if not dry_run:
                try:
                    file_path.unlink()
                    deleted_files.append(file_path)
                except Exception as e:
                    logging.error(f"Failed to delete {file_path}: {e}")
            else:
                deleted_files.append(file_path)
    
    return deleted_files

def backup_file(file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None, 
               timestamp: bool = True) -> Optional[Path]:
    """Create backup of file
    
    Args:
        file_path: File to backup
        backup_dir: Directory to store backup (default: same directory)
        timestamp: Whether to add timestamp to backup name
        
    Returns:
        Path to backup file or None if failed
    """
    try:
        source_path = Path(file_path)
        
        if backup_dir is None:
            backup_dir = source_path.parent
        else:
            backup_dir = Path(backup_dir)
            backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create backup filename
        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{source_path.stem}_{timestamp_str}{source_path.suffix}"
        else:
            backup_name = f"{source_path.stem}_backup{source_path.suffix}"
        
        backup_path = backup_dir / backup_name
        
        # Copy file
        shutil.copy2(source_path, backup_path)
        
        return backup_path
    except Exception as e:
        logging.error(f"Failed to backup file {file_path}: {e}")
        return None

def load_image(file_path: Union[str, Path], mode: str = 'RGB') -> Optional[Image.Image]:
    """Load image using PIL
    
    Args:
        file_path: Path to image file
        mode: Image mode ('RGB', 'L', etc.)
        
    Returns:
        PIL Image or None if failed
    """
    try:
        image = Image.open(file_path)
        if mode:
            image = image.convert(mode)
        return image
    except Exception as e:
        logging.error(f"Failed to load image {file_path}: {e}")
        return None

def save_image(image: Union[Image.Image, np.ndarray], file_path: Union[str, Path], 
              create_dirs: bool = True, quality: int = 95) -> bool:
    """Save image to file
    
    Args:
        image: PIL Image or numpy array
        file_path: Path to save image
        create_dirs: Whether to create directories
        quality: JPEG quality (1-100)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(image, 'RGB')
            elif len(image.shape) == 2:
                image = Image.fromarray(image, 'L')
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        
        # Save with appropriate parameters
        if path.suffix.lower() in ['.jpg', '.jpeg']:
            image.save(path, quality=quality, optimize=True)
        else:
            image.save(path)
        
        return True
    except Exception as e:
        logging.error(f"Failed to save image to {file_path}: {e}")
        return False

def get_image_info(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Get image information
    
    Args:
        file_path: Path to image file
        
    Returns:
        Dictionary with image info or None if failed
    """
    try:
        with Image.open(file_path) as img:
            info = {
                'filename': Path(file_path).name,
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height,
                'file_size': get_file_size(file_path)
            }
            
            # Add EXIF data if available
            if hasattr(img, '_getexif') and img._getexif() is not None:
                info['exif'] = img._getexif()
            
            return info
    except Exception as e:
        logging.error(f"Failed to get image info for {file_path}: {e}")
        return None