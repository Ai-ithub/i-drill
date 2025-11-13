"""
Unit tests for file_utils
"""
import pytest
import os
import json
import yaml
import pickle
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src" / "predictive_maintenance" / "utils"
sys.path.insert(0, str(src_path))

from file_utils import (
    ensure_dir,
    get_file_size,
    get_file_hash,
    copy_file,
    move_file,
    delete_file,
    delete_directory,
    list_files,
    find_files_by_extension,
    save_json,
    load_json,
    save_yaml,
    load_yaml,
    save_pickle,
    load_pickle,
    format_file_size,
    clean_directory,
    backup_file,
    load_image,
    save_image,
    get_image_info
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_file(temp_dir):
    """Create temporary file for testing"""
    file_path = temp_dir / "test_file.txt"
    file_path.write_text("test content")
    return file_path


class TestFileUtils:
    """Test suite for file_utils"""

    def test_ensure_dir_creates_directory(self, temp_dir):
        """Test ensuring directory exists"""
        # Setup
        new_dir = temp_dir / "new_directory"
        
        # Execute
        result = ensure_dir(new_dir)
        
        # Assert
        assert result.exists()
        assert result.is_dir()

    def test_ensure_dir_existing_directory(self, temp_dir):
        """Test ensuring existing directory"""
        # Setup
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()
        
        # Execute
        result = ensure_dir(existing_dir)
        
        # Assert
        assert result.exists()
        assert result.is_dir()

    def test_ensure_dir_nested_directory(self, temp_dir):
        """Test ensuring nested directory"""
        # Setup
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        
        # Execute
        result = ensure_dir(nested_dir)
        
        # Assert
        assert result.exists()
        assert result.is_dir()

    def test_get_file_size(self, temp_file):
        """Test getting file size"""
        # Execute
        size = get_file_size(temp_file)
        
        # Assert
        assert size > 0
        assert size == len("test content")

    def test_get_file_hash_md5(self, temp_file):
        """Test getting file hash with MD5"""
        # Execute
        hash_value = get_file_hash(temp_file, algorithm='md5')
        
        # Assert
        assert len(hash_value) == 32  # MD5 produces 32 hex characters
        assert isinstance(hash_value, str)

    def test_get_file_hash_sha256(self, temp_file):
        """Test getting file hash with SHA256"""
        # Execute
        hash_value = get_file_hash(temp_file, algorithm='sha256')
        
        # Assert
        assert len(hash_value) == 64  # SHA256 produces 64 hex characters
        assert isinstance(hash_value, str)

    def test_copy_file_success(self, temp_file, temp_dir):
        """Test copying file successfully"""
        # Setup
        dest_path = temp_dir / "copied_file.txt"
        
        # Execute
        result = copy_file(temp_file, dest_path)
        
        # Assert
        assert result is True
        assert dest_path.exists()
        assert dest_path.read_text() == temp_file.read_text()

    def test_copy_file_create_dirs(self, temp_file, temp_dir):
        """Test copying file with directory creation"""
        # Setup
        dest_path = temp_dir / "subdir" / "copied_file.txt"
        
        # Execute
        result = copy_file(temp_file, dest_path, create_dirs=True)
        
        # Assert
        assert result is True
        assert dest_path.exists()
        assert dest_path.parent.exists()

    def test_move_file_success(self, temp_file, temp_dir):
        """Test moving file successfully"""
        # Setup
        dest_path = temp_dir / "moved_file.txt"
        
        # Execute
        result = move_file(temp_file, dest_path)
        
        # Assert
        assert result is True
        assert dest_path.exists()
        assert not temp_file.exists()

    def test_delete_file_success(self, temp_file):
        """Test deleting file successfully"""
        # Execute
        result = delete_file(temp_file)
        
        # Assert
        assert result is True
        assert not temp_file.exists()

    def test_delete_file_nonexistent(self, temp_dir):
        """Test deleting nonexistent file"""
        # Setup
        nonexistent_file = temp_dir / "nonexistent.txt"
        
        # Execute
        result = delete_file(nonexistent_file)
        
        # Assert
        assert result is False

    def test_delete_directory_recursive(self, temp_dir):
        """Test deleting directory recursively"""
        # Setup
        test_dir = temp_dir / "test_dir"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")
        
        # Execute
        result = delete_directory(test_dir, recursive=True)
        
        # Assert
        assert result is True
        assert not test_dir.exists()

    def test_list_files(self, temp_dir):
        """Test listing files in directory"""
        # Setup
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.txt").write_text("content2")
        (temp_dir / "file3.py").write_text("content3")
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "file4.txt").write_text("content4")
        
        # Execute
        files = list_files(temp_dir, pattern="*.txt")
        
        # Assert
        assert len(files) == 2
        assert all(f.suffix == ".txt" for f in files)

    def test_list_files_recursive(self, temp_dir):
        """Test listing files recursively"""
        # Setup
        (temp_dir / "file1.txt").write_text("content1")
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("content2")
        
        # Execute
        files = list_files(temp_dir, pattern="*.txt", recursive=True)
        
        # Assert
        assert len(files) >= 2

    def test_find_files_by_extension(self, temp_dir):
        """Test finding files by extension"""
        # Setup
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.py").write_text("content2")
        (temp_dir / "file3.txt").write_text("content3")
        
        # Execute
        files = find_files_by_extension(temp_dir, extensions=".txt")
        
        # Assert
        assert len(files) == 2
        assert all(f.suffix == ".txt" for f in files)

    def test_find_files_by_extension_multiple(self, temp_dir):
        """Test finding files by multiple extensions"""
        # Setup
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.py").write_text("content2")
        (temp_dir / "file3.json").write_text("content3")
        
        # Execute
        files = find_files_by_extension(temp_dir, extensions=[".txt", ".py"])
        
        # Assert
        assert len(files) == 2

    def test_save_json_success(self, temp_dir):
        """Test saving JSON file successfully"""
        # Setup
        data = {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}
        file_path = temp_dir / "test.json"
        
        # Execute
        result = save_json(data, file_path)
        
        # Assert
        assert result is True
        assert file_path.exists()
        loaded_data = json.loads(file_path.read_text())
        assert loaded_data == data

    def test_load_json_success(self, temp_dir):
        """Test loading JSON file successfully"""
        # Setup
        data = {"key1": "value1", "key2": 123}
        file_path = temp_dir / "test.json"
        file_path.write_text(json.dumps(data))
        
        # Execute
        loaded_data = load_json(file_path)
        
        # Assert
        assert loaded_data == data

    def test_load_json_nonexistent(self, temp_dir):
        """Test loading nonexistent JSON file"""
        # Setup
        file_path = temp_dir / "nonexistent.json"
        
        # Execute
        loaded_data = load_json(file_path)
        
        # Assert
        assert loaded_data is None

    def test_save_yaml_success(self, temp_dir):
        """Test saving YAML file successfully"""
        # Setup
        data = {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}
        file_path = temp_dir / "test.yaml"
        
        # Execute
        result = save_yaml(data, file_path)
        
        # Assert
        assert result is True
        assert file_path.exists()

    def test_load_yaml_success(self, temp_dir):
        """Test loading YAML file successfully"""
        # Setup
        data = {"key1": "value1", "key2": 123}
        file_path = temp_dir / "test.yaml"
        file_path.write_text(yaml.dump(data))
        
        # Execute
        loaded_data = load_yaml(file_path)
        
        # Assert
        assert loaded_data == data

    def test_save_pickle_success(self, temp_dir):
        """Test saving pickle file successfully"""
        # Setup
        data = {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}
        file_path = temp_dir / "test.pkl"
        
        # Execute
        result = save_pickle(data, file_path)
        
        # Assert
        assert result is True
        assert file_path.exists()

    def test_load_pickle_success(self, temp_dir):
        """Test loading pickle file successfully"""
        # Setup
        data = {"key1": "value1", "key2": 123}
        file_path = temp_dir / "test.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Execute
        loaded_data = load_pickle(file_path)
        
        # Assert
        assert loaded_data == data

    def test_format_file_size_bytes(self):
        """Test formatting file size in bytes"""
        # Execute
        result = format_file_size(1023)
        
        # Assert
        assert result == "1023.00 B"

    def test_format_file_size_kb(self):
        """Test formatting file size in KB"""
        # Execute
        result = format_file_size(2048)
        
        # Assert
        assert "KB" in result

    def test_format_file_size_mb(self):
        """Test formatting file size in MB"""
        # Execute
        result = format_file_size(2 * 1024 * 1024)
        
        # Assert
        assert "MB" in result

    def test_format_file_size_zero(self):
        """Test formatting zero file size"""
        # Execute
        result = format_file_size(0)
        
        # Assert
        assert result == "0 B"

    def test_clean_directory(self, temp_dir):
        """Test cleaning directory"""
        # Setup
        import time
        old_file = temp_dir / "old_file.txt"
        old_file.write_text("content")
        # Set file modification time to 8 days ago
        old_time = time.time() - (8 * 24 * 60 * 60)
        os.utime(old_file, (old_time, old_time))
        
        new_file = temp_dir / "new_file.txt"
        new_file.write_text("content")
        
        # Execute
        deleted_files = clean_directory(temp_dir, older_than_days=7, dry_run=False)
        
        # Assert
        assert len(deleted_files) >= 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_clean_directory_dry_run(self, temp_dir):
        """Test cleaning directory in dry run mode"""
        # Setup
        import time
        old_file = temp_dir / "old_file.txt"
        old_file.write_text("content")
        old_time = time.time() - (8 * 24 * 60 * 60)
        os.utime(old_file, (old_time, old_time))
        
        # Execute
        deleted_files = clean_directory(temp_dir, older_than_days=7, dry_run=True)
        
        # Assert
        assert len(deleted_files) >= 1
        assert old_file.exists()  # File should still exist in dry run

    def test_backup_file_with_timestamp(self, temp_file, temp_dir):
        """Test backing up file with timestamp"""
        # Execute
        backup_path = backup_file(temp_file, backup_dir=temp_dir, timestamp=True)
        
        # Assert
        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path != temp_file
        assert backup_path.read_text() == temp_file.read_text()

    def test_backup_file_without_timestamp(self, temp_file, temp_dir):
        """Test backing up file without timestamp"""
        # Execute
        backup_path = backup_file(temp_file, backup_dir=temp_dir, timestamp=False)
        
        # Assert
        assert backup_path is not None
        assert backup_path.exists()
        assert "backup" in backup_path.name

    def test_backup_file_same_directory(self, temp_file):
        """Test backing up file to same directory"""
        # Execute
        backup_path = backup_file(temp_file, backup_dir=None, timestamp=True)
        
        # Assert
        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.parent == temp_file.parent

    @pytest.mark.skipif(not hasattr(__import__('PIL'), 'Image'), reason="PIL not available")
    def test_load_image(self, temp_dir):
        """Test loading image"""
        # Setup
        from PIL import Image as PILImage
        test_image = PILImage.new('RGB', (100, 100), color='red')
        image_path = temp_dir / "test_image.png"
        test_image.save(image_path)
        
        # Execute
        loaded_image = load_image(image_path)
        
        # Assert
        assert loaded_image is not None
        assert loaded_image.size == (100, 100)

    @pytest.mark.skipif(not hasattr(__import__('PIL'), 'Image'), reason="PIL not available")
    def test_save_image(self, temp_dir):
        """Test saving image"""
        # Setup
        from PIL import Image as PILImage
        import numpy as np
        test_image = PILImage.new('RGB', (100, 100), color='blue')
        image_path = temp_dir / "saved_image.png"
        
        # Execute
        result = save_image(test_image, image_path)
        
        # Assert
        assert result is True
        assert image_path.exists()

    @pytest.mark.skipif(not hasattr(__import__('PIL'), 'Image'), reason="PIL not available")
    def test_save_image_numpy_array(self, temp_dir):
        """Test saving image from numpy array"""
        # Setup
        import numpy as np
        image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image_path = temp_dir / "saved_image_np.png"
        
        # Execute
        result = save_image(image_array, image_path)
        
        # Assert
        assert result is True
        assert image_path.exists()

    @pytest.mark.skipif(not hasattr(__import__('PIL'), 'Image'), reason="PIL not available")
    def test_get_image_info(self, temp_dir):
        """Test getting image information"""
        # Setup
        from PIL import Image as PILImage
        test_image = PILImage.new('RGB', (100, 200), color='green')
        image_path = temp_dir / "test_image.png"
        test_image.save(image_path)
        
        # Execute
        info = get_image_info(image_path)
        
        # Assert
        assert info is not None
        assert info['width'] == 100
        assert info['height'] == 200
        assert info['format'] == 'PNG'

