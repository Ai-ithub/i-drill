"""
Unit tests for Backup Service
"""
import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from services.backup_service import BackupService


class TestBackupService:
    """Tests for BackupService"""
    
    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def backup_service(self, temp_backup_dir, monkeypatch):
        """Create BackupService instance with temp directory"""
        monkeypatch.setenv("BACKUP_DIR", str(temp_backup_dir))
        monkeypatch.setenv("ENABLE_AUTO_BACKUP", "false")  # Disable scheduler for tests
        return BackupService()
    
    def test_init(self, temp_backup_dir, monkeypatch):
        """Test BackupService initialization"""
        monkeypatch.setenv("BACKUP_DIR", str(temp_backup_dir))
        monkeypatch.setenv("ENABLE_AUTO_BACKUP", "false")
        
        service = BackupService()
        
        assert service.backup_dir == temp_backup_dir
        assert service.backup_dir.exists()
        assert service.enabled is False
        assert service.retention_days == 30  # Default
    
    def test_init_with_custom_config(self, temp_backup_dir, monkeypatch):
        """Test BackupService with custom configuration"""
        monkeypatch.setenv("BACKUP_DIR", str(temp_backup_dir))
        monkeypatch.setenv("ENABLE_AUTO_BACKUP", "true")
        monkeypatch.setenv("BACKUP_RETENTION_DAYS", "60")
        
        with patch('services.backup_service.BackgroundScheduler') as mock_scheduler:
            service = BackupService()
            assert service.retention_days == 60
            assert service.enabled is True
    
    def test_create_backup_metadata(self, backup_service):
        """Test creating backup metadata"""
        metadata = backup_service._create_backup_metadata("test_backup", "database")
        
        assert metadata["backup_name"] == "test_backup"
        assert metadata["backup_type"] == "database"
        assert "timestamp" in metadata
        assert "created_at" in metadata
        assert isinstance(metadata["timestamp"], str)
    
    def test_backup_database(self, backup_service, monkeypatch):
        """Test backing up database"""
        # Mock database connection
        mock_db_manager = Mock()
        mock_db_manager.get_backup_data = Mock(return_value={"tables": ["users", "sensor_data"]})
        
        with patch('services.backup_service.db_manager', mock_db_manager):
            result = backup_service.backup_database()
            
            assert result["success"] is True
            assert "backup_path" in result
            assert Path(result["backup_path"]).exists()
    
    def test_backup_models(self, backup_service, monkeypatch):
        """Test backing up ML models"""
        # Create mock model directory
        model_dir = backup_service.backup_dir.parent / "models"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "test_model.pkl").write_text("model data")
        
        monkeypatch.setenv("MODEL_PATH", str(model_dir))
        
        result = backup_service.backup_models()
        
        assert result["success"] is True
        assert "backup_path" in result
    
    def test_backup_config(self, backup_service, monkeypatch):
        """Test backing up configuration"""
        # Create mock config file
        config_file = backup_service.backup_dir.parent / "config.env"
        config_file.write_text("TEST_CONFIG=value")
        
        result = backup_service.backup_config()
        
        assert result["success"] is True
        assert "backup_path" in result
    
    def test_backup_logs(self, backup_service):
        """Test backing up logs"""
        # Create mock log directory
        log_dir = backup_service.backup_dir.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        (log_dir / "app.log").write_text("log content")
        
        result = backup_service.backup_logs()
        
        assert result["success"] is True
        assert "backup_path" in result
    
    def test_cleanup_old_backups(self, backup_service):
        """Test cleaning up old backups"""
        # Create old backup file
        old_backup = backup_service.backup_dir / "old_backup_2020-01-01.tar.gz"
        old_backup.touch()
        
        # Create recent backup file
        recent_backup = backup_service.backup_dir / f"recent_backup_{datetime.now().strftime('%Y-%m-%d')}.tar.gz"
        recent_backup.touch()
        
        backup_service.cleanup_old_backups()
        
        # Old backup should be deleted
        assert not old_backup.exists()
        # Recent backup should remain
        assert recent_backup.exists()
    
    def test_list_backups(self, backup_service):
        """Test listing backups"""
        # Create some backup files
        backup1 = backup_service.backup_dir / "backup1_2024-01-01.tar.gz"
        backup2 = backup_service.backup_dir / "backup2_2024-01-02.tar.gz"
        backup1.touch()
        backup2.touch()
        
        backups = backup_service.list_backups()
        
        assert len(backups) >= 2
        assert any("backup1" in b["name"] for b in backups)
        assert any("backup2" in b["name"] for b in backups)
    
    def test_restore_backup(self, backup_service):
        """Test restoring from backup"""
        # Create a test backup file
        backup_file = backup_service.backup_dir / "test_backup.tar.gz"
        
        # Create a simple tar.gz file
        import tarfile
        with tarfile.open(backup_file, "w:gz") as tar:
            # Add a test file
            test_file = backup_service.backup_dir / "test_file.txt"
            test_file.write_text("test content")
            tar.add(test_file, arcname="test_file.txt")
            test_file.unlink()
        
        result = backup_service.restore_backup(str(backup_file))
        
        assert result["success"] is True
    
    def test_get_backup_info(self, backup_service):
        """Test getting backup information"""
        # Create a backup file
        backup_file = backup_service.backup_dir / "test_backup.tar.gz"
        backup_file.touch()
        
        info = backup_service.get_backup_info(str(backup_file))
        
        assert info["exists"] is True
        assert "size" in info
        assert "created_at" in info

