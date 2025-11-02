"""
Comprehensive tests for DVR Controller
این فایل شامل تست‌های کامل برای ماژول dvr_controller است
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd
from src.processing.dvr_controller import (
    process_data,
    get_validation_report,
    reset_validator_cache,
    get_validator
)
from src.processing.data_quality_validator import DataQualityReport, ValidationResult


class TestProcessData:
    """Test cases for process_data function"""
    
    def setup_method(self):
        """Setup before each test"""
        reset_validator_cache()
    
    @patch('src.processing.dvr_controller.get_last_n_rows')
    @patch('src.processing.dvr_controller.DataReconciler')
    def test_process_data_valid_record(self, mock_reconciler, mock_get_history):
        """
        تست پردازش یک رکورد معتبر
        """
        # Mock historical data
        mock_get_history.return_value = pd.DataFrame({
            'wob': [10000, 12000, 11000],
            'rpm': [100, 110, 105],
            'torque': [2000, 2200, 2100]
        })
        
        # Mock reconciler
        mock_reconciler_instance = Mock()
        mock_reconciler.return_value = mock_reconciler_instance
        mock_reconciler_instance.reconcile.return_value = pd.DataFrame([{
            'timestamp': '2024-01-01T00:00:00',
            'rig_id': 'RIG_001',
            'wob': 15000,
            'rpm': 120,
            'torque': 2500,
            'rop': 10,
            'depth': 1000,
            'mud_pressure': 15000000,
            'mud_temperature': 60
        }])
        
        # Valid record with all required fields
        record = {
            "timestamp": "2024-01-01T00:00:00",
            "rig_id": "RIG_001",
            "wob": 15000,  # Valid range: 0-50000
            "rpm": 120,    # Valid range: 0-200
            "torque": 2500,  # Valid range: 0-10000
            "rop": 10,     # Valid range: 0-100
            "depth": 1000,  # Valid range: 0-15000
            "mud_pressure": 15000000,  # Valid range: 0-50000000
            "mud_temperature": 60  # Valid range: -10-150
        }
        
        result = process_data(record)
        
        assert result is not None
        assert result.get("rig_id") == "RIG_001"
        assert "_validation_metadata" in result
        assert result["_validation_metadata"]["is_valid"] is True
    
    @patch('src.processing.dvr_controller.get_last_n_rows')
    def test_process_data_missing_required_fields(self, mock_get_history):
        """
        تست پردازش رکورد با فیلدهای ضروری گم شده
        """
        mock_get_history.return_value = None
        
        # Missing required fields (timestamp, rig_id, etc.)
        record = {
            "sensor_id": 1,
            "value": 50
        }
        
        result = process_data(record)
        
        # Should return None due to missing required fields
        assert result is None
    
    @patch('src.processing.dvr_controller.get_last_n_rows')
    def test_process_data_out_of_range_value(self, mock_get_history):
        """
        تست پردازش رکورد با مقدار خارج از محدوده
        """
        mock_get_history.return_value = None
        
        # RPM out of range (max is 200)
        record = {
            "timestamp": "2024-01-01T00:00:00",
            "rig_id": "RIG_001",
            "wob": 15000,
            "rpm": 250,  # Out of range (> 200)
            "torque": 2500,
            "rop": 10,
            "depth": 1000,
            "mud_pressure": 15000000,
            "mud_temperature": 60
        }
        
        result = process_data(record, use_corrections=True)
        
        # Should still return result with correction (not None)
        # But RPM should be clipped to max value (200)
        assert result is not None
        assert result["rpm"] == 200  # Clipped value
    
    @patch('src.processing.dvr_controller.get_last_n_rows')
    def test_process_data_critical_error(self, mock_get_history):
        """
        تست پردازش رکورد با خطای بحرانی (مثلاً depth کاهش یابد)
        """
        mock_get_history.return_value = None
        
        # First record (normal)
        record1 = {
            "timestamp": "2024-01-01T00:00:00",
            "rig_id": "RIG_001",
            "wob": 15000,
            "rpm": 120,
            "torque": 2500,
            "rop": 10,
            "depth": 1000,  # Initial depth
            "mud_pressure": 15000000,
            "mud_temperature": 60
        }
        
        result1 = process_data(record1)
        assert result1 is not None
        
        # Second record with decreased depth (critical error)
        record2 = {
            "timestamp": "2024-01-01T00:01:00",
            "rig_id": "RIG_001",
            "wob": 15000,
            "rpm": 120,
            "torque": 2500,
            "rop": 10,
            "depth": 900,  # Depth decreased (impossible!)
            "mud_pressure": 15000000,
            "mud_temperature": 60
        }
        
        result2 = process_data(record2)
        
        # Should return None due to critical temporal consistency error
        assert result2 is None
    
    @patch('src.processing.dvr_controller.get_last_n_rows')
    def test_process_data_with_historical_validation(self, mock_get_history):
        """
        تست اعتبارسنجی با داده‌های تاریخی (Z-score, IQR)
        """
        # Mock historical data with consistent values
        mock_get_history.return_value = pd.DataFrame({
            'wob': [10000, 11000, 10500, 12000, 11500] * 10,  # 50 records
            'rpm': [100, 110, 105, 115, 108] * 10,
            'torque': [2000, 2100, 2050, 2200, 2150] * 10
        })
        
        # Record with value far from historical mean (outlier)
        record = {
            "timestamp": "2024-01-01T00:00:00",
            "rig_id": "RIG_001",
            "wob": 50000,  # Very high compared to historical mean (~11000)
            "rpm": 120,
            "torque": 2500,
            "rop": 10,
            "depth": 1000,
            "mud_pressure": 15000000,
            "mud_temperature": 60
        }
        
        result = process_data(record, use_corrections=True)
        
        # Should return result (not None) but with outlier detected
        assert result is not None
        assert "_validation_metadata" in result
        # Outlier should be detected
        assert result["_validation_metadata"]["outlier_fields_count"] > 0
    
    @patch('src.processing.dvr_controller.get_last_n_rows')
    def test_process_data_without_corrections(self, mock_get_history):
        """
        تست پردازش بدون اعمال اصلاحات خودکار
        """
        mock_get_history.return_value = None
        
        record = {
            "timestamp": "2024-01-01T00:00:00",
            "rig_id": "RIG_001",
            "wob": 15000,
            "rpm": 250,  # Out of range
            "torque": 2500,
            "rop": 10,
            "depth": 1000,
            "mud_pressure": 15000000,
            "mud_temperature": 60
        }
        
        result = process_data(record, use_corrections=False)
        
        assert result is not None
        # Original value should be kept (not corrected)
        assert result["rpm"] == 250
    
    @patch('src.processing.dvr_controller.get_last_n_rows')
    def test_process_data_reconciliation_failure(self, mock_get_history):
        """
        تست رفتار در صورت خطای reconciliation
        """
        mock_get_history.return_value = None
        
        record = {
            "timestamp": "2024-01-01T00:00:00",
            "rig_id": "RIG_001",
            "wob": 15000,
            "rpm": 120,
            "torque": 2500,
            "rop": 10,
            "depth": 1000,
            "mud_pressure": 15000000,
            "mud_temperature": 60
        }
        
        # Mock reconciliation to raise exception
        with patch('src.processing.dvr_controller.DataReconciler') as mock_reconciler:
            mock_reconciler_instance = Mock()
            mock_reconciler.return_value = mock_reconciler_instance
            mock_reconciler_instance.reconcile.side_effect = Exception("Reconciliation error")
            
            result = process_data(record, use_corrections=True)
            
            # Should still return result (fallback to corrected data)
            assert result is not None


class TestGetValidationReport:
    """Test cases for get_validation_report function"""
    
    def setup_method(self):
        """Setup before each test"""
        reset_validator_cache()
    
    @patch('src.processing.dvr_controller.get_last_n_rows')
    def test_get_validation_report_success(self, mock_get_history):
        """
        تست دریافت گزارش اعتبارسنجی موفق
        """
        mock_get_history.return_value = None
        
        record = {
            "timestamp": "2024-01-01T00:00:00",
            "rig_id": "RIG_001",
            "wob": 15000,
            "rpm": 120,
            "torque": 2500,
            "rop": 10,
            "depth": 1000,
            "mud_pressure": 15000000,
            "mud_temperature": 60
        }
        
        report = get_validation_report(record)
        
        assert report is not None
        assert isinstance(report, DataQualityReport)
        assert report.rig_id == "RIG_001"
        assert report.overall_score >= 0
        assert report.overall_score <= 1
    
    @patch('src.processing.dvr_controller.get_last_n_rows')
    def test_get_validation_report_with_errors(self, mock_get_history):
        """
        تست گزارش اعتبارسنجی با خطاها
        """
        mock_get_history.return_value = None
        
        # Record with out-of-range values
        record = {
            "timestamp": "2024-01-01T00:00:00",
            "rig_id": "RIG_001",
            "wob": 60000,  # Out of range
            "rpm": 250,    # Out of range
            "torque": 2500,
            "rop": 10,
            "depth": 1000,
            "mud_pressure": 15000000,
            "mud_temperature": 60
        }
        
        report = get_validation_report(record)
        
        assert report is not None
        assert len(report.validation_results) > 0
        # Should have some invalid results
        invalid_results = [r for r in report.validation_results if not r.is_valid]
        assert len(invalid_results) > 0


class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_reset_validator_cache(self):
        """
        تست ریست کردن کش validator
        """
        reset_validator_cache()
        # Should not raise exception
        assert True
    
    def test_get_validator_singleton(self):
        """
        تست الگوی singleton برای validator
        """
        validator1 = get_validator()
        validator2 = get_validator()
        
        # Should return the same instance
        assert validator1 is validator2
        assert isinstance(validator1, type(validator2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

