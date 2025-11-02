"""
Data Quality Validation Module for Drilling Rig Sensor Data

این ماژول شامل اعتبارسنجی‌های پیشرفته برای داده‌های سنسور حفاری است:
- Range validation
- Statistical checks (Z-score, IQR)
- Missing data detection
- Outlier detection (multiple methods)
- Data consistency checks
- Temporal consistency checks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """نتیجه اعتبارسنجی یک فیلد داده"""
    field_name: str
    is_valid: bool
    validation_type: str
    message: str
    original_value: Any
    corrected_value: Optional[Any] = None
    severity: str = "warning"  # "info", "warning", "error", "critical"
    score: float = 1.0  # کیفیت داده (0-1)


@dataclass
class DataQualityReport:
    """گزارش کامل اعتبارسنجی کیفیت داده"""
    timestamp: datetime
    rig_id: str
    overall_score: float  # میانگین کیفیت همه فیلدها
    is_valid: bool
    validation_results: List[ValidationResult]
    missing_fields: List[str]
    outlier_fields: List[str]
    corrected_data: Dict[str, Any]


class DataQualityValidator:
    """
    کلاس اصلی برای اعتبارسنجی کیفیت داده‌های سنسور حفاری
    """
    
    # محدوده‌های معتبر برای هر سنسور (مبتنی بر استانداردهای صنعت)
    SENSOR_RANGES = {
        "wob": (0, 50000),  # Weight on Bit (Newtons)
        "rpm": (0, 200),  # Rotations per Minute
        "torque": (0, 10000),  # Torque (Nm)
        "rop": (0, 100),  # Rate of Penetration (m/h)
        "mud_flow_rate": (0, 2000),  # Mud Flow Rate (L/min)
        "mud_pressure": (0, 50000000),  # Mud Pressure (Pa)
        "mud_temperature": (-10, 150),  # Mud Temperature (°C)
        "mud_density": (800, 2500),  # Mud Density (kg/m³)
        "mud_viscosity": (0, 500),  # Mud Viscosity (cP)
        "mud_ph": (6, 12),  # Mud pH
        "gamma_ray": (0, 500),  # Gamma Ray (API units)
        "resistivity": (0, 1000),  # Resistivity (Ohm.m)
        "power_consumption": (0, 1000),  # Power (kW)
        "vibration_level": (0, 10),  # Vibration (g)
        "bit_temperature": (0, 200),  # Bit Temperature (°C)
        "motor_temperature": (0, 150),  # Motor Temperature (°C)
        "depth": (0, 15000),  # Depth (m)
    }
    
    # فیلدهای boolean/status
    STATUS_FIELDS = ["pump_status", "compressor_status", "maintenance_flag"]
    
    def __init__(
        self,
        z_score_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        enable_isolation_forest: bool = True,
        enable_temporal_checks: bool = True,
    ):
        """
        Args:
            z_score_threshold: آستانه Z-score برای outlier detection
            iqr_multiplier: ضریب IQR برای outlier detection
            enable_isolation_forest: استفاده از Isolation Forest برای outlier detection
            enable_temporal_checks: بررسی یکنواختی زمانی داده‌ها
        """
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self.enable_isolation_forest = enable_isolation_forest
        self.enable_temporal_checks = enable_temporal_checks
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.historical_data = []  # برای بررسی temporal consistency
        
    def validate_range(self, field_name: str, value: Any) -> ValidationResult:
        """
        اعتبارسنجی محدوده داده
        """
        if value is None:
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                validation_type="missing_data",
                message=f"Field '{field_name}' is missing",
                original_value=value,
                severity="error"
            )
        
        # فیلدهای status را بررسی نکن
        if field_name in self.STATUS_FIELDS:
            return ValidationResult(
                field_name=field_name,
                is_valid=True,
                validation_type="status_field",
                message=f"Status field '{field_name}' is valid",
                original_value=value,
                severity="info"
            )
        
        # اگر محدوده تعریف شده ندارد، بررسی نکن
        if field_name not in self.SENSOR_RANGES:
            return ValidationResult(
                field_name=field_name,
                is_valid=True,
                validation_type="no_range_defined",
                message=f"No range defined for '{field_name}'",
                original_value=value,
                severity="info"
            )
        
        try:
            value = float(value)
            min_val, max_val = self.SENSOR_RANGES[field_name]
            
            if min_val <= value <= max_val:
                return ValidationResult(
                    field_name=field_name,
                    is_valid=True,
                    validation_type="range_check",
                    message=f"'{field_name}' is within valid range [{min_val}, {max_val}]",
                    original_value=value,
                    severity="info",
                    score=1.0
                )
            else:
                # محاسبه فاصله از محدوده
                distance = min(abs(value - min_val), abs(value - max_val))
                range_width = max_val - min_val
                severity = "error" if distance > 0.2 * range_width else "warning"
                
                return ValidationResult(
                    field_name=field_name,
                    is_valid=False,
                    validation_type="range_check",
                    message=f"'{field_name}' = {value} is out of range [{min_val}, {max_val}]",
                    original_value=value,
                    corrected_value=np.clip(value, min_val, max_val),
                    severity=severity,
                    score=max(0, 1 - distance / range_width)
                )
        except (ValueError, TypeError):
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                validation_type="type_check",
                message=f"'{field_name}' is not a valid number: {value}",
                original_value=value,
                severity="error"
            )
    
    def validate_with_history(
        self,
        field_name: str,
        value: Any,
        historical_values: List[float]
    ) -> List[ValidationResult]:
        """
        اعتبارسنجی با استفاده از داده‌های تاریخی (Z-score, IQR)
        """
        results = []
        
        if not historical_values or len(historical_values) < 5:
            # برای اعتبارسنجی آماری حداقل 5 داده تاریخی نیاز است
            return results
        
        try:
            value = float(value)
            hist_array = np.array(historical_values)
            
            # 1. Z-score validation
            mean = np.mean(hist_array)
            std = np.std(hist_array)
            
            if std > 0:
                z_score = abs((value - mean) / std)
                
                if z_score > self.z_score_threshold:
                    severity = "error" if z_score > 4.0 else "warning"
                    results.append(ValidationResult(
                        field_name=field_name,
                        is_valid=False,
                        validation_type="z_score_outlier",
                        message=f"'{field_name}' = {value} is an outlier (Z-score: {z_score:.2f}, mean: {mean:.2f})",
                        original_value=value,
                        corrected_value=mean,  # پیشنهاد: استفاده از میانگین
                        severity=severity,
                        score=max(0, 1 - (z_score - self.z_score_threshold) / self.z_score_threshold)
                    ))
                else:
                    results.append(ValidationResult(
                        field_name=field_name,
                        is_valid=True,
                        validation_type="z_score_check",
                        message=f"'{field_name}' passed Z-score check (Z-score: {z_score:.2f})",
                        original_value=value,
                        severity="info",
                        score=1.0
                    ))
            
            # 2. IQR (Interquartile Range) validation
            Q1 = np.percentile(hist_array, 25)
            Q3 = np.percentile(hist_array, 75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - self.iqr_multiplier * IQR
                upper_bound = Q3 + self.iqr_multiplier * IQR
                
                if value < lower_bound or value > upper_bound:
                    results.append(ValidationResult(
                        field_name=field_name,
                        is_valid=False,
                        validation_type="iqr_outlier",
                        message=f"'{field_name}' = {value} is outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]",
                        original_value=value,
                        corrected_value=np.clip(value, lower_bound, upper_bound),
                        severity="warning",
                        score=0.7
                    ))
                else:
                    results.append(ValidationResult(
                        field_name=field_name,
                        is_valid=True,
                        validation_type="iqr_check",
                        message=f"'{field_name}' passed IQR check",
                        original_value=value,
                        severity="info",
                        score=1.0
                    ))
        
        except Exception as e:
            logger.error(f"Error in validate_with_history for {field_name}: {e}")
        
        return results
    
    def validate_temporal_consistency(
        self,
        data: Dict[str, Any],
        previous_data: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """
        بررسی یکنواختی زمانی داده‌ها (تغییرات ناگهانی)
        """
        results = []
        
        if not previous_data or not self.enable_temporal_checks:
            return results
        
        # نرخ تغییر مجاز برای هر فیلد (درصد)
        max_change_rates = {
            "wob": 20,  # 20% تغییر در هر خواندن
            "rpm": 15,
            "torque": 25,
            "rop": 30,
            "mud_pressure": 15,
            "mud_temperature": 10,
            "depth": 5,  # عمق نمی‌تواند کاهش یابد
        }
        
        for field_name, max_change in max_change_rates.items():
            if field_name not in data or field_name not in previous_data:
                continue
            
            try:
                current = float(data[field_name])
                previous = float(previous_data[field_name])
                
                if previous == 0:
                    continue
                
                change_rate = abs((current - previous) / previous) * 100
                
                # بررسی خاص برای depth: نمی‌تواند کاهش یابد
                if field_name == "depth" and current < previous:
                    results.append(ValidationResult(
                        field_name=field_name,
                        is_valid=False,
                        validation_type="temporal_consistency",
                        message=f"'{field_name}' decreased from {previous} to {current} (impossible)",
                        original_value=current,
                        corrected_value=previous,  # نگه داشتن مقدار قبلی
                        severity="error",
                        score=0.0
                    ))
                elif change_rate > max_change:
                    results.append(ValidationResult(
                        field_name=field_name,
                        is_valid=False,
                        validation_type="temporal_consistency",
                        message=f"'{field_name}' changed by {change_rate:.1f}% (max allowed: {max_change}%)",
                        original_value=current,
                        severity="warning",
                        score=max(0, 1 - change_rate / (max_change * 2))
                    ))
            
            except (ValueError, TypeError, ZeroDivisionError):
                continue
        
        return results
    
    def validate_data_consistency(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """
        بررسی سازگاری بین فیلدهای مرتبط
        """
        results = []
        
        # 1. WOB و Torque باید همبستگی داشته باشند
        if "wob" in data and "torque" in data:
            try:
                wob = float(data["wob"])
                torque = float(data["torque"])
                
                # رابطه تقریبی: torque ≈ wob * 0.2 (بسته به نوع مته)
                expected_torque = wob * 0.2
                torque_diff = abs(torque - expected_torque)
                
                if torque_diff > expected_torque * 0.5:  # اختلاف بیشتر از 50%
                    results.append(ValidationResult(
                        field_name="torque_wob_consistency",
                        is_valid=False,
                        validation_type="data_consistency",
                        message=f"Torque ({torque}) and WOB ({wob}) are inconsistent (expected torque ~{expected_torque:.1f})",
                        original_value=(wob, torque),
                        severity="warning",
                        score=0.6
                    ))
            except (ValueError, TypeError):
                pass
        
        # 2. RPM و ROP باید منطقی باشند
        if "rpm" in data and "rop" in data:
            try:
                rpm = float(data["rpm"])
                rop = float(data["rop"])
                
                # RPM صفر نباید ROP داشته باشد
                if rpm == 0 and rop > 0.1:
                    results.append(ValidationResult(
                        field_name="rpm_rop_consistency",
                        is_valid=False,
                        validation_type="data_consistency",
                        message=f"RPM is 0 but ROP is {rop} (inconsistent)",
                        original_value=(rpm, rop),
                        severity="error",
                        score=0.0
                    ))
            except (ValueError, TypeError):
                pass
        
        # 3. Mud pressure باید با depth و mud density مرتبط باشد
        if "mud_pressure" in data and "depth" in data and "mud_density" in data:
            try:
                pressure = float(data["mud_pressure"])
                depth = float(data["depth"])
                density = float(data["mud_density"])
                
                # فشار = چگالی * عمق * شتاب گرانش (تقریبی)
                expected_pressure = density * depth * 9.81
                pressure_diff = abs(pressure - expected_pressure)
                
                if pressure_diff > expected_pressure * 0.3:  # اختلاف بیشتر از 30%
                    results.append(ValidationResult(
                        field_name="mud_pressure_consistency",
                        is_valid=False,
                        validation_type="data_consistency",
                        message=f"Mud pressure ({pressure}) inconsistent with depth ({depth}m) and density ({density}kg/m³)",
                        original_value=(pressure, depth, density),
                        severity="warning",
                        score=0.7
                    ))
            except (ValueError, TypeError):
                pass
        
        return results
    
    def validate_completeness(self, data: Dict[str, Any]) -> List[str]:
        """
        بررسی کامل بودن داده (Missing fields)
        """
        required_fields = [
            "timestamp", "rig_id", "wob", "rpm", "torque", "rop",
            "depth", "mud_pressure", "mud_temperature"
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)
        
        return missing_fields
    
    def validate(
        self,
        data: Dict[str, Any],
        historical_data: Optional[pd.DataFrame] = None,
        previous_record: Optional[Dict[str, Any]] = None
    ) -> DataQualityReport:
        """
        اعتبارسنجی کامل یک رکورد داده
        
        Args:
            data: داده سنسور برای اعتبارسنجی
            historical_data: داده‌های تاریخی برای outlier detection
            previous_record: رکورد قبلی برای بررسی temporal consistency
        
        Returns:
            DataQualityReport: گزارش کامل اعتبارسنجی
        """
        validation_results = []
        corrected_data = data.copy()
        
        # 1. بررسی کامل بودن داده
        missing_fields = self.validate_completeness(data)
        
        # 2. اعتبارسنجی محدوده برای همه فیلدهای عددی
        numeric_fields = [
            field for field in data.keys()
            if field not in ["timestamp", "rig_id", "failure_type"] and field in self.SENSOR_RANGES
        ]
        
        for field in numeric_fields:
            result = self.validate_range(field, data.get(field))
            validation_results.append(result)
            
            # اعمال اصلاحات
            if not result.is_valid and result.corrected_value is not None:
                corrected_data[field] = result.corrected_value
        
        # 3. اعتبارسنجی با داده‌های تاریخی (Z-score, IQR)
        if historical_data is not None and len(historical_data) > 0:
            for field in numeric_fields:
                if field in historical_data.columns:
                    hist_values = historical_data[field].dropna().tolist()
                    if len(hist_values) >= 5:
                        hist_results = self.validate_with_history(field, data.get(field), hist_values)
                        validation_results.extend(hist_results)
                        
                        # اعمال اصلاحات
                        for result in hist_results:
                            if not result.is_valid and result.corrected_value is not None:
                                # فقط اگر محدوده اصلی نبوده باشد
                                if result.validation_type in ["z_score_outlier", "iqr_outlier"]:
                                    corrected_data[field] = result.corrected_value
        
        # 4. بررسی یکنواختی زمانی
        temporal_results = self.validate_temporal_consistency(data, previous_record)
        validation_results.extend(temporal_results)
        
        # 5. بررسی سازگاری داده‌ها
        consistency_results = self.validate_data_consistency(data)
        validation_results.extend(consistency_results)
        
        # 6. محاسبه نمره کلی
        valid_results = [r for r in validation_results if hasattr(r, 'score')]
        overall_score = np.mean([r.score for r in valid_results]) if valid_results else 1.0
        
        # 7. جمع‌آوری فیلدهای outlier
        outlier_fields = [
            r.field_name for r in validation_results
            if not r.is_valid and "outlier" in r.validation_type
        ]
        
        # 8. بررسی اعتبار کلی
        critical_errors = [
            r for r in validation_results
            if not r.is_valid and r.severity == "error"
        ]
        is_valid = len(critical_errors) == 0 and len(missing_fields) == 0
        
        return DataQualityReport(
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            rig_id=data.get("rig_id", "UNKNOWN"),
            overall_score=overall_score,
            is_valid=is_valid,
            validation_results=validation_results,
            missing_fields=missing_fields,
            outlier_fields=list(set(outlier_fields)),
            corrected_data=corrected_data
        )

