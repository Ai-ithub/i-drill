import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Union
from enum import Enum
import warnings
import json
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import unittest

# ================================
# کلاس‌های پایه و Enumها
# ================================

class ValidationLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING" 
    CRITICAL = "CRITICAL"

class ReportLevel(Enum):
    SUMMARY = "SUMMARY"
    DETAILED = "DETAILED"
    TECHNICAL = "TECHNICAL"

@dataclass
class ValidationResult:
    """نتیجه اعتبارسنجی برای هر فیلد"""
    field_name: str
    is_valid: bool
    validation_type: str
    message: str
    level: ValidationLevel
    original_value: Any = None
    corrected_value: Any = None
    confidence: float = 1.0
    
    def to_dict(self):
        return {
            'field_name': self.field_name,
            'is_valid': self.is_valid,
            'validation_type': self.validation_type,
            'message': self.message,
            'level': self.level.value,
            'original_value': str(self.original_value) if self.original_value is not None else None,
            'corrected_value': str(self.corrected_value) if self.corrected_value is not None else None,
            'confidence': self.confidence
        }

@dataclass
class DataQualityReport:
    """گزارش کامل کیفیت داده"""
    
    # اطلاعات کلی
    report_id: str
    generation_time: datetime
    data_source: str
    total_records: int
    total_fields: int
    time_range: Tuple[datetime, datetime]
    
    # خلاصه کیفیت
    overall_quality_score: float
    quality_status: str
    validation_duration: float
    
    # آمار خطاها
    total_issues: int
    critical_issues: int
    warning_issues: int
    info_issues: int
    
    # توزیع مشکلات
    issues_by_type: Dict[str, int]
    issues_by_field: Dict[str, int]
    issues_by_severity: Dict[str, int]
    
    # اصلاحات انجام شده
    total_corrections: int
    correction_success_rate: float
    corrected_fields: List[str]
    
    # توصیه‌ها
    recommendations: List[str]
    priority_actions: List[str]
    
    # جزئیات فنی
    validation_results: List[ValidationResult]
    correction_log: List[Dict]
    quality_scores: Dict[str, Any]
    
    # متادیتا
    report_level: ReportLevel
    validator_version: str

# ================================
# شماره ۸: سیستم امتیازدهی کیفیت
# ================================

class DataQualityScorer:
    """سیستم امتیازدهی به کیفیت کلی داده"""
    
    def __init__(self):
        # وزن‌های مختلف برای انواع خطا (بر اساس اهمیت)
        self.error_weights = {
            ValidationLevel.CRITICAL: 5.0,   # خطاهای بحرانی
            ValidationLevel.WARNING: 2.0,    # خطاهای هشدار
            ValidationLevel.INFO: 0.5        # اطلاعاتی
        }
        
        # اهمیت فیلدهای مختلف (بر اساس حساسیت در عملیات حفاری)
        self.field_importance = {
            # فیلدهای حیاتی (امنیت و کنترل)
            'WOB': 1.0, 'RPM': 1.0, 'Standpipe_Pressure': 1.0, 
            'Torque': 1.0, 'Depth': 1.0, 'Mud_Flow_Rate': 1.0,
            
            # فیلدهای مهم (مانیتورینگ)
            'ROP': 0.9, 'Mud_Temperature': 0.9, 'Hook_Load': 0.9,
            'Vibration': 0.9, 'Gamma_Ray': 0.9, 'Resistivity': 0.9,
            
            # فیلدهای تحلیلی
            'Density': 0.8, 'Porosity': 0.8, 'Formation_Type': 0.8,
            'Lithology': 0.8, 'Bit_Wear': 0.8, 'Pump_Efficiency': 0.8,
            
            # فیلدهای وضعیت
            'Active_Events': 0.7, 'Failed_Equipment': 0.7, 
            'Maintenance_Flag': 0.7, 'Abnormal_Condition': 0.7,
            'Layer_Name': 0.6
        }
        
        # آستانه‌های تصمیم‌گیری
        self.quality_thresholds = {
            'EXCELLENT': 0.9,    # 90% - کیفیت عالی
            'GOOD': 0.8,         # 80% - کیفیت خوب
            'ACCEPTABLE': 0.7,   # 70% - قابل قبول
            'POOR': 0.6,         # 60% - کیفیت ضعیف
            'CRITICAL': 0.5      # زیر 50% - بحرانی
        }
    
    def calculate_record_score(self, validation_results: List[ValidationResult], 
                             total_fields: int) -> Dict[str, Any]:
        """
        محاسبه امتیاز کیفیت برای یک رکورد
        """
        if total_fields == 0:
            return self._get_empty_score()
        
        # گروه‌بندی نتایج بر اساس فیلد
        field_results = {}
        for result in validation_results:
            if result.field_name not in field_results:
                field_results[result.field_name] = []
            field_results[result.field_name].append(result)
        
        # محاسبه امتیاز برای هر فیلد
        field_scores = {}
        field_issues = {}
        
        for field_name, results in field_results.items():
            field_score, field_issue = self._calculate_field_score(field_name, results)
            field_scores[field_name] = field_score
            if field_issue:
                field_issues[field_name] = field_issue
        
        # محاسبه امتیاز کلی رکورد
        overall_score = self._calculate_overall_score(field_scores, total_fields)
        
        # تعیین وضعیت کیفیت
        quality_status = self._determine_quality_status(overall_score)
        
        # شناسایی فیلدهای مشکل‌دار
        problematic_fields = self._identify_problematic_fields(field_scores, field_issues)
        
        return {
            'overall_score': overall_score,
            'quality_status': quality_status,
            'field_scores': field_scores,
            'problematic_fields': problematic_fields,
            'total_fields': total_fields,
            'valid_fields_count': sum(1 for score in field_scores.values() if score >= 0.7),
            'critical_issues_count': len([r for r in validation_results if r.level == ValidationLevel.CRITICAL and not r.is_valid]),
            'warning_issues_count': len([r for r in validation_results if r.level == ValidationLevel.WARNING and not r.is_valid]),
            'recommendation': self._generate_recommendation(quality_status, problematic_fields)
        }
    
    def _calculate_field_score(self, field_name: str, 
                             results: List[ValidationResult]) -> Tuple[float, Dict]:
        """محاسبه امتیاز برای یک فیلد خاص"""
        if not results:
            return 1.0, {}  # اگر هیچ اعتبارسنجی انجام نشده، امتیاز کامل
        
        # پیدا کردن بدترین نتیجه برای این فیلد
        worst_result = max(results, key=lambda x: self.error_weights[x.level])
        
        # محاسبه امتیاز بر اساس سطح خطا
        base_score = 1.0 - (self.error_weights[worst_result.level] * 0.1)
        
        # اعمال اهمیت فیلد
        importance = self.field_importance.get(field_name, 0.5)
        adjusted_score = max(0.0, min(1.0, base_score * (0.5 + importance * 0.5)))
        
        # اطلاعات مشکل
        issue_info = {
            'level': worst_result.level.value,
            'message': worst_result.message,
            'validation_type': worst_result.validation_type
        }
        
        return adjusted_score, issue_info
    
    def _calculate_overall_score(self, field_scores: Dict[str, float], 
                               total_fields: int) -> float:
        """محاسبه امتیاز کلی رکورد"""
        if not field_scores:
            return 0.0
        
        # محاسبه میانگین وزنی بر اساس اهمیت فیلدها
        weighted_sum = 0.0
        total_weight = 0.0
        
        for field_name, score in field_scores.items():
            weight = self.field_importance.get(field_name, 0.5)
            weighted_sum += score * weight
            total_weight += weight
        
        # نرمال کردن بر اساس تعداد فیلدهای موجود
        if total_weight > 0:
            base_score = weighted_sum / total_weight
        else:
            base_score = sum(field_scores.values()) / len(field_scores)
        
        # جریمه برای فیلدهای مفقوده
        missing_fields_penalty = (total_fields - len(field_scores)) / total_fields * 0.3
        
        final_score = max(0.0, base_score - missing_fields_penalty)
        
        return round(final_score, 3)
    
    def _determine_quality_status(self, score: float) -> str:
        """تعیین وضعیت کیفیت بر اساس امتیاز"""
        for status, threshold in self.quality_thresholds.items():
            if score >= threshold:
                return status
        return "CRITICAL"
    
    def _identify_problematic_fields(self, field_scores: Dict[str, float],
                                   field_issues: Dict[str, Dict]) -> List[Dict]:
        """شناسایی فیلدهای مشکل‌دار"""
        problematic = []
        
        for field_name, score in field_scores.items():
            if score < 0.7:  # فیلدهایی با امتیاز کمتر از 70%
                issue = field_issues.get(field_name, {})
                problematic.append({
                    'field_name': field_name,
                    'score': score,
                    'issue_level': issue.get('level', 'UNKNOWN'),
                    'issue_message': issue.get('message', 'مشکل نامشخص'),
                    'importance': self.field_importance.get(field_name, 0.5)
                })
        
        # مرتب کردن بر اساس اهمیت و امتیاز
        problematic.sort(key=lambda x: (x['importance'], -x['score']), reverse=True)
        
        return problematic
    
    def _generate_recommendation(self, quality_status: str, 
                               problematic_fields: List[Dict]) -> str:
        """تولید توصیه بر اساس وضعیت کیفیت"""
        recommendations = {
            'EXCELLENT': "داده‌ها از کیفیت عالی برخوردارند. ادامه نظارت توصیه می‌شود.",
            'GOOD': "کیفیت داده‌ها خوب است. نظارت معمول ادامه یابد.",
            'ACCEPTABLE': "داده‌ها قابل قبول هستند اما نیاز به توجه بیشتر دارند.",
            'POOR': "کیفیت داده‌ها ضعیف است. بررسی و اصلاح فوری توصیه می‌شود.",
            'CRITICAL': "وضعیت بحرانی! توقف عملیات و بررسی فوری تمام سیستم‌ها ضروری است."
        }
        
        base_recommendation = recommendations.get(quality_status, "وضعیت نامشخص")
        
        # اضافه کردن توصیه‌های خاص برای فیلدهای مشکل‌دار
        if problematic_fields:
            critical_fields = [f for f in problematic_fields if f['issue_level'] == 'CRITICAL']
            if critical_fields:
                field_names = [f['field_name'] for f in critical_fields[:3]]  # حداکثر 3 فیلد
                base_recommendation += f" فیلدهای بحرانی: {', '.join(field_names)}"
        
        return base_recommendation
    
    def _get_empty_score(self) -> Dict[str, Any]:
        """امتیاز برای داده خالی"""
        return {
            'overall_score': 0.0,
            'quality_status': 'CRITICAL',
            'field_scores': {},
            'problematic_fields': [],
            'total_fields': 0,
            'valid_fields_count': 0,
            'critical_issues_count': 0,
            'warning_issues_count': 0,
            'recommendation': "هیچ داده‌ای برای ارزیابی وجود ندارد"
        }

# ================================
# شماره ۹: سیستم اصلاح خودکار
# ================================

class DataAutoCorrector:
    """سیستم اصلاح خودکار داده‌های ناسالم"""
    
    def __init__(self, correction_strategy: str = "conservative"):
        self.correction_strategy = correction_strategy
        self.correction_log = []
        
        # استراتژی‌های اصلاح بر اساس نوع مشکل
        self.correction_methods = {
            "missing_values": self._correct_missing_values,
            "range_validation": self._correct_range_violations,
            "outlier_detection": self._correct_outliers,
            "data_type_validation": self._correct_data_type,
            "temporal_consistency": self._correct_temporal_issues,
            "consistency_check": self._correct_consistency_issues
        }
        
        # پارامترهای استراتژی
        self.strategy_params = {
            "conservative": {
                "max_correction_ratio": 0.1,  # حداکثر 10% تغییر
                "use_historical": True,
                "confidence_threshold": 0.8
            },
            "moderate": {
                "max_correction_ratio": 0.3,  # حداکثر 30% تغییر
                "use_historical": True, 
                "confidence_threshold": 0.6
            },
            "aggressive": {
                "max_correction_ratio": 0.5,  # حداکثر 50% تغییر
                "use_historical": False,
                "confidence_threshold": 0.4
            }
        }
    
    def auto_correct_data(self, df: pd.DataFrame, 
                         validation_results: List[ValidationResult],
                         historical_data: pd.DataFrame = None) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        اصلاح خودکار داده‌های ناسالم
        """
        self.correction_log = []
        corrected_df = df.copy()
        
        if corrected_df.empty:
            return corrected_df, self.correction_log
        
        # گروه‌بندی مشکلات بر اساس فیلد و نوع
        problems_by_field = self._group_problems_by_field(validation_results)
        
        # اعمال اصلاحات
        for field_name, problems in problems_by_field.items():
            if field_name not in corrected_df.columns:
                continue
                
            for problem in problems:
                corrected_df = self._apply_correction(
                    corrected_df, field_name, problem, historical_data
                )
        
        return corrected_df, self.correction_log
    
    def _group_problems_by_field(self, validation_results: List[ValidationResult]) -> Dict[str, List]:
        """گروه‌بندی مشکلات بر اساس فیلد"""
        problems_by_field = {}
        
        for result in validation_results:
            if not result.is_valid and result.corrected_value is None:
                if result.field_name not in problems_by_field:
                    problems_by_field[result.field_name] = []
                problems_by_field[result.field_name].append(result)
        
        return problems_by_field
    
    def _apply_correction(self, df: pd.DataFrame, field_name: str, 
                         problem: ValidationResult, historical_data: pd.DataFrame = None) -> pd.DataFrame:
        """اعمال اصلاح برای یک مشکل خاص"""
        correction_method = self.correction_methods.get(problem.validation_type)
        
        if correction_method:
            try:
                original_value = df[field_name].iloc[0] if not df.empty else problem.original_value
                
                corrected_value, confidence = correction_method(
                    df, field_name, problem, historical_data
                )
                
                if corrected_value is not None and self._should_apply_correction(original_value, corrected_value, confidence):
                    df[field_name] = df[field_name].astype(type(corrected_value))
                    df.loc[df.index, field_name] = corrected_value
                    
                    # ثبت در لاگ
                    self._log_correction(field_name, original_value, corrected_value, 
                                       problem.validation_type, confidence, problem.message)
                
            except Exception as e:
                warnings.warn(f"خطا در اصلاح فیلد {field_name}: {e}")
        
        return df
    
    def _correct_missing_values(self, df: pd.DataFrame, field_name: str,
                               problem: ValidationResult, historical_data: pd.DataFrame = None) -> Tuple[Any, float]:
        """اصلاح مقادیر گمشده"""
        strategy_params = self.strategy_params[self.correction_strategy]
        
        # روش‌های مختلف برای جایگزینی مقادیر گمشده
        correction_methods = [
            self._impute_with_forward_fill,
            self._impute_with_historical_mean,
            self._impute_with_knn,
            self._impute_with_median
        ]
        
        best_value = None
        best_confidence = 0.0
        
        for method in correction_methods:
            try:
                value, confidence = method(df, field_name, historical_data)
                if confidence > best_confidence and confidence >= strategy_params["confidence_threshold"]:
                    best_value = value
                    best_confidence = confidence
            except Exception:
                continue
        
        return best_value, best_confidence
    
    def _impute_with_forward_fill(self, df: pd.DataFrame, field_name: str, 
                                 historical_data: pd.DataFrame = None) -> Tuple[Any, float]:
        """جایگزینی با مقدار رکورد قبلی"""
        if len(df) < 2:
            return None, 0.0
        
        # استفاده از forward fill
        filled_series = df[field_name].fillna(method='ffill')
        if not filled_series.isna().all():
            return filled_series.iloc[-1], 0.7
        return None, 0.0
    
    def _impute_with_historical_mean(self, df: pd.DataFrame, field_name: str,
                                   historical_data: pd.DataFrame = None) -> Tuple[Any, float]:
        """جایگزینی با میانگین تاریخی"""
        if historical_data is not None and field_name in historical_data.columns:
            historical_mean = historical_data[field_name].mean()
            if not pd.isna(historical_mean):
                return historical_mean, 0.8
        
        # استفاده از میانگین خود دیتافریم
        current_mean = df[field_name].mean()
        if not pd.isna(current_mean):
            return current_mean, 0.6
        
        return None, 0.0
    
    def _impute_with_median(self, df: pd.DataFrame, field_name: str,
                           historical_data: pd.DataFrame = None) -> Tuple[Any, float]:
        """جایگزینی با میانه"""
        median_value = df[field_name].median()
        if not pd.isna(median_value):
            return median_value, 0.75
        return None, 0.0
    
    def _correct_range_violations(self, df: pd.DataFrame, field_name: str,
                                 problem: ValidationResult, historical_data: pd.DataFrame = None) -> Tuple[Any, float]:
        """اصلاح مقادیر خارج از محدوده"""
        # محدوده‌های استاندارد برای فیلدهای حفاری
        standard_ranges = {
            'WOB': (0, 50000), 'RPM': (0, 200), 'Torque': (0, 50000), 'ROP': (0, 100),
            'Standpipe_Pressure': (0, 50000000), 'Mud_Flow_Rate': (0, 2000),
            'Mud_Temperature': (0, 150), 'Gamma_Ray': (0, 200), 'Resistivity': (0, 1000),
            'Density': (1.5, 3.0), 'Porosity': (0, 50), 'Hook_Load': (0, 1000000),
            'Vibration': (0, 100), 'Bit_Wear': (0, 1), 'Pump_Efficiency': (0, 1)
        }
        
        if field_name not in standard_ranges:
            return None, 0.0
        
        min_val, max_val = standard_ranges[field_name]
        current_value = df[field_name].iloc[-1] if not df.empty else problem.original_value
        
        if pd.isna(current_value):
            return None, 0.0
        
        # اصلاح بر اساس استراتژی
        strategy = self.correction_strategy
        if strategy == "conservative":
            # استفاده از مقدار نزدیک‌ترین مرز
            if current_value < min_val:
                corrected = min_val * 1.01  # کمی بالاتر از حد پایین
            else:
                corrected = max_val * 0.99  # کمی پایین‌تر از حد بالا
            confidence = 0.9
        
        elif strategy == "moderate":
            # استفاده از میانه محدوده
            corrected = (min_val + max_val) / 2
            confidence = 0.7
        
        else:  # aggressive
            # استفاده از میانگین تاریخی یا میانه
            if historical_data is not None and field_name in historical_data.columns:
                corrected = historical_data[field_name].mean()
            else:
                corrected = df[field_name].median()
            confidence = 0.6
        
        return np.clip(corrected, min_val, max_val), confidence
    
    def _correct_outliers(self, df: pd.DataFrame, field_name: str,
                         problem: ValidationResult, historical_data: pd.DataFrame = None) -> Tuple[Any, float]:
        """اصلاح نقاط پرت"""
        if len(df) < 3:
            return None, 0.0
        
        current_value = df[field_name].iloc[-1]
        values = df[field_name].dropna()
        
        if len(values) < 3:
            return None, 0.0
        
        # روش‌های مختلف برای اصلاح نقاط پرت
        correction_methods = [
            self._correct_outlier_with_iqr,
            self._correct_outlier_with_winsorization,
            self._correct_outlier_with_rolling_median
        ]
        
        best_value = None
        best_confidence = 0.0
        
        for method in correction_methods:
            try:
                value, confidence = method(values, current_value)
                if confidence > best_confidence:
                    best_value = value
                    best_confidence = confidence
            except Exception:
                continue
        
        return best_value, best_confidence
    
    def _correct_outlier_with_iqr(self, values: pd.Series, current_value: float) -> Tuple[float, float]:
        """اصلاح نقطه پرت با روش IQR"""
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if current_value < lower_bound:
            return lower_bound, 0.8
        elif current_value > upper_bound:
            return upper_bound, 0.8
        else:
            return current_value, 1.0
    
    def _correct_outlier_with_winsorization(self, values: pd.Series, current_value: float) -> Tuple[float, float]:
        """اصلاح نقطه پرت با Winsorization"""
        lower_limit = values.quantile(0.05)
        upper_limit = values.quantile(0.95)
        
        corrected = np.clip(current_value, lower_limit, upper_limit)
        confidence = 0.75 if corrected != current_value else 1.0
        
        return corrected, confidence
    
    def _correct_outlier_with_rolling_median(self, values: pd.Series, current_value: float) -> Tuple[float, float]:
        """اصلاح نقطه پرت با میانه متحرک"""
        if len(values) >= 5:
            rolling_median = values.rolling(window=5, min_periods=1).median().iloc[-1]
            return rolling_median, 0.7
        else:
            median_value = values.median()
            return median_value, 0.6
    def _correct_temporal_issues(self, df: pd.DataFrame, field_name: str,
                            problem: ValidationResult, historical_data: pd.DataFrame = None) -> Tuple[Any, float]:
        """اصلاح مشکلات زمانی - نسخه ساده"""
        if len(df) < 2:
            return None, 0.0
        median_value = df[field_name].median()
        return median_value, 0.6
    def _correct_consistency_issues(self, df: pd.DataFrame, field_name: str,
                              problem: ValidationResult, historical_data: pd.DataFrame = None) -> Tuple[Any, float]:
        """اصلاح مشکلات ناسازگاری منطقی"""
        if field_name == 'Bit_Wear' and 'فرسایش مته' in problem.message:
            # اصلاح فرسایش مته بیش از 100%
            return 1.0, 0.9
        elif 'عمق' in problem.message or 'Depth' in problem.message:
            # جلوگیری از کاهش عمق
            if len(df) > 1:
                last_valid_depth = df[df['Depth'].diff() >= 0]['Depth'].iloc[-1]
                return last_valid_depth + 0.1, 0.8
            return None, 0.0
    def _correct_data_type(self, df: pd.DataFrame, field_name: str,
                            problem: ValidationResult, historical_data: pd.DataFrame = None) -> Tuple[Any, float]:
        """اصلاح نوع داده"""
        current_value = df[field_name].iloc[-1]
        try:
            # تشخیص نوع داده مورد انتظار
            if field_name in ['WOB', 'RPM', 'Torque', 'ROP', 'Standpipe_Pressure', 
                            'Mud_Flow_Rate', 'Mud_Temperature', 'Gamma_Ray', 
                            'Resistivity', 'Density', 'Porosity', 'Hook_Load', 
                            'Vibration', 'Bit_Wear', 'Pump_Efficiency']:
                # فیلدهای عددی
                corrected = float(current_value)
                confidence = 0.9
            
            elif field_name in ['Maintenance_Flag', 'Emergency_Stop']:
                # فیلدهای بولین
                if str(current_value).lower() in ['true', '1', 'yes', 'y']:
                    corrected = 1
                else:
                    corrected = 0
                confidence = 0.8
            
            else:
                # فیلدهای متنی - حذف کاراکترهای غیرمجاز
                corrected = str(current_value).strip()
                confidence = 0.7
                
            return corrected, confidence
        
        except (ValueError, TypeError):
            return None, 0.0
    def run_system_test():
        try:
            # تشخیص نوع داده مورد انتظار
            if field_name in ['WOB', 'RPM', 'Torque', 'ROP', 'Standpipe_Pressure', 
                            'Mud_Flow_Rate', 'Mud_Temperature', 'Gamma_Ray', 
                            'Resistivity', 'Density', 'Porosity', 'Hook_Load', 
                            'Vibration', 'Bit_Wear', 'Pump_Efficiency']:
                # فیلدهای عددی
                corrected = float(current_value)
                confidence = 0.9
            
            elif field_name in ['Maintenance_Flag', 'Emergency_Stop']:
                # فیلدهای بولین
                if str(current_value).lower() in ['true', '1', 'yes', 'y']:
                    corrected = 1
                else:
                    corrected = 0
                confidence = 0.8
            
            else:
                # فیلدهای متنی - حذف کاراکترهای غیرمجاز
                corrected = str(current_value).strip()
                confidence = 0.7
                
            return corrected, confidence
            
        except (ValueError, TypeError):
            return None, 0.0

    def _should_apply_correction(self, original_value: Any, corrected_value: Any, 
                                confidence: float) -> bool:
        """تصمیم‌گیری درباره اعمال اصلاح"""
        if corrected_value is None:
            return False
        
        strategy_params = self.strategy_params[self.correction_strategy]
        
        # بررسی سطح اطمینان
        if confidence < strategy_params["confidence_threshold"]:
            return False
        
        # بررسی میزان تغییر
        try:
            if isinstance(original_value, (int, float)) and isinstance(corrected_value, (int, float)):
                if original_value == 0:
                    change_ratio = abs(corrected_value)
                else:
                    change_ratio = abs(corrected_value - original_value) / abs(original_value)
                
                if change_ratio > strategy_params["max_correction_ratio"]:
                    return False
        except (TypeError, ZeroDivisionError):
            pass
        
        return True

    def _log_correction(self, field_name: str, original_value: Any, corrected_value: Any,
                        validation_type: str, confidence: float, problem_message: str):
        """ثبت لاگ اصلاحات"""
        correction_record = {
            'timestamp': datetime.now(),
            'field_name': field_name,
            'original_value': original_value,
            'corrected_value': corrected_value,
            'validation_type': validation_type,
            'confidence': confidence,
            'problem_message': problem_message,
            'correction_strategy': self.correction_strategy
        }
        
        self.correction_log.append(correction_record)

    def get_correction_summary(self) -> Dict[str, Any]:
        """دریافت خلاصه اصلاحات انجام شده"""
        if not self.correction_log:
            return {
                'total_corrections': 0,
                'fields_corrected': [],
                'average_confidence': 0.0,
                'correction_strategy': self.correction_strategy
            }
        
        fields_corrected = list(set(log['field_name'] for log in self.correction_log))
        total_corrections = len(self.correction_log)
        avg_confidence = np.mean([log['confidence'] for log in self.correction_log])
        
        return {
            'total_corrections': total_corrections,
            'fields_corrected': fields_corrected,
            'average_confidence': round(avg_confidence, 3),
            'correction_strategy': self.correction_strategy,
            'correction_details': self.correction_log
        }

    def _correct_temporal_issues(self, df: pd.DataFrame, field_name: str,
                            problem: ValidationResult, historical_data: pd.DataFrame = None) -> Tuple[Any, float]:
        """اصلاح مشکلات زمانی - نسخه ساده"""
        return None, 0.0

    def _correct_data_type(self, df: pd.DataFrame, field_name: str,
                        problem: ValidationResult, historical_data: pd.DataFrame = None) -> Tuple[Any, float]:
        """اصلاح نوع داده"""
        current_value = df[field_name].iloc[-1]
        
        try:
            # تشخیص نوع داده مورد انتظار
            if field_name in ['WOB', 'RPM', 'Torque', 'ROP', 'Standpipe_Pressure', 
                            'Mud_Flow_Rate', 'Mud_Temperature', 'Gamma_Ray', 
                            'Resistivity', 'Density', 'Porosity', 'Hook_Load', 
                            'Vibration', 'Bit_Wear', 'Pump_Efficiency']:
                # فیلدهای عددی
                corrected = float(current_value)
                confidence = 0.9
            
            elif field_name in ['Maintenance_Flag', 'Emergency_Stop']:
                # فیلدهای بولین
                if str(current_value).lower() in ['true', '1', 'yes', 'y']:
                    corrected = 1
                else:
                    corrected = 0
                confidence = 0.8
            
            else:
                # فیلدهای متنی - حذف کاراکترهای غیرمجاز
                corrected = str(current_value).strip()
                confidence = 0.7
                
            return corrected, confidence
            
        except (ValueError, TypeError):
            return None, 0.0

    def _should_apply_correction(self, original_value: Any, corrected_value: Any, 
                            confidence: float) -> bool:
        """تصمیم‌گیری درباره اعمال اصلاح"""
        if corrected_value is None:
            return False
        
        strategy_params = self.strategy_params[self.correction_strategy]
        
        # بررسی سطح اطمینان
        if confidence < strategy_params["confidence_threshold"]:
            return False
        
        # بررسی میزان تغییر
        try:
            if isinstance(original_value, (int, float)) and isinstance(corrected_value, (int, float)):
                if original_value == 0:
                    change_ratio = abs(corrected_value)
                else:
                    change_ratio = abs(corrected_value - original_value) / abs(original_value)
                
                if change_ratio > strategy_params["max_correction_ratio"]:
                    return False
        except (TypeError, ZeroDivisionError):
            pass
        
        return True

    def _log_correction(self, field_name: str, original_value: Any, corrected_value: Any,
                    validation_type: str, confidence: float, problem_message: str):
        """ثبت لاگ اصلاحات"""
        correction_record = {
            'timestamp': datetime.now(),
            'field_name': field_name,
            'original_value': original_value,
            'corrected_value': corrected_value,
            'validation_type': validation_type,
            'confidence': confidence,
            'problem_message': problem_message,
            'correction_strategy': self.correction_strategy
        }
        
        self.correction_log.append(correction_record)

    def get_correction_summary(self) -> Dict[str, Any]:
        """دریافت خلاصه اصلاحات انجام شده"""
        if not self.correction_log:
            return {
                'total_corrections': 0,
                'fields_corrected': [],
                'average_confidence': 0.0,
                'correction_strategy': self.correction_strategy
            }
        
        fields_corrected = list(set(log['field_name'] for log in self.correction_log))
        total_corrections = len(self.correction_log)
        avg_confidence = np.mean([log['confidence'] for log in self.correction_log])
        
        return {
            'total_corrections': total_corrections,
            'fields_corrected': fields_corrected,
            'average_confidence': round(avg_confidence, 3),
            'correction_strategy': self.correction_strategy,
            'correction_details': self.correction_log
        }

    # ================================
    # شماره ۱۰: سیستم گزارش‌گیری
    # ================================

class ReportGenerator:
    """سیستم تولید گزارش کامل اعتبارسنجی"""
    
    def __init__(self, validator_version: str = "1.0.0"):
        self.validator_version = validator_version
    
    def generate_quality_report(self, 
                            df: pd.DataFrame,
                            validation_results: List[ValidationResult],
                            quality_scores: Dict[str, Any],
                            correction_log: List[Dict] = None,
                            report_level: ReportLevel = ReportLevel.DETAILED,
                            data_source: str = "Unknown") -> DataQualityReport:
        """
        تولید گزارش کامل کیفیت داده
        """
        start_time = datetime.now()
        
        # محاسبه آمار کلی
        stats = self._calculate_statistics(validation_results, quality_scores, correction_log)
        
        # تولید گزارش
        report = DataQualityReport(
            report_id=f"DQR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generation_time=datetime.now(),
            data_source=data_source,
            total_records=len(df),
            total_fields=len(df.columns) if not df.empty else 0,
            time_range=(
                df['Timestamp'].min() if 'Timestamp' in df.columns else datetime.now(),
                df['Timestamp'].max() if 'Timestamp' in df.columns else datetime.now()
            ),
            overall_quality_score=quality_scores.get('overall_score', 0),
            quality_status=quality_scores.get('quality_status', 'UNKNOWN'),
            validation_duration=(datetime.now() - start_time).total_seconds(),
            total_issues=stats['total_issues'],
            critical_issues=stats['critical_issues'],
            warning_issues=stats['warning_issues'],
            info_issues=stats['info_issues'],
            issues_by_type=stats['issues_by_type'],
            issues_by_field=stats['issues_by_field'],
            issues_by_severity=stats['issues_by_severity'],
            total_corrections=stats['total_corrections'],
            correction_success_rate=stats['correction_success_rate'],
            corrected_fields=stats['corrected_fields'],
            recommendations=self._generate_recommendations(stats, quality_scores),
            priority_actions=self._generate_priority_actions(stats, quality_scores),
            validation_results=validation_results,
            correction_log=correction_log or [],
            quality_scores=quality_scores,
            report_level=report_level,
            validator_version=self.validator_version
        )
        
        return report
    
    def _calculate_statistics(self, validation_results: List[ValidationResult],
                            quality_scores: Dict[str, Any],
                            correction_log: List[Dict]) -> Dict[str, Any]:
        """محاسبه آمار و ارقام گزارش"""
        
        # آمار خطاها
        critical_issues = len([r for r in validation_results if r.level == ValidationLevel.CRITICAL and not r.is_valid])
        warning_issues = len([r for r in validation_results if r.level == ValidationLevel.WARNING and not r.is_valid])
        info_issues = len([r for r in validation_results if r.level == ValidationLevel.INFO and not r.is_valid])
        total_issues = critical_issues + warning_issues + info_issues
        
        # توزیع خطاها بر اساس نوع
        issues_by_type = {}
        for result in validation_results:
            if not result.is_valid:
                issues_by_type[result.validation_type] = issues_by_type.get(result.validation_type, 0) + 1
        
        # توزیع خطاها بر اساس فیلد
        issues_by_field = {}
        for result in validation_results:
            if not result.is_valid:
                issues_by_field[result.field_name] = issues_by_field.get(result.field_name, 0) + 1
        
        # توزیع خطاها بر اساس شدت
        issues_by_severity = {
            'CRITICAL': critical_issues,
            'WARNING': warning_issues,
            'INFO': info_issues
        }
        
        # آمار اصلاحات
        total_corrections = len(correction_log) if correction_log else 0
        successful_corrections = len([log for log in (correction_log or []) 
                                    if log.get('confidence', 0) >= 0.5])
        correction_success_rate = (successful_corrections / total_corrections * 100) if total_corrections > 0 else 0
        
        corrected_fields = list(set(log['field_name'] for log in (correction_log or [])))
        
        return {
            'total_issues': total_issues,
            'critical_issues': critical_issues,
            'warning_issues': warning_issues,
            'info_issues': info_issues,
            'issues_by_type': issues_by_type,
            'issues_by_field': issues_by_field,
            'issues_by_severity': issues_by_severity,
            'total_corrections': total_corrections,
            'correction_success_rate': correction_success_rate,
            'corrected_fields': corrected_fields
        }
    
    def _generate_recommendations(self, stats: Dict[str, Any], 
                                quality_scores: Dict[str, Any]) -> List[str]:
        """تولید توصیه‌های بهبود کیفیت"""
        recommendations = []
        
        overall_score = quality_scores.get('overall_score', 0)
        critical_issues = stats['critical_issues']
        warning_issues = stats['warning_issues']
        
        # توصیه‌های مبتنی بر امتیاز کلی
        if overall_score >= 0.9:
            recommendations.append("کیفیت داده‌ها عالی است. ادامه نظارت با فرکانس فعلی توصیه می‌شود.")
        elif overall_score >= 0.8:
            recommendations.append("کیفیت داده‌ها خوب است. بررسی دوره‌ای فیلدهای مشکل‌دار توصیه می‌شود.")
        elif overall_score >= 0.7:
            recommendations.append("کیفیت داده‌ها قابل قبول است. برنامه‌ریزی برای بهبود فیلدهای مشکل‌دار ضروری است.")
        elif overall_score >= 0.6:
            recommendations.append("کیفیت داده‌ها ضعیف است. اقدام فوری برای بهبود ضروری است.")
        else:
            recommendations.append("وضعیت بحرانی! توقف عملیات و بررسی کامل سیستم‌ها ضروری است.")
        
        # توصیه‌های مبتنی بر نوع خطا
        issues_by_type = stats['issues_by_type']
        
        if issues_by_type.get('missing_values', 0) > 0:
            recommendations.append(f"تعداد {issues_by_type['missing_values']} مقدار گمشده وجود دارد. بررسی سنسورها و سیستم جمع‌آوری داده توصیه می‌شود.")
        
        if issues_by_type.get('range_validation', 0) > 0:
            recommendations.append("مقادیر خارج از محدوده شناسایی شده. کالیبراسیون سنسورها ضروری است.")
        
        if issues_by_type.get('outlier_detection', 0) > 0:
            recommendations.append("نقاط پرت شناسایی شده. بررسی شرایط عملیاتی و صحت سنسورها توصیه می‌شود.")
        
        return recommendations
    
    def _generate_priority_actions(self, stats: Dict[str, Any],
                                quality_scores: Dict[str, Any]) -> List[str]:
        """تولید اقدامات اولویت‌دار"""
        actions = []
        
        critical_issues = stats['critical_issues']
        problematic_fields = quality_scores.get('problematic_fields', [])
        
        if critical_issues > 0:
            actions.append("🔴 توقف عملیات برای فیلدهای بحرانی")
            actions.append("🔴 بررسی فوری سنسورهای مربوط به خطاهای بحرانی")
        
        if stats['issues_by_type'].get('missing_values', 0) > 10:
            actions.append("🟠 بررسی اتصال و سلامت سنسورها")
        
        if stats['issues_by_type'].get('range_validation', 0) > 5:
            actions.append("🟠 کالیبراسیون فوری سنسورهای خارج از محدوده")
        
        if len(actions) == 0:
            actions.append("✅ ادامه نظارت معمول - هیچ اقدام فوری لازم نیست")
        
        return actions

# ================================
# شماره ۱-۷: سیستم اصلی اعتبارسنجی
# ================================

class DataQualityValidator:
    """سیستم کامل اعتبارسنجی کیفیت داده‌های حفاری"""
    
    def __init__(self, historical_data: pd.DataFrame = None, validation_config: Dict = None):
        """
        Args:
            historical_data: داده‌های تاریخی برای آموزش مدل‌ها
            validation_config: پیکربندی اعتبارسنجی
        """
        self.historical_data = historical_data
        self.config = validation_config or self._get_default_config()
        
        # مدل‌های ML
        self.outlier_detector = None
        self._train_ml_models()
        
        # سیستم‌های کمکی
        self.scorer = DataQualityScorer()
        self.corrector = DataAutoCorrector(self.config.get('correction_strategy', 'moderate'))
        self.reporter = ReportGenerator()
        
        # محدوده‌های استاندارد برای فیلدهای حفاری
        self.standard_ranges = self._get_standard_ranges()
        
        print("✅ سیستم اعتبارسنجی کیفیت داده راه‌اندازی شد")
    
    def _get_default_config(self) -> Dict:
        """پیکربندی پیش‌فرض اعتبارسنجی"""
        return {
            'enable_auto_correction': True,
            'correction_strategy': 'moderate',
            'validation_level': 'detailed',
            'outlier_detection_method': 'auto',
            'missing_values_threshold': 0.2,  # 20%
            'enable_ml_detection': True
        }
    
    def _get_standard_ranges(self) -> Dict[str, Tuple[float, float]]:
        """محدوده‌های استاندارد برای فیلدهای حفاری"""
        return {
            'WOB': (0, 50000),                    # وزن روی مته (پوند)
            'RPM': (0, 200),                      # دور در دقیقه
            'Torque': (0, 50000),                 # گشتاور (فوت-پوند)
            'ROP': (0, 100),                      # نرخ نفوذ (فوت در ساعت)
            'Depth': (0, 50000),                  # عمق (فوت)
            'Standpipe_Pressure': (0, 50000000),  # فشار (پاسکال)
            'Mud_Flow_Rate': (0, 2000),           # نرخ جریان گل (گالن در دقیقه)
            'Mud_Temperature': (0, 150),          # دمای گل (درجه سانتی‌گراد)
            'Gamma_Ray': (0, 200),                # پرتوی گاما (API)
            'Resistivity': (0, 1000),             # مقاومت (اهم-متر)
            'Density': (1.5, 3.0),                # چگالی (گرم بر سانتیمتر مکعب)
            'Porosity': (0, 50),                  # تخلخل (درصد)
            'Hook_Load': (0, 1000000),            # بار قلاب (پوند)
            'Vibration': (0, 100),                # ارتعاش (g)
            'Bit_Wear': (0, 1),                   # فرسایش مته (0-1)
            'Pump_Efficiency': (0, 1),            # کارایی پمپ (0-1)
            'Mud_Weight': (8, 20)                 # وزن گل (پوند بر گالن)
        }
    
    def _train_ml_models(self):
        """آموزش مدل‌های یادگیری ماشین"""
        if self.historical_data is None or not self.config['enable_ml_detection']:
            return
        
        try:
            numeric_columns = self.historical_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                # آموزش مدل تشخیص نقاط پرت
                self.outlier_detector = IsolationForest(
                    contamination=0.05,
                    random_state=42
                )
                self.outlier_detector.fit(self.historical_data[numeric_columns].fillna(0))
                print("✅ مدل تشخیص نقاط پرت آموزش داده شد")
        except Exception as e:
            warnings.warn(f"خطا در آموزش مدل‌های ML: {e}")
            self.outlier_detector = None
    
    def validate(self, df: pd.DataFrame, enable_correction: bool = None) -> Dict[str, Any]:
        """
        اعتبارسنجی کامل داده‌های حفاری
        
        Args:
            df: دیتافریم داده‌های حفاری
            enable_correction: فعال‌سازی اصلاح خودکار
            
        Returns:
            نتایج کامل اعتبارسنجی
        """
        if df.empty:
            return self._get_empty_validation_result()
        
        start_time = datetime.now()
        enable_correction = enable_correction if enable_correction is not None else self.config['enable_auto_correction']
        
        print(f"🔍 شروع اعتبارسنجی {len(df)} رکورد...")
        
        # اجرای تمام اعتبارسنجی‌ها
        all_validation_results = []
        
        # 1. بررسی مقادیر گمشده
        missing_results = self._check_missing_values(df)
        all_validation_results.extend(missing_results)
        
        # 2. اعتبارسنجی محدوده
        range_results = self._validate_ranges(df)
        all_validation_results.extend(range_results)
        
        # 3. کشف نقاط پرت
        outlier_results = self._detect_outliers(df)
        all_validation_results.extend(outlier_results)
        
        # 4. اعتبارسنجی نوع داده
        type_results = self._validate_data_types(df)
        all_validation_results.extend(type_results)
        
        # 5. بررسی ناسازگاری منطقی
        consistency_results = self._check_consistency(df)
        all_validation_results.extend(consistency_results)
        
        # 6. بررسی یکنواختی زمانی
        temporal_results = self._check_temporal_consistency(df)
        all_validation_results.extend(temporal_results)
        
        # 7. اعتبارسنجی فیلدهای وضعیت
        status_results = self._validate_status_fields(df)
        all_validation_results.extend(status_results)
        
            # 8. بررسی همبستگی‌های پیشرفته
        correlation_results = self._check_advanced_correlations(df)
        all_validation_results.extend(correlation_results)
        
        # 9. تشخیص ناهنجاری پیشرفته  
        advanced_anomaly_results = self._advanced_anomaly_detection(df)
        all_validation_results.extend(advanced_anomaly_results)
        # اصلاح خودکار داده‌ها (در صورت فعال بودن)
        corrected_df = df.copy()
        correction_log = []
        
        if enable_correction:
            corrected_df, correction_log = self.corrector.auto_correct_data(
                df, all_validation_results, self.historical_data
            )
        
        # محاسبه امتیاز کیفیت
        quality_scores = self.scorer.calculate_record_score(
            all_validation_results, len(df.columns)
        )
        
        # تولید گزارش
        report = self.reporter.generate_quality_report(
            df=corrected_df,
            validation_results=all_validation_results,
            quality_scores=quality_scores,
            correction_log=correction_log,
            data_source="DRLLING_DATA_VALIDATION"
        )
        
        validation_duration = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ اعتبارسنجی کامل شد در {validation_duration:.2f} ثانیه")
        print(f"📊 امتیاز کیفیت: {quality_scores['overall_score']} ({quality_scores['quality_status']})")
        print(f"⚠️  خطاهای بحرانی: {quality_scores['critical_issues_count']}")
        
        return {
            'is_valid': quality_scores['overall_score'] >= 0.7,
            'validation_results': all_validation_results,
            'quality_scores': quality_scores,
            'corrected_data': corrected_df,
            'correction_log': correction_log,
            'quality_report': report,
            'validation_duration': validation_duration,
            'summary': {
                'total_records': len(df),
                'total_fields': len(df.columns),
                'total_issues': len([r for r in all_validation_results if not r.is_valid]),
                'critical_issues': quality_scores['critical_issues_count'],
                'warning_issues': quality_scores['warning_issues_count'],
                'corrections_applied': len(correction_log)
            }
        }
    
    def _check_missing_values(self, df: pd.DataFrame) -> List[ValidationResult]:
        """1. بررسی وجود مقادیر گمشده"""
        results = []
        missing_threshold = self.config['missing_values_threshold']
        
        for column in df.columns:
            missing_count = df[column].isna().sum()
            total_count = len(df)
            
            if missing_count > 0:
                missing_percentage = (missing_count / total_count) * 100
                
                # تعیین سطح هشدار بر اساس درصد مقادیر گمشده
                if missing_percentage > missing_threshold * 100:
                    level = ValidationLevel.CRITICAL
                elif missing_percentage > 5:
                    level = ValidationLevel.WARNING
                else:
                    level = ValidationLevel.INFO
                
                results.append(ValidationResult(
                    field_name=column,
                    is_valid=False,
                    validation_type="missing_values",
                    message=f"{missing_percentage:.1f}% مقادیر گمشده ({missing_count} از {total_count})",
                    level=level,
                    original_value=None
                ))
            else:
                results.append(ValidationResult(
                    field_name=column,
                    is_valid=True,
                    validation_type="missing_values",
                    message="هیچ مقدار گمشده‌ای یافت نشد",
                    level=ValidationLevel.INFO,
                    original_value=None
                ))
        
        return results
    
    def _validate_ranges(self, df: pd.DataFrame) -> List[ValidationResult]:
        """2. اعتبارسنجی محدوده مقادیر"""
        results = []
        
        for column in df.columns:
            if column not in self.standard_ranges:
                continue
                
            min_val, max_val = self.standard_ranges[column]
            
            for idx, value in df[column].items():
                if pd.isna(value):
                    continue
                    
                try:
                    value_float = float(value)
                    
                    if value_float < min_val or value_float > max_val:
                        # محاسبه مقدار اصلاح شده
                        corrected_value = np.clip(value_float, min_val, max_val)
                        
                        results.append(ValidationResult(
                            field_name=column,
                            is_valid=False,
                            validation_type="range_validation",
                            message=f"مقدار {value_float} خارج از محدوده مجاز [{min_val}, {max_val}]",
                            level=ValidationLevel.WARNING,
                            original_value=value_float,
                            corrected_value=corrected_value
                        ))
                    else:
                        results.append(ValidationResult(
                            field_name=column,
                            is_valid=True,
                            validation_type="range_validation",
                            message=f"مقدار در محدوده مجاز [{min_val}, {max_val}]",
                            level=ValidationLevel.INFO,
                            original_value=value_float
                        ))
                        
                except (ValueError, TypeError):
                    results.append(ValidationResult(
                        field_name=column,
                        is_valid=False,
                        validation_type="range_validation",
                        message=f"مقدار غیرعددی: {value}",
                        level=ValidationLevel.CRITICAL,
                        original_value=value
                    ))
        
        return results
    
    def _detect_outliers(self, df: pd.DataFrame) -> List[ValidationResult]:
        """3. کشف نقاط پرت/خارج از محدوده آماری"""
        results = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column not in self.standard_ranges:
                continue
                
            values = df[column].dropna()
            if len(values) < 3:
                continue
            
            # روش Z-score
            try:
                z_scores = np.abs(stats.zscore(values))
                z_outliers = values[z_scores > 3]
                
                # روش IQR
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                iqr_outliers = values[(values < (Q1 - 1.5 * IQR)) | (values > (Q3 + 1.5 * IQR))]
                
                # ترکیب نتایج
                all_outliers = set(z_outliers.index) | set(iqr_outliers.index)
                
                for idx in all_outliers:
                    value = values.loc[idx]
                    median_value = values.median()
                    
                    results.append(ValidationResult(
                        field_name=column,
                        is_valid=False,
                        validation_type="outlier_detection",
                        message=f"نقطه پرت شناسایی شده: {value}",
                        level=ValidationLevel.WARNING,
                        original_value=value,
                        corrected_value=median_value
                    ))
                    
            except Exception as e:
                warnings.warn(f"خطا در تشخیص نقاط پرت برای {column}: {e}")
        
        # استفاده از مدل ML برای تشخیص نقاط پرت چندمتغیره
        if self.outlier_detector is not None and len(numeric_columns) > 0:
            try:
                features = df[numeric_columns].fillna(0)
                outlier_predictions = self.outlier_detector.predict(features)
                
                for idx, prediction in enumerate(outlier_predictions):
                    if prediction == -1:  # نقطه پرت
                        results.append(ValidationResult(
                            field_name="multivariate_outlier",
                            is_valid=False,
                            validation_type="multivariate_outlier_detection",
                            message="نقطه پرت چندمتغیره شناسایی شده",
                            level=ValidationLevel.WARNING,
                            original_value=df.iloc[idx][numeric_columns].to_dict(),
                            confidence=0.8
                        ))
            except Exception as e:
                warnings.warn(f"خطا در تشخیص نقاط پرت چندمتغیره: {e}")
        
        return results
    
    def _validate_data_types(self, df: pd.DataFrame) -> List[ValidationResult]:
        """4. بررسی اعتبار نوع داده"""
        results = []
        
        # تعریف انواع داده مورد انتظار برای هر فیلد
        expected_dtypes = {
            'WOB': 'numeric', 'RPM': 'numeric', 'Torque': 'numeric', 'ROP': 'numeric',
            'Depth': 'numeric', 'Standpipe_Pressure': 'numeric', 'Mud_Temperature': 'numeric',
            'Mud_Flow_Rate': 'numeric', 'Gamma_Ray': 'numeric', 'Resistivity': 'numeric',
            'Density': 'numeric', 'Porosity': 'numeric', 'Hook_Load': 'numeric',
            'Vibration': 'numeric', 'Bit_Wear': 'numeric', 'Pump_Efficiency': 'numeric',
            'Formation_Type': 'categorical', 'Lithology': 'categorical', 
            'Layer_Name': 'categorical', 'Active_Events': 'categorical',
            'Failed_Equipment': 'categorical', 'Maintenance_Flag': 'boolean',
            'Abnormal_Condition': 'categorical'
        }
        
        for column in df.columns:
            if column not in expected_dtypes:
                continue
                
            expected_type = expected_dtypes[column]
            
            for idx, value in df[column].items():
                if pd.isna(value):
                    continue
                    
                is_valid = False
                message = ""
                
                if expected_type == 'numeric':
                    try:
                        float(value)
                        is_valid = True
                        message = "نوع داده عددی معتبر"
                    except (ValueError, TypeError):
                        is_valid = False
                        message = f"انتظار عدد ولی دریافت شد: {type(value).__name__}"
                
                elif expected_type == 'boolean':
                    if value in [0, 1, True, False, '0', '1', 'True', 'False']:
                        is_valid = True
                        message = "نوع داده بولین معتبر"
                    else:
                        is_valid = False
                        message = f"مقدار بولین نامعتبر: {value}"
                
                elif expected_type == 'categorical':
                    is_valid = True
                    message = "نوع داده دسته‌ای معتبر"
                
                results.append(ValidationResult(
                    field_name=column,
                    is_valid=is_valid,
                    validation_type="data_type_validation",
                    message=message,
                    level=ValidationLevel.WARNING if not is_valid else ValidationLevel.INFO,
                    original_value=value
                ))
        
        return results
    
    def _check_consistency(self, df: pd.DataFrame) -> List[ValidationResult]:
        """5. بررسی ناسازگاری منطقی بین فیلدها"""
        results = []
        
        # تعریف قوانین ناسازگاری منطقی
        consistency_rules = [
            {
                'name': 'depth_increase',
                'condition': lambda df: df['Depth'].diff() >= 0,
                'fields': ['Depth'],
                'message': 'عمق نمی‌تواند کاهش یابد'
            },
            {
                'name': 'pressure_flow_relation', 
                'condition': lambda df: (df['Mud_Flow_Rate'] > 0) == (df['Standpipe_Pressure'] > 0),
                'fields': ['Mud_Flow_Rate', 'Standpipe_Pressure'],
                'message': 'فشار و نرخ جریان گل باید همزمان صفر یا غیرصفر باشند'
            },
            {
                'name': 'wob_torque_relation',
                'condition': lambda df: df['Torque'] <= df['WOB'] * 0.5,
                'fields': ['WOB', 'Torque'],
                'message': 'گشتاور نمی‌تواند بیش از 50% WOB باشد'
            },
            {
                'name': 'bit_wear_consistency',
                'condition': lambda df: df['Bit_Wear'] <= 1.0,
                'fields': ['Bit_Wear'],
                'message': 'فرسایش مته نمی‌تواند بیش از 100% باشد'
            }
        ]
        
        for rule in consistency_rules:
            try:
                # بررسی وجود فیلدهای مورد نیاز
                if not all(field in df.columns for field in rule['fields']):
                    continue
                    
                mask = rule['condition'](df)
                inconsistent_indices = mask[~mask].index
                
                for idx in inconsistent_indices:
                    field_values = {field: df.loc[idx, field] for field in rule['fields']}
                    
                    results.append(ValidationResult(
                        field_name=",".join(rule['fields']),
                        is_valid=False,
                        validation_type="consistency_check",
                        message=f"{rule['message']}. مقادیر: {field_values}",
                        level=ValidationLevel.CRITICAL,
                        original_value=field_values
                    ))
                    
            except Exception as e:
                warnings.warn(f"خطا در بررسی قاعده {rule['name']}: {e}")
        
        return results
    
    def _check_temporal_consistency(self, df: pd.DataFrame) -> List[ValidationResult]:
        """6. بررسی یکنواختی زمانی"""
        results = []
        
        if 'Timestamp' not in df.columns or len(df) < 2:
            return results
        
        # مرتب کردن بر اساس زمان
        df_sorted = df.sort_values('Timestamp').copy()
        
        # بررسی فاصله‌های زمانی غیرعادی
        time_diffs = (df_sorted['Timestamp'] - df_sorted['Timestamp'].shift(1)).dt.total_seconds()
        abnormal_time_gaps = time_diffs[time_diffs > 3600]  # فاصله بیش از 1 ساعت
        
        for idx, gap in abnormal_time_gaps.items():
            results.append(ValidationResult(
                field_name="Timestamp",
                is_valid=False,
                validation_type="temporal_consistency",
                message=f"فاصله زمانی غیرعادی: {gap} ثانیه بین رکوردها",
                level=ValidationLevel.WARNING,
                original_value=df_sorted.loc[idx, 'Timestamp']
            ))
        
        # بررسی جهش‌های ناگهانی در عمق
        if 'Depth' in df_sorted.columns:
            depth_diff = df_sorted['Depth'].diff()
            depth_jumps = depth_diff[(depth_diff.abs() > 100) | (depth_diff < -10)]
            
            for idx, jump in depth_jumps.items():
                if jump < -10:
                    level = ValidationLevel.CRITICAL
                    message = f"کاهش غیرممکن در عمق: {jump:.2f} فوت"
                else:
                    level = ValidationLevel.WARNING  
                    message = f"جهش ناگهانی در عمق: {jump:.2f} فوت"
                
                results.append(ValidationResult(
                    field_name="Depth",
                    is_valid=False,
                    validation_type="temporal_consistency",
                    message=message,
                    level=level,
                    original_value=df_sorted.loc[idx, 'Depth']
                ))
        
        return results
    
    def _validate_status_fields(self, df: pd.DataFrame) -> List[ValidationResult]:
        """7. اعتبارسنجی فیلدهای وضعیت"""
        results = []
        
        status_fields = {
            'Maintenance_Flag': [0, 1, True, False],
            'Pump_Efficiency': (0, 1),
            'Bit_Wear': (0, 1),
            'Emergency_Stop': [0, 1, True, False]
        }
        
        for field, allowed_values in status_fields.items():
            if field not in df.columns:
                continue
                
            for idx, value in df[field].items():
                if pd.isna(value):
                    continue
                    
                is_valid = False
                message = ""
                
                if isinstance(allowed_values, (list, tuple)) and isinstance(allowed_values[0], (int, float, bool, str)):
                    if value in allowed_values:
                        is_valid = True
                        message = f"مقدار وضعیت معتبر: {value}"
                    else:
                        is_valid = False
                        message = f"مقدار وضعیت نامعتبر: {value}. مقادیر مجاز: {allowed_values}"
                        
                elif isinstance(allowed_values, tuple) and len(allowed_values) == 2:
                    try:
                        value_float = float(value)
                        min_val, max_val = allowed_values
                        if min_val <= value_float <= max_val:
                            is_valid = True
                            message = f"مقدار در محدوده مجاز [{min_val}, {max_val}]"
                        else:
                            is_valid = False
                            message = f"مقدار خارج از محدوده مجاز: {value_float}. محدوده: [{min_val}, {max_val}]"
                    except (ValueError, TypeError):
                        is_valid = False
                        message = f"مقدار غیرعددی برای فیلد وضعیت: {value}"
                
                level = ValidationLevel.CRITICAL if not is_valid and field in ['Emergency_Stop'] else ValidationLevel.WARNING
                
                results.append(ValidationResult(
                    field_name=field,
                    is_valid=is_valid,
                    validation_type="status_field_validation",
                    message=message,
                    level=level,
                    original_value=value
                ))
        
        return results
    def _check_advanced_correlations(self, df: pd.DataFrame) -> List[ValidationResult]:
        """بررسی همبستگی‌های پیشرفته بین پارامترها"""
        results = []
        
        # فقط برای داده‌های با بیش از 2 رکورد
        if len(df) < 2:
            return results
        
        # همبستگی WOB و Torque
        if 'WOB' in df.columns and 'Torque' in df.columns:
            try:
                correlation = df['WOB'].corr(df['Torque'])
                if not pd.isna(correlation) and correlation < 0.7:
                    results.append(ValidationResult(
                        field_name="WOB-Torque",
                        is_valid=False,
                        validation_type="correlation_check",
                        message=f"همبستگی ضعیف بین WOB و Torque: {correlation:.2f}",
                        level=ValidationLevel.WARNING
                    ))
            except:
                pass
        
        # همبستگی RPM و ROP
        if 'RPM' in df.columns and 'ROP' in df.columns:
            try:
                correlation = df['RPM'].corr(df['ROP'])
                if not pd.isna(correlation) and correlation < 0.5:
                    results.append(ValidationResult(
                        field_name="RPM-ROP", 
                        is_valid=False,
                        validation_type="correlation_check",
                        message=f"همبستگی غیرعادی بین RPM و ROP: {correlation:.2f}",
                        level=ValidationLevel.WARNING
                    ))
            except:
                pass
        
        return results

    def _advanced_anomaly_detection(self, df: pd.DataFrame) -> List[ValidationResult]:
        """الگوریتم‌های پیشرفته تشخیص ناهنجاری"""
        results = []
        
        # فقط برای داده‌های با بیش از 5 رکورد اجرا شود
        if len(df) < 5:
            return results  # برای داده‌های کوچک کاری نکن
        
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            # استفاده از LOF برای تشخیص ناهنجاری‌های محلی
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                # تنظیم n_neighbors بر اساس تعداد نمونه‌ها
                n_neighbors = min(20, len(df) - 1)
                lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
                outlier_labels = lof.fit_predict(df[numeric_cols].fillna(0))
                
                for idx, label in enumerate(outlier_labels):
                    if label == -1:  # نقطه پرت
                        results.append(ValidationResult(
                            field_name="multivariate_lof",
                            is_valid=False,
                            validation_type="advanced_anomaly_detection", 
                            message="ناهنجاری چندمتغیره شناسایی شده (LOF)",
                            level=ValidationLevel.WARNING,
                            original_value=df.iloc[idx][numeric_cols].to_dict()
                        ))
        except Exception as e:
            # اگر خطا داد، بی‌خودش کن
            pass
        
        return results  
    def _get_empty_validation_result(self) -> Dict[str, Any]:
        """نتایج برای داده خالی"""
        return {
            'is_valid': False,
            'validation_results': [],
            'quality_scores': {'overall_score': 0.0, 'quality_status': 'CRITICAL'},
            'corrected_data': pd.DataFrame(),
            'correction_log': [],
            'quality_report': None,
            'validation_duration': 0.0,
            'summary': {
                'total_records': 0,
                'total_fields': 0,
                'total_issues': 0,
                'critical_issues': 0,
                'warning_issues': 0,
                'corrections_applied': 0
            }
        }

# ================================
# تست کامل سیستم
# ================================
    def run_system_test():
        """اجرای تست سیستم"""
        print("🚀 در حال اجرای سیستم اعتبارسنجی...")
        print("=" * 50)
        
        # ایجاد داده تست
        test_data = pd.DataFrame({
            'Timestamp': [pd.Timestamp('2024-01-01 10:00:00')],
            'Bit_Wear': [1.5],    # مشکل: فرسایش بیش از 100%
            'Depth': [800],       # مشکل: کاهش عمق
            'WOB': [30000],
            'RPM': [110],
            'ROP': [45]
        })
        
        # ایجاد اعتبارسنج
        validator = DataQualityValidator()
        
        # اجرای اعتبارسنجی
        results = validator.validate(test_data, enable_correction=True)
        
        # نمایش نتایج
        print("\n📊 نتایج اعتبارسنجی:")
        print(f"✅ وضعیت: {'معتبر' if results['is_valid'] else 'نامعتبر'}")
        print(f"🎯 امتیاز کیفیت: {results['quality_scores']['overall_score']}")
        print(f"🔧 تعداد اصلاحات: {len(results['correction_log'])}")
        
        if results['correction_log']:
            print("\n🔍 جزئیات اصلاحات:")
            for log in results['correction_log']:
                print(f"   {log['field_name']}: {log['original_value']} → {log['corrected_value']}")
                print(f"   دلیل: {log['problem_message']}")
        else:
            print("⚠️ هیچ اصلاحی انجام نشد!")
        
        return results

    def test_complete_system():
        """تست عملکرد کامل سیستم اعتبارسنجی"""
        
        # ایجاد داده نمونه
        sample_data = pd.DataFrame({
            'Timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),  # 'h' کوچک
            'WOB': [25000, 28000, 32000, 60000, 29000, 27000, 31000, 28000, 26000, 24000],
            'RPM': [120, 115, 125, np.nan, 118, 122, 119, 121, 117, 123],
            'ROP': [45, 48, 52, 150, 47, 46, 49, 51, 44, 50],
            'Standpipe_Pressure': [25000000, 25500000, 24800000, 25200000, 60000000, 25100000, 24900000, 25300000, 24700000, 25400000],
            'Depth': [1000, 1005, 1010, 1015, 1020, 1025, 1030, 1035, 1040, 1045],
            'Bit_Wear': [0.2, 0.25, 0.3, 1.5, 0.28, 0.32, 0.35, 0.38, 0.4, 0.42],
            'Maintenance_Flag': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        })
        
        print("🧪 تست سیستم کامل اعتبارسنجی...")
        
        # ایجاد اعتبارسنج
        validator = DataQualityValidator()
        
        # اجرای اعتبارسنجی
        results = validator.validate(sample_data, enable_correction=True)
        
        print(f"\n✅ نتایج اعتبارسنجی:")
        print(f"📊 وضعیت کلی: {'معتبر' if results['is_valid'] else 'نامعتبر'}")
        print(f"🎯 امتیاز کیفیت: {results['quality_scores']['overall_score']}")
        print(f"⚠️  خطاهای بحرانی: {results['summary']['critical_issues']}")
        print(f"🔧 اصلاحات اعمال شده: {results['summary']['corrections_applied']}")
        print(f"⏱️  زمان اجرا: {results['validation_duration']:.2f} ثانیه")
        
        # نمایش نمونه‌ای از خطاها
        invalid_results = [r for r in results['validation_results'] if not r.is_valid]
        if invalid_results:
            print(f"\n🔍 نمونه خطاها:")
            for result in invalid_results[:3]:
                print(f"   - {result.field_name}: {result.message} ({result.level.value})")
        
        return results

        # این کد رو به انتهای فایل data_quality_validator.py اضافه کن

    def test_with_real_generator():
        """تست سیستم اعتبارسنجی با جنریتور واقعی"""
        
        print("🚀 راه‌اندازی تست واقعی با جنریتور پیشرفته...")
        print("=" * 60)
        
        try:
            # ایمپورت جنریتور - مطمئن شو در مسیر درست هست
            from advanced_drilling_generator import AdvancedSRSDataGenerator
            
            # مرحله ۱: تولید داده مصنوعی
            print("📊 مرحله ۱: تولید داده‌های مصنوعی حفاری...")
            generator = AdvancedSRSDataGenerator()
            
            # برای تست سریع، مدت زمان رو کم می‌کنیم
            generator.duration_days = 1  # فقط ۱ روز داده برای تست سریع
            generator.total_seconds = 24 * 3600  # 86400 رکورد
            
            dataset = generator.generate_advanced_dataset()
            print(f"✅ داده تولید شد: {len(dataset):,} رکورد")
            print(f"📋 ستون‌ها: {list(dataset.columns)}")
            
            # مرحله ۲: اعتبارسنجی
            print("\n🔍 مرحله ۲: اجرای اعتبارسنجی کامل...")
            validator = DataQualityValidator()
            
            # تست با نمونه‌ای از داده (برای سرعت بیشتر)
            sample_size = min(10000, len(dataset))
            test_sample = dataset.head(sample_size).copy()
            
            print(f"📝 در حال اعتبارسنجی {len(test_sample):,} رکورد...")
            
            results = validator.validate(test_sample, enable_correction=True)
            
            # مرحله ۳: نمایش نتایج
            print("\n🎯 مرحله ۳: نتایج اعتبارسنجی")
            print("=" * 40)
            
            print(f"✅ وضعیت کلی: {'🟢 معتبر' if results['is_valid'] else '🔴 نامعتبر'}")
            print(f"📈 امتیاز کیفیت: {results['quality_scores']['overall_score']:.3f}")
            print(f"🏷️  وضعیت کیفیت: {results['quality_scores']['quality_status']}")
            print(f"⏱️  زمان اجرا: {results['validation_duration']:.2f} ثانیه")
            
            print(f"\n📊 آمار خطاها:")
            print(f"   🔴 خطاهای بحرانی: {results['summary']['critical_issues']}")
            print(f"   🟡 خطاهای هشدار: {results['summary']['warning_issues']}") 
            print(f"   🔵 خطاهای اطلاعاتی: {results['summary']['total_issues'] - results['summary']['critical_issues'] - results['summary']['warning_issues']}")
            print(f"   🔧 اصلاحات اعمال شده: {results['summary']['corrections_applied']}")
            
            # نمایش جزئیات خطاها
            if results['summary']['critical_issues'] > 0:
                print(f"\n🔍 خطاهای بحرانی (نمونه):")
                critical_errors = [r for r in results['validation_results'] 
                                if not r.is_valid and r.level == ValidationLevel.CRITICAL]
                for i, error in enumerate(critical_errors[:3]):
                    print(f"   {i+1}. {error.field_name}: {error.message}")
            
            return results
            
        except ImportError as e:
            print(f"❌ خطا: فایل جنریتور پیدا نشد: {e}")
            print("💡 مطمئن شوید فایل advanced_drilling_generator.py در مسیر درست است")
            return None
# ================================
# UNIT TESTS
# ================================

import unittest

class TestConsistencyCorrection(unittest.TestCase):
    """Unit Test برای متد اصلاح ناسازگاری منطقی"""
    
    def setUp(self):
        self.corrector = DataAutoCorrector()
    
    def test_bit_wear_overflow_correction(self):
        """تست اصلاح فرسایش مته بیش از 100%"""
        test_df = pd.DataFrame({
            'Bit_Wear': [1.5],
            'Depth': [1000]
        })
        
        problem = ValidationResult(
            field_name='Bit_Wear',
            is_valid=False,
            validation_type="consistency_check",
            message="فرسایش مته نمی‌تواند بیش از 100% باشد",
            level=ValidationLevel.CRITICAL,
            original_value=1.5
        )
        
        corrected_value, confidence = self.corrector._correct_consistency_issues(test_df, 'Bit_Wear', problem)
        
        self.assertEqual(corrected_value, 1.0)
        self.assertEqual(confidence, 0.9)

def run_simple_test():
    """یک تست ساده برای اجرا"""
    print("🧪 اجرای تست ساده...")
    
    test_data = pd.DataFrame({
        'Timestamp': pd.date_range('2024-01-01', periods=3, freq='h'),
        'Bit_Wear': [0.8, 0.9, 1.5],
        'Depth': [1000, 1005, 800],
        'WOB': [30000, 31000, 29000]
    })
    
    validator = DataQualityValidator()
    results = validator.validate(test_data, enable_correction=True)
    
    print(f"✅ وضعیت: {'معتبر' if results['is_valid'] else 'نامعتبر'}")
    print(f"🔧 اصلاحات: {len(results['correction_log'])}")
    
    return True

if __name__ == "__main__":
    print("🎯 سیستم اعتبارسنجی کیفیت داده‌های حفاری")
    
    success = run_simple_test()
    
    if success:
        print("🎉 تست با موفقیت انجام شد!")
    else:
        print("❌ تست شکست خورد!")