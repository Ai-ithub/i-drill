"""
Data Validation & Reconciliation (DVR) Controller
این ماژول کنترل اصلی فرآیند اعتبارسنجی و تصحیح داده‌های سنسور است.
"""

import logging
import pandas as pd
from typing import Dict, Optional, Tuple
from processing.data_quality_validator import DataQualityValidator, DataQualityReport
from processing.data_reconciler import DataReconciler
from processing.dvr import get_last_n_rows

# --- Setup a proper logger ---
logging.basicConfig(
    filename="dvr_processing.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global validator instance (singleton pattern)
_validator_instance = None
_previous_record = {}  # Store previous record for temporal consistency checks

def get_validator() -> DataQualityValidator:
    """Get or create validator instance"""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = DataQualityValidator(
            z_score_threshold=3.0,
            iqr_multiplier=1.5,
            enable_isolation_forest=True,
            enable_temporal_checks=True
        )
    return _validator_instance


def process_data(record: Dict, use_corrections: bool = True) -> Optional[Dict]:
    """
    Processes a single sensor data record using advanced data quality validation.
    
    Args:
        record: Dictionary containing sensor data
        use_corrections: If True, apply automatic corrections to fixable issues
    
    Returns:
        Dictionary with validated/corrected data, or None if validation fails critically
    """
    global _previous_record
    
    validator = get_validator()
    
    try:
        # Step 1: Get historical data for statistical validation (last 50 records)
        historical_data = None
        try:
            historical_data = get_last_n_rows(50)
            if historical_data is not None and len(historical_data) > 0:
                logger.debug(f"Retrieved {len(historical_data)} historical records")
        except Exception as e:
            logger.warning(f"Could not retrieve historical data: {e}")
        
        # Step 2: Get previous record for temporal consistency
        previous_record = None
        rig_id = record.get("rig_id", "UNKNOWN")
        if rig_id in _previous_record:
            previous_record = _previous_record[rig_id]
        
        # Step 3: Run comprehensive validation
        validation_report: DataQualityReport = validator.validate(
            data=record,
            historical_data=historical_data,
            previous_record=previous_record
        )
        
        # Step 4: Log validation results
        logger.info(f"Validation completed for {rig_id}: Score={validation_report.overall_score:.2f}, Valid={validation_report.is_valid}")
        
        # Step 5: Handle validation failures
        if not validation_report.is_valid:
            # Log all validation issues
            error_count = sum(1 for r in validation_report.validation_results if not r.is_valid)
            logger.warning(
                f"Record from {rig_id} failed validation: {error_count} issues found. "
                f"Missing fields: {validation_report.missing_fields}, "
                f"Outlier fields: {validation_report.outlier_fields}"
            )
            
            # Log critical errors
            critical_errors = [
                r for r in validation_report.validation_results 
                if not r.is_valid and r.severity == "error"
            ]
            for error in critical_errors:
                logger.error(
                    f"CRITICAL: {error.field_name} - {error.message} "
                    f"(Original: {error.original_value})"
                )
            
            # If critical errors exist, return None (reject data)
            if critical_errors:
                logger.error(f"Rejecting record from {rig_id} due to critical errors")
                return None
        
        # Step 6: Apply corrections if enabled
        output_data = validation_report.corrected_data if use_corrections else record
        
        # Step 7: Apply data reconciliation (imputation and smoothing)
        if use_corrections:
            try:
                # Convert to DataFrame for reconciliation
                df = pd.DataFrame([output_data])
                reconciler = DataReconciler()
                df_reconciled = reconciler.reconcile(df)
                output_data = df_reconciled.iloc[0].to_dict()
                logger.debug("Data reconciliation applied")
            except Exception as e:
                logger.warning(f"Reconciliation failed: {e}, using corrected data")
        
        # Step 8: Store current record as previous for next iteration
        _previous_record[rig_id] = record.copy()
        
        # Step 9: Log success
        logger.info(
            f"Record from {rig_id} successfully processed. "
            f"Quality score: {validation_report.overall_score:.2f}"
        )
        
        # Add validation metadata to output
        output_data["_validation_metadata"] = {
            "quality_score": validation_report.overall_score,
            "is_valid": validation_report.is_valid,
            "issue_count": error_count if not validation_report.is_valid else 0,
            "missing_fields_count": len(validation_report.missing_fields),
            "outlier_fields_count": len(validation_report.outlier_fields)
        }
        
        return output_data
        
    except Exception as e:
        logger.error(f"Error processing record from {record.get('rig_id', 'UNKNOWN')}: {e}", exc_info=True)
        return None


def get_validation_report(record: Dict) -> Optional[DataQualityReport]:
    """
    Get detailed validation report without applying corrections.
    Useful for monitoring and analysis.
    
    Args:
        record: Dictionary containing sensor data
    
    Returns:
        DataQualityReport object with detailed validation results
    """
    validator = get_validator()
    
    try:
        historical_data = None
        try:
            historical_data = get_last_n_rows(50)
        except Exception:
            pass
        
        global _previous_record
        rig_id = record.get("rig_id", "UNKNOWN")
        previous_record = _previous_record.get(rig_id)
        
        report = validator.validate(
            data=record,
            historical_data=historical_data,
            previous_record=previous_record
        )
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating validation report: {e}", exc_info=True)
        return None


def reset_validator_cache():
    """Reset the previous records cache. Useful for testing or after system restart."""
    global _previous_record
    _previous_record = {}
    logger.info("Validator cache reset")