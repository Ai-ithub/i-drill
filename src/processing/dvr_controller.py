"""
DVR Controller Module
Handles data validation and reconciliation processing for sensor data records.
"""
import logging
from processing.dvr_stats import run_statistical_checks
from processing.dvr_reconciliation import reconcile_data
from typing import Dict, Any, Optional

# --- Setup a proper logger ---
# This sets up a professional logger that will write to a file
logging.basicConfig(
    filename="dvr_processing.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_data(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process a single sensor data record by running validation and reconciliation.
    
    This function performs a two-step process:
    1. Validates the record using statistical checks
    2. If valid, reconciles the data to correct any inconsistencies
    
    Args:
        record: Dictionary containing sensor data record to process
        
    Returns:
        Reconciled record dictionary if validation passes, None otherwise
    """
    # Step 1: Run the statistical check and get the status and reason
    is_valid, reason = run_statistical_checks(record)

    # Step 2: Check the status and decide what to do
    if not is_valid:
        # If the data is invalid, log a warning and stop processing this record
        log_message = f"Record failed validation: {record} - Reason: {reason}"
        logger.warning(log_message)
        # We can also print to the console for immediate feedback
        print(log_message) 
        return None # Return None to indicate failure

    # Step 3: If the data is valid, proceed with reconciliation
    reconciled_record = reconcile_data(record)
    log_message = f"Record successfully processed and reconciled: {reconciled_record}"
    logger.info(log_message)
    print(log_message)

    return reconciled_record