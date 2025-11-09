from typing import Dict, Any, Optional
import logging

from processing.dvr_controller import process_data
from processing.dvr import (
    get_last_n_rows,
    get_history_for_anomaly,
    flag_anomaly,
)

logger = logging.getLogger(__name__)


class DVRService:
    """Service layer for Data Validation & Reconciliation operations."""

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = process_data(record)
            if result is None:
                return {
                    "success": False,
                    "processed_record": None,
                    "message": "Record failed validation",
                }
            return {
                "success": True,
                "processed_record": result,
                "message": "Record processed successfully",
            }
        except Exception as exc:
            logger.error(f"Error processing DVR record: {exc}")
            return {
                "success": False,
                "processed_record": None,
                "message": str(exc),
            }

    def get_recent_stats(self, limit: int = 50) -> Dict[str, Any]:
        try:
            df = get_last_n_rows(limit)
            if df.empty:
                return {
                    "success": False,
                    "summary": {},
                    "message": "No DVR data available",
                }

            numeric_cols = df.select_dtypes(include=["number"]).columns
            summary = {
                "count": int(len(df)),
                "rig_ids": df["rig_id"].dropna().unique().tolist() if "rig_id" in df else [],
                "latest": df.iloc[0].to_dict(),
                "averages": {col: float(df[col].mean()) for col in numeric_cols},
            }
            return {
                "success": True,
                "summary": summary,
                "message": None,
            }
        except Exception as exc:
            logger.error(f"Error retrieving DVR stats: {exc}")
            return {
                "success": False,
                "summary": {},
                "message": str(exc),
            }

    def get_anomaly_snapshot(self, history_size: int = 100) -> Dict[str, Any]:
        try:
            history_dict, numeric_cols = get_history_for_anomaly(history_size)
            snapshot = {key: len(values) for key, values in history_dict.items()}
            return {
                "success": True,
                "numeric_columns": numeric_cols,
                "history_sizes": snapshot,
            }
        except Exception as exc:
            logger.error(f"Error retrieving DVR anomaly snapshot: {exc}")
            return {
                "success": False,
                "numeric_columns": [],
                "history_sizes": {},
                "message": str(exc),
            }

    def evaluate_record_anomaly(self, record: Dict[str, Any], history_size: int = 100) -> Dict[str, Any]:
        try:
            history_dict, numeric_cols = get_history_for_anomaly(history_size)
            evaluated = flag_anomaly(record.copy(), history_dict, numeric_cols)
            return {
                "success": True,
                "record": evaluated,
            }
        except Exception as exc:
            logger.error(f"Error evaluating DVR anomaly: {exc}")
            return {
                "success": False,
                "record": None,
                "message": str(exc),
            }


dvr_service = DVRService()
