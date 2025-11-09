from __future__ import annotations

from typing import Dict, Any, Optional, List
import logging

from sqlalchemy import desc

from api.models.database_models import DVRProcessHistoryDB
from database import db_manager
from processing.dvr_stats import run_statistical_checks
from processing.dvr_reconciliation import reconcile_data
from processing.dvr import (
    get_last_n_rows,
    get_history_for_anomaly,
    flag_anomaly,
)

logger = logging.getLogger(__name__)


class DVRService:
    """Service layer for Data Validation & Reconciliation operations."""

    def __init__(self):
        self.db_manager = db_manager

    # ------------------------------------------------------------------ #
    # Core processing
    # ------------------------------------------------------------------ #
    def process_record(self, record: Dict[str, Any], source: str = "api") -> Dict[str, Any]:
        """Validate, reconcile, and persist a DVR record."""
        raw_payload = record.copy()
        try:
            is_valid, reason = run_statistical_checks(record)
            if not is_valid:
                history_id = self._persist_history(
                    raw_record=raw_payload,
                    reconciled_record=None,
                    is_valid=False,
                    reason=reason,
                    status="invalid",
                    anomaly_flag=False,
                    anomaly_details=None,
                    source=source,
                )
                return {
                    "success": False,
                    "processed_record": None,
                    "message": reason,
                    "history_id": history_id,
                }

            reconciled_record = reconcile_data(record.copy())
            anomaly_flag = False
            anomaly_details: Optional[Dict[str, Any]] = None

            try:
                history_dict, numeric_cols = get_history_for_anomaly(100)
                evaluated = flag_anomaly(reconciled_record.copy(), history_dict, numeric_cols)
                anomaly_flag = bool(evaluated.get("Anomaly"))
                anomaly_details = {"Anomaly": anomaly_flag}
            except Exception as anomaly_exc:  # pragma: no cover - best effort only
                logger.debug(f"Anomaly evaluation skipped: {anomaly_exc}")

            history_id = self._persist_history(
                raw_record=raw_payload,
                reconciled_record=reconciled_record,
                is_valid=True,
                reason=reason,
                status="processed",
                anomaly_flag=anomaly_flag,
                anomaly_details=anomaly_details,
                source=source,
            )

            return {
                "success": True,
                "processed_record": reconciled_record,
                "message": "Record processed successfully",
                "history_id": history_id,
                "anomaly_flag": anomaly_flag,
            }
        except Exception as exc:
            logger.error(f"Error processing DVR record: {exc}")
            history_id = self._persist_history(
                raw_record=raw_payload,
                reconciled_record=None,
                is_valid=False,
                reason=str(exc),
                status="error",
                anomaly_flag=False,
                anomaly_details=None,
                source=source,
            )
            return {
                "success": False,
                "processed_record": None,
                "message": str(exc),
                "history_id": history_id,
            }

    # ------------------------------------------------------------------ #
    # Analytics & anomaly helpers
    # ------------------------------------------------------------------ #
    def get_recent_stats(self, limit: int = 50) -> Dict[str, Any]:
        try:
            if self._db_ready():
                entries = self.get_history(limit=limit)
                if entries:
                    processed = sum(1 for entry in entries if entry["status"] == "processed")
                    invalid = sum(1 for entry in entries if entry["status"] == "invalid")
                    summary = {
                        "count": len(entries),
                        "processed": processed,
                        "invalid": invalid,
                        "rig_ids": sorted({entry["rig_id"] for entry in entries if entry.get("rig_id")}),
                        "latest": entries[0],
                        "recent_invalid_reasons": [entry["reason"] for entry in entries if not entry["is_valid"] and entry.get("reason")][:5],
                    }
                    return {"success": True, "summary": summary, "message": None}

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

    # ------------------------------------------------------------------ #
    # History CRUD helpers
    # ------------------------------------------------------------------ #
    def get_history(self, limit: int = 100, rig_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._db_ready():
            return []
        try:
            with self.db_manager.session_scope() as session:
                query = session.query(DVRProcessHistoryDB)
                if rig_id:
                    query = query.filter(DVRProcessHistoryDB.rig_id == rig_id)
                if status:
                    query = query.filter(DVRProcessHistoryDB.status == status)
                entries = (
                    query.order_by(desc(DVRProcessHistoryDB.created_at))
                    .limit(limit)
                    .all()
                )
                return [self._history_to_dict(entry) for entry in entries]
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Error retrieving DVR history: {exc}")
            return []

    def update_history_entry(self, entry_id: int, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._db_ready():
            return None
        try:
            with self.db_manager.session_scope() as session:
                entry = session.query(DVRProcessHistoryDB).filter(DVRProcessHistoryDB.id == entry_id).first()
                if entry is None:
                    return None
                if "status" in updates and updates["status"]:
                    entry.status = updates["status"]
                if "notes" in updates:
                    entry.notes = updates["notes"]
                session.flush()
                session.refresh(entry)
                return self._history_to_dict(entry)
        except Exception as exc:
            logger.error(f"Error updating DVR history entry {entry_id}: {exc}")
            return None

    def delete_history_entry(self, entry_id: int) -> bool:
        if not self._db_ready():
            return False
        try:
            with self.db_manager.session_scope() as session:
                entry = session.query(DVRProcessHistoryDB).filter(DVRProcessHistoryDB.id == entry_id).first()
                if entry is None:
                    return False
                session.delete(entry)
                return True
        except Exception as exc:
            logger.error(f"Error deleting DVR history entry {entry_id}: {exc}")
            return False

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _persist_history(
        self,
        raw_record: Dict[str, Any],
        reconciled_record: Optional[Dict[str, Any]],
        is_valid: bool,
        reason: Optional[str],
        status: str,
        anomaly_flag: bool,
        anomaly_details: Optional[Dict[str, Any]],
        source: str,
    ) -> Optional[int]:
        if not self._db_ready():
            return None
        try:
            with self.db_manager.session_scope() as session:
                entry = DVRProcessHistoryDB(
                    rig_id=raw_record.get("rig_id") or raw_record.get("Rig_ID"),
                    raw_record=raw_record,
                    reconciled_record=reconciled_record,
                    is_valid=is_valid,
                    reason=reason,
                    anomaly_flag=anomaly_flag,
                    anomaly_details=anomaly_details,
                    status=status,
                    source=source,
                )
                session.add(entry)
                session.flush()
                entry_id = entry.id
            return entry_id
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Failed to persist DVR history: {exc}")
            return None

    def _history_to_dict(self, entry: DVRProcessHistoryDB) -> Dict[str, Any]:
        return {
            "id": entry.id,
            "rig_id": entry.rig_id,
            "raw_record": entry.raw_record,
            "reconciled_record": entry.reconciled_record,
            "is_valid": entry.is_valid,
            "reason": entry.reason,
            "anomaly_flag": entry.anomaly_flag,
            "anomaly_details": entry.anomaly_details,
            "status": entry.status,
            "notes": entry.notes,
            "source": entry.source,
            "created_at": entry.created_at.isoformat() if entry.created_at else None,
            "updated_at": entry.updated_at.isoformat() if entry.updated_at else None,
        }

    def _db_ready(self) -> bool:
        return getattr(self.db_manager, "_initialized", False)


dvr_service = DVRService()
