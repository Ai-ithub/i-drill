"""
Alert Management Service
Comprehensive alert system with visual alerts, sound alerts, email/SMS notifications,
acknowledgment system, alert history, and escalation rules
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum

from services.websocket_manager import websocket_manager
from services.email_service import email_service
from api.models.database_models import UserDB
from database import db_manager

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(str, Enum):
    """Alert types"""
    SAFETY = "safety"
    KICK = "kick"
    STUCK_PIPE = "stuck_pipe"
    FORMATION_CHANGE = "formation_change"
    DVR_VALIDATION = "dvr_validation"
    MAINTENANCE = "maintenance"
    EQUIPMENT_FAILURE = "equipment_failure"
    SYSTEM = "system"
    CUSTOM = "custom"


class AlertStatus(str, Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class AlertManagementService:
    """
    Comprehensive Alert Management Service.
    
    Features:
    - Real-time visual alerts in dashboard
    - Sound alerts for critical conditions
    - Email/SMS notifications for engineers
    - Acknowledgment system
    - Alert history
    - Escalation rules
    """
    
    def __init__(self):
        """Initialize AlertManagementService."""
        # Alert storage (in-memory for now, can be persisted to DB)
        self.active_alerts: Dict[int, Dict[str, Any]] = {}  # alert_id -> alert_data
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_counter = 0
        
        # Escalation rules
        self.escalation_rules = {
            "critical": {
                "acknowledge_timeout_minutes": 5,
                "escalate_to": ["admin", "engineer"],
                "auto_escalate": True
            },
            "high": {
                "acknowledge_timeout_minutes": 15,
                "escalate_to": ["engineer"],
                "auto_escalate": True
            },
            "medium": {
                "acknowledge_timeout_minutes": 30,
                "escalate_to": ["engineer"],
                "auto_escalate": False
            },
            "low": {
                "acknowledge_timeout_minutes": 60,
                "escalate_to": [],
                "auto_escalate": False
            }
        }
        
        # Sound alert configuration
        self.sound_alerts_enabled = True
        self.sound_alert_severities = ["critical", "high"]  # Severities that trigger sound
        
        # Email notification configuration
        self.email_notifications_enabled = True
        self.email_notification_severities = ["critical", "high", "medium"]
        self.email_notification_roles = ["admin", "engineer"]  # Roles to notify
        
        # SMS notification configuration (placeholder - requires SMS service)
        self.sms_notifications_enabled = False
        self.sms_notification_severities = ["critical"]
        self.sms_notification_roles = ["admin", "engineer"]
        
        # Escalation monitoring
        self.escalation_check_interval = 60  # Check every minute
        self.escalation_monitoring_running = False
        
        logger.info("Alert management service initialized")
    
    def create_alert(
        self,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        rig_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        requires_acknowledgment: bool = True,
        sound_alert: Optional[bool] = None  # None = auto-detect based on severity
    ) -> Dict[str, Any]:
        """
        Create a new alert.
        
        Args:
            alert_type: Type of alert (safety, kick, stuck_pipe, etc.)
            severity: Severity level (critical, high, medium, low, info)
            title: Alert title
            message: Alert message
            rig_id: Optional rig ID
            metadata: Optional additional data
            requires_acknowledgment: Whether alert requires acknowledgment
            sound_alert: Whether to play sound (None = auto-detect)
        
        Returns:
            Created alert dictionary
        """
        self.alert_counter += 1
        alert_id = self.alert_counter
        
        # Auto-detect sound alert if not specified
        if sound_alert is None:
            sound_alert = severity in self.sound_alert_severities
        
        alert = {
            "id": alert_id,
            "alert_type": alert_type,
            "severity": severity,
            "title": title,
            "message": message,
            "rig_id": rig_id,
            "status": AlertStatus.ACTIVE.value,
            "requires_acknowledgment": requires_acknowledgment,
            "sound_alert": sound_alert,
            "created_at": datetime.now().isoformat(),
            "acknowledged_at": None,
            "acknowledged_by": None,
            "resolved_at": None,
            "resolved_by": None,
            "escalated": False,
            "escalated_at": None,
            "escalated_to": [],
            "metadata": metadata or {}
        }
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert.copy())
        
        # Keep only last 10000 alerts in history
        if len(self.alert_history) > 10000:
            self.alert_history = self.alert_history[-10000:]
        
        # Send real-time alert
        asyncio.create_task(self._send_realtime_alert(alert))
        
        # Send email/SMS notifications
        if severity in self.email_notification_severities:
            asyncio.create_task(self._send_email_notifications(alert))
        
        if severity in self.sms_notification_severities and self.sms_notifications_enabled:
            asyncio.create_task(self._send_sms_notifications(alert))
        
        logger.info(f"Alert created: {alert_type} - {severity} - {title} (ID: {alert_id})")
        
        return alert
    
    async def _send_realtime_alert(self, alert: Dict[str, Any]) -> None:
        """Send real-time alert via WebSocket."""
        try:
            alert_message = {
                "message_type": "alert",
                "data": {
                    "id": alert["id"],
                    "alert_type": alert["alert_type"],
                    "severity": alert["severity"],
                    "title": alert["title"],
                    "message": alert["message"],
                    "rig_id": alert.get("rig_id"),
                    "status": alert["status"],
                    "requires_acknowledgment": alert["requires_acknowledgment"],
                    "sound_alert": alert["sound_alert"],
                    "created_at": alert["created_at"],
                    "metadata": alert.get("metadata", {})
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to specific rig if specified
            if alert.get("rig_id"):
                await websocket_manager.send_to_rig(alert["rig_id"], alert_message)
            else:
                # Broadcast to all connections
                await websocket_manager.broadcast_to_all(alert_message)
        
        except Exception as e:
            logger.error(f"Error sending real-time alert: {e}")
    
    async def _send_email_notifications(self, alert: Dict[str, Any]) -> None:
        """Send email notifications to engineers."""
        if not self.email_notifications_enabled:
            return
        
        try:
            # Get engineers and admins
            with db_manager.session_scope() as session:
                users = session.query(UserDB).filter(
                    UserDB.role.in_(self.email_notification_roles),
                    UserDB.is_active == True
                ).all()
                
                for user in users:
                    if user.email:
                        # Create email content
                        subject = f"[{alert['severity'].upper()}] {alert['title']} - i-Drill"
                        
                        html_body = self._create_alert_email_html(alert, user)
                        text_body = self._create_alert_email_text(alert, user)
                        
                        # Send email
                        result = email_service._send_email(
                            to_email=user.email,
                            subject=subject,
                            text_body=text_body,
                            html_body=html_body
                        )
                        
                        if result.get("success"):
                            logger.info(f"Alert email sent to {user.email} for alert {alert['id']}")
                        else:
                            logger.warning(f"Failed to send alert email to {user.email}: {result.get('error')}")
        
        except Exception as e:
            logger.error(f"Error sending email notifications: {e}")
    
    def _create_alert_email_html(self, alert: Dict[str, Any], user: UserDB) -> str:
        """Create HTML email body for alert."""
        severity_colors = {
            "critical": "#dc2626",
            "high": "#ea580c",
            "medium": "#f59e0b",
            "low": "#3b82f6",
            "info": "#6b7280"
        }
        
        color = severity_colors.get(alert["severity"], "#6b7280")
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: {color}; color: white; padding: 20px; text-align: center; }}
                .content {{ background-color: #f9fafb; padding: 30px; }}
                .alert-box {{ background-color: white; border-left: 4px solid {color}; padding: 15px; margin: 20px 0; }}
                .footer {{ text-align: center; padding: 20px; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚨 Alert Notification</h1>
                </div>
                <div class="content">
                    <h2>{alert['title']}</h2>
                    <div class="alert-box">
                        <p><strong>Severity:</strong> {alert['severity'].upper()}</p>
                        <p><strong>Type:</strong> {alert['alert_type']}</p>
                        {f"<p><strong>Rig ID:</strong> {alert['rig_id']}</p>" if alert.get('rig_id') else ""}
                        <p><strong>Message:</strong></p>
                        <p>{alert['message']}</p>
                        <p><strong>Time:</strong> {alert['created_at']}</p>
                    </div>
                    <p>Please log in to the i-Drill dashboard to view and acknowledge this alert.</p>
                </div>
                <div class="footer">
                    <p>This is an automated alert from i-Drill System.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _create_alert_email_text(self, alert: Dict[str, Any], user: UserDB) -> str:
        """Create plain text email body for alert."""
        return f"""
        Alert Notification - i-Drill
        
        {alert['title']}
        
        Severity: {alert['severity'].upper()}
        Type: {alert['alert_type']}
        {f"Rig ID: {alert['rig_id']}" if alert.get('rig_id') else ""}
        
        Message:
        {alert['message']}
        
        Time: {alert['created_at']}
        
        Please log in to the i-Drill dashboard to view and acknowledge this alert.
        
        This is an automated alert from i-Drill System.
        """
    
    async def _send_sms_notifications(self, alert: Dict[str, Any]) -> None:
        """Send SMS notifications (placeholder - requires SMS service integration)."""
        if not self.sms_notifications_enabled:
            return
        
        # TODO: Integrate with SMS service (Twilio, AWS SNS, etc.)
        logger.info(f"SMS notification would be sent for alert {alert['id']} (SMS service not implemented)")
    
    def acknowledge_alert(
        self,
        alert_id: int,
        user_id: int,
        user_name: str,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID to acknowledge
            user_id: User ID acknowledging
            user_name: User name acknowledging
            notes: Optional acknowledgment notes
        
        Returns:
            Updated alert dictionary
        """
        if alert_id not in self.active_alerts:
            raise ValueError(f"Alert {alert_id} not found")
        
        alert = self.active_alerts[alert_id]
        
        if alert["status"] != AlertStatus.ACTIVE.value:
            raise ValueError(f"Alert {alert_id} is not active (status: {alert['status']})")
        
        # Update alert
        alert["status"] = AlertStatus.ACKNOWLEDGED.value
        alert["acknowledged_at"] = datetime.now().isoformat()
        alert["acknowledged_by"] = user_name
        alert["acknowledgment_notes"] = notes
        
        # Update in history
        for hist_alert in self.alert_history:
            if hist_alert["id"] == alert_id:
                hist_alert.update(alert)
                break
        
        # Broadcast acknowledgment
        asyncio.create_task(self._broadcast_alert_update(alert))
        
        logger.info(f"Alert {alert_id} acknowledged by {user_name} (user_id: {user_id})")
        
        return alert
    
    def resolve_alert(
        self,
        alert_id: int,
        user_id: int,
        user_name: str,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert ID to resolve
            user_id: User ID resolving
            user_name: User name resolving
            notes: Optional resolution notes
        
        Returns:
            Updated alert dictionary
        """
        if alert_id not in self.active_alerts:
            raise ValueError(f"Alert {alert_id} not found")
        
        alert = self.active_alerts[alert_id]
        
        # Update alert
        alert["status"] = AlertStatus.RESOLVED.value
        alert["resolved_at"] = datetime.now().isoformat()
        alert["resolved_by"] = user_name
        alert["resolution_notes"] = notes
        
        # If not acknowledged, acknowledge it
        if alert["status"] != AlertStatus.ACKNOWLEDGED.value:
            alert["acknowledged_at"] = alert["resolved_at"]
            alert["acknowledged_by"] = user_name
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        # Update in history
        for hist_alert in self.alert_history:
            if hist_alert["id"] == alert_id:
                hist_alert.update(alert)
                break
        
        # Broadcast resolution
        asyncio.create_task(self._broadcast_alert_update(alert))
        
        logger.info(f"Alert {alert_id} resolved by {user_name} (user_id: {user_id})")
        
        return alert
    
    async def _broadcast_alert_update(self, alert: Dict[str, Any]) -> None:
        """Broadcast alert update via WebSocket."""
        try:
            update_message = {
                "message_type": "alert_update",
                "data": alert,
                "timestamp": datetime.now().isoformat()
            }
            
            if alert.get("rig_id"):
                await websocket_manager.send_to_rig(alert["rig_id"], update_message)
            else:
                await websocket_manager.broadcast_to_all(update_message)
        
        except Exception as e:
            logger.error(f"Error broadcasting alert update: {e}")
    
    def get_active_alerts(
        self,
        rig_id: Optional[str] = None,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get active alerts with optional filtering.
        
        Args:
            rig_id: Filter by rig ID
            severity: Filter by severity
            alert_type: Filter by alert type
        
        Returns:
            List of active alerts
        """
        alerts = list(self.active_alerts.values())
        
        if rig_id:
            alerts = [a for a in alerts if a.get("rig_id") == rig_id]
        if severity:
            alerts = [a for a in alerts if a.get("severity") == severity]
        if alert_type:
            alerts = [a for a in alerts if a.get("alert_type") == alert_type]
        
        # Sort by created_at (newest first)
        alerts.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return alerts
    
    def get_alert_history(
        self,
        rig_id: Optional[str] = None,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get alert history with optional filtering.
        
        Args:
            rig_id: Filter by rig ID
            severity: Filter by severity
            alert_type: Filter by alert type
            status: Filter by status
            limit: Maximum number of results
        
        Returns:
            List of historical alerts
        """
        alerts = self.alert_history.copy()
        
        if rig_id:
            alerts = [a for a in alerts if a.get("rig_id") == rig_id]
        if severity:
            alerts = [a for a in alerts if a.get("severity") == severity]
        if alert_type:
            alerts = [a for a in alerts if a.get("alert_type") == alert_type]
        if status:
            alerts = [a for a in alerts if a.get("status") == status]
        
        # Sort by created_at (newest first)
        alerts.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return alerts[:limit]
    
    def check_escalations(self) -> List[Dict[str, Any]]:
        """
        Check for alerts that need escalation.
        
        Returns:
            List of escalated alerts
        """
        escalated = []
        now = datetime.now()
        
        for alert_id, alert in list(self.active_alerts.items()):
            if alert["status"] != AlertStatus.ACTIVE.value:
                continue
            
            severity = alert["severity"]
            rule = self.escalation_rules.get(severity)
            
            if not rule or not rule.get("auto_escalate"):
                continue
            
            created_at = datetime.fromisoformat(alert["created_at"])
            timeout_minutes = rule.get("acknowledge_timeout_minutes", 60)
            timeout = created_at + timedelta(minutes=timeout_minutes)
            
            if now >= timeout and not alert.get("escalated"):
                # Escalate alert
                alert["escalated"] = True
                alert["escalated_at"] = now.isoformat()
                alert["escalated_to"] = rule.get("escalate_to", [])
                alert["status"] = AlertStatus.ESCALATED.value
                
                escalated.append(alert)
                
                # Send escalation notifications
                asyncio.create_task(self._send_escalation_notifications(alert))
                
                logger.warning(
                    f"Alert {alert_id} escalated to {rule.get('escalate_to')} "
                    f"after {timeout_minutes} minutes without acknowledgment"
                )
        
        return escalated
    
    async def _send_escalation_notifications(self, alert: Dict[str, Any]) -> None:
        """Send escalation notifications."""
        try:
            # Send email to escalated roles
            if alert.get("escalated_to"):
                with db_manager.session_scope() as session:
                    users = session.query(UserDB).filter(
                        UserDB.role.in_(alert["escalated_to"]),
                        UserDB.is_active == True
                    ).all()
                    
                    for user in users:
                        if user.email:
                            subject = f"[ESCALATED] {alert['title']} - i-Drill"
                            message = f"Alert {alert['id']} has been escalated due to lack of acknowledgment."
                            
                            email_service._send_email(
                                to_email=user.email,
                                subject=subject,
                                text_body=message,
                                html_body=f"<p>{message}</p><p>Alert details: {alert['message']}</p>"
                            )
            
            # Broadcast escalation
            await self._broadcast_alert_update(alert)
        
        except Exception as e:
            logger.error(f"Error sending escalation notifications: {e}")
    
    def start_escalation_monitoring(self) -> None:
        """Start background escalation monitoring."""
        if self.escalation_monitoring_running:
            return
        
        self.escalation_monitoring_running = True
        import threading
        
        def monitoring_loop():
            import time
            while self.escalation_monitoring_running:
                try:
                    self.check_escalations()
                except Exception as e:
                    logger.error(f"Error in escalation monitoring: {e}")
                time.sleep(self.escalation_check_interval)
        
        thread = threading.Thread(target=monitoring_loop, daemon=True, name="AlertEscalationMonitoring")
        thread.start()
        logger.info("Alert escalation monitoring started")
    
    def stop_escalation_monitoring(self) -> None:
        """Stop background escalation monitoring."""
        self.escalation_monitoring_running = False
        logger.info("Alert escalation monitoring stopped")


# Global instance
alert_management_service = AlertManagementService()

