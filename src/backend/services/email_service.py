"""
Email Service
Handles sending emails for password reset, notifications, etc.
"""
from typing import Optional, Dict, Any
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# Email configuration from environment variables
SMTP_ENABLED = os.getenv("SMTP_ENABLED", "false").lower() == "true"
SMTP_HOST = os.getenv("SMTP_HOST", "localhost")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", "noreply@i-drill.local")
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "i-Drill System")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3001")
EMAIL_MAX_RETRIES = int(os.getenv("EMAIL_MAX_RETRIES", "3"))
EMAIL_RETRY_DELAY = int(os.getenv("EMAIL_RETRY_DELAY", "60"))

# Try to import email libraries (optional dependencies)
try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_LIBS_AVAILABLE = True
except ImportError:
    EMAIL_LIBS_AVAILABLE = False
    logger.warning("Email libraries not available. Email sending will be disabled.")


class EmailService:
    """
    Service for sending emails.
    
    Supports SMTP for sending emails. In development, emails are logged instead of sent.
    In production, configure SMTP settings via environment variables.
    """
    
    def __init__(self):
        """Initialize EmailService."""
        self.enabled = SMTP_ENABLED and EMAIL_LIBS_AVAILABLE
        self.smtp_host = SMTP_HOST
        self.smtp_port = SMTP_PORT
        self.smtp_user = SMTP_USER
        self.smtp_password = SMTP_PASSWORD
        self.from_email = SMTP_FROM_EMAIL
        self.from_name = SMTP_FROM_NAME
        self.use_tls = SMTP_USE_TLS
        self.frontend_url = FRONTEND_URL
        self.max_retries = EMAIL_MAX_RETRIES
        self.retry_delay = EMAIL_RETRY_DELAY
        
        if not self.enabled:
            logger.info(
                "Email service disabled. Emails will be logged instead of sent. "
                "To enable, set SMTP_ENABLED=true and configure SMTP settings."
            )
    
    def send_password_reset_email(
        self,
        email: str,
        reset_token: str,
        username: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send password reset email to user.
        
        Args:
            email: User's email address
            reset_token: Password reset token
            username: Optional username for personalization
            
        Returns:
            Dictionary containing:
            - success: Boolean indicating if email was sent/logged
            - message: Status message
            - email_logged: Boolean indicating if email was logged (for development)
            
        Example:
            ```python
            result = email_service.send_password_reset_email(
                email="user@example.com",
                reset_token="abc123...",
                username="john_doe"
            )
            ```
        """
        try:
            # Build reset link
            reset_link = f"{self.frontend_url}/auth/password/reset/confirm?token={reset_token}"
            
            # Create email content
            subject = "Password Reset Request - i-Drill"
            
            # HTML email body
            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background-color: #0891b2; color: white; padding: 20px; text-align: center; }}
                    .content {{ background-color: #f9fafb; padding: 30px; }}
                    .button {{ display: inline-block; padding: 12px 24px; background-color: #0891b2; color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                    .footer {{ text-align: center; padding: 20px; font-size: 12px; color: #666; }}
                    .warning {{ color: #dc2626; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>i-Drill System</h1>
                    </div>
                    <div class="content">
                        <h2>Password Reset Request</h2>
                        <p>Hello{' ' + username if username else ''},</p>
                        <p>You have requested to reset your password for your i-Drill account.</p>
                        <p>Click the button below to reset your password:</p>
                        <p style="text-align: center;">
                            <a href="{reset_link}" class="button">Reset Password</a>
                        </p>
                        <p>Or copy and paste this link into your browser:</p>
                        <p style="word-break: break-all; color: #0891b2;">{reset_link}</p>
                        <p class="warning">This link will expire in 24 hours.</p>
                        <p>If you did not request this password reset, please ignore this email or contact support if you have concerns.</p>
                    </div>
                    <div class="footer">
                        <p>This is an automated message from i-Drill System. Please do not reply to this email.</p>
                        <p>&copy; {datetime.now().year} i-Drill. All rights reserved.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Plain text email body
            text_body = f"""
            Password Reset Request - i-Drill
            
            Hello{(' ' + username) if username else ''},
            
            You have requested to reset your password for your i-Drill account.
            
            Click the following link to reset your password:
            {reset_link}
            
            This link will expire in 24 hours.
            
            If you did not request this password reset, please ignore this email or contact support if you have concerns.
            
            This is an automated message from i-Drill System. Please do not reply to this email.
            """
            
            # Send email with retry logic
            if self.enabled:
                result = self._send_email_with_retry(
                    to_email=email,
                    subject=subject,
                    text_body=text_body,
                    html_body=html_body
                )
                
                if result["success"]:
                    logger.info(f"Password reset email sent to {email}")
                    return {
                        "success": True,
                        "message": "Password reset email sent successfully",
                        "email_logged": False
                    }
                else:
                    logger.error(f"Failed to send password reset email to {email} after {self.max_retries} attempts: {result.get('error')}")
                    return {
                        "success": False,
                        "message": f"Failed to send email: {result.get('error', 'Unknown error')}",
                        "email_logged": False
                    }
            else:
                # Log email instead of sending (development mode)
                logger.info(
                    f"Password reset email (not sent - email service disabled):\n"
                    f"To: {email}\n"
                    f"Subject: {subject}\n"
                    f"Reset Link: {reset_link}\n"
                    f"Token: {reset_token}"
                )
                return {
                    "success": True,
                    "message": "Password reset email logged (email service disabled)",
                    "email_logged": True,
                    "reset_link": reset_link  # Include in response for development
                }
                
        except Exception as e:
            logger.error(f"Error sending password reset email to {email}: {e}")
            return {
                "success": False,
                "message": f"Failed to send password reset email: {str(e)}",
                "email_logged": False
            }
    
    def send_welcome_email(
        self,
        email: str,
        username: str,
        full_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send welcome email to newly registered user.
        
        Args:
            email: User's email address
            username: User's username
            full_name: Optional full name for personalization
            
        Returns:
            Dictionary containing:
            - success: Boolean indicating if email was sent/logged
            - message: Status message
            - email_logged: Boolean indicating if email was logged (for development)
            
        Example:
            ```python
            result = email_service.send_welcome_email(
                email="user@example.com",
                username="john_doe",
                full_name="John Doe"
            )
            ```
        """
        try:
            # Create email content
            subject = "Welcome to i-Drill System"
            display_name = full_name if full_name else username
            
            # HTML email body
            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background-color: #0891b2; color: white; padding: 20px; text-align: center; }}
                    .content {{ background-color: #f9fafb; padding: 30px; }}
                    .footer {{ text-align: center; padding: 20px; font-size: 12px; color: #666; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>i-Drill System</h1>
                    </div>
                    <div class="content">
                        <h2>Welcome to i-Drill!</h2>
                        <p>Hello {display_name},</p>
                        <p>Thank you for registering with the i-Drill System. Your account has been successfully created.</p>
                        <p><strong>Username:</strong> {username}</p>
                        <p>You can now log in to the system and start using all the features available to you.</p>
                        <p>If you have any questions or need assistance, please don't hesitate to contact our support team.</p>
                        <p>Welcome aboard!</p>
                    </div>
                    <div class="footer">
                        <p>This is an automated message from i-Drill System. Please do not reply to this email.</p>
                        <p>&copy; {datetime.now().year} i-Drill. All rights reserved.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Plain text email body
            text_body = f"""
            Welcome to i-Drill System
            
            Hello {display_name},
            
            Thank you for registering with the i-Drill System. Your account has been successfully created.
            
            Username: {username}
            
            You can now log in to the system and start using all the features available to you.
            
            If you have any questions or need assistance, please don't hesitate to contact our support team.
            
            Welcome aboard!
            
            This is an automated message from i-Drill System. Please do not reply to this email.
            """
            
            # Send email
            if self.enabled:
                result = self._send_email(
                    to_email=email,
                    subject=subject,
                    text_body=text_body,
                    html_body=html_body
                )
                
                if result["success"]:
                    logger.info(f"Welcome email sent to {email}")
                    return {
                        "success": True,
                        "message": "Welcome email sent successfully",
                        "email_logged": False
                    }
                else:
                    logger.error(f"Failed to send welcome email to {email}: {result.get('error')}")
                    return {
                        "success": False,
                        "message": f"Failed to send email: {result.get('error', 'Unknown error')}",
                        "email_logged": False
                    }
            else:
                # Log email instead of sending (development mode)
                logger.info(
                    f"Welcome email (not sent - email service disabled):\n"
                    f"To: {email}\n"
                    f"Subject: {subject}\n"
                    f"Username: {username}"
                )
                return {
                    "success": True,
                    "message": "Welcome email logged (email service disabled)",
                    "email_logged": True
                }
                
        except Exception as e:
            logger.error(f"Error sending welcome email to {email}: {e}")
            return {
                "success": False,
                "message": f"Failed to send welcome email: {str(e)}",
                "email_logged": False
            }
    
    def _send_email_with_retry(
        self,
        to_email: str,
        subject: str,
        text_body: str,
        html_body: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send email via SMTP with retry logic.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            text_body: Plain text email body
            html_body: Optional HTML email body
            
        Returns:
            Dictionary with success status and error message if failed
        """
        import time
        
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                result = self._send_email(
                    to_email=to_email,
                    subject=subject,
                    text_body=text_body,
                    html_body=html_body
                )
                
                if result["success"]:
                    if retries > 0:
                        logger.info(f"Email sent successfully after {retries} retry(ies)")
                    return result
                else:
                    last_error = result.get("error", "Unknown error")
                    retries += 1
                    
                    if retries < self.max_retries:
                        logger.warning(
                            f"Email send failed (attempt {retries}/{self.max_retries}), "
                            f"retrying in {self.retry_delay}s: {last_error}"
                        )
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(
                            f"Email send failed after {self.max_retries} attempts: {last_error}"
                        )
                        return {
                            "success": False,
                            "error": f"Failed after {self.max_retries} attempts: {last_error}"
                        }
                        
            except Exception as e:
                last_error = str(e)
                retries += 1
                
                if retries < self.max_retries:
                    logger.warning(
                        f"Email send exception (attempt {retries}/{self.max_retries}): {e}. "
                        f"Retrying in {self.retry_delay}s"
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Email send exception after {self.max_retries} attempts: {e}")
                    return {
                        "success": False,
                        "error": f"Exception after {self.max_retries} attempts: {str(e)}"
                    }
        
        return {
            "success": False,
            "error": f"Max retries ({self.max_retries}) exceeded. Last error: {last_error}"
        }
    
    def _send_email(
        self,
        to_email: str,
        subject: str,
        text_body: str,
        html_body: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send email via SMTP (single attempt, no retry).
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            text_body: Plain text email body
            html_body: Optional HTML email body
            
        Returns:
            Dictionary with success status and error message if failed
        """
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = to_email
            
            # Add plain text part
            text_part = MIMEText(text_body, 'plain')
            msg.attach(text_part)
            
            # Add HTML part if provided
            if html_body:
                html_part = MIMEText(html_body, 'html')
                msg.attach(html_part)
            
            # Send email via SMTP
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                
                server.send_message(msg)
            
            return {"success": True, "error": None}
            
        except Exception as e:
            logger.error(f"SMTP error sending email to {to_email}: {e}")
            return {"success": False, "error": str(e)}
    
    def is_enabled(self) -> bool:
        """
        Check if email service is enabled and configured.
        
        Returns:
            True if email service is enabled, False otherwise
        """
        return self.enabled


# Global singleton instance
email_service = EmailService()

