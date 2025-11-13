# تحلیل TODO‌ها و راه‌حل‌های پیشنهادی

## 📊 خلاصه TODO‌ها

**مجموع TODO‌ها: 4 مورد**

### 1. TODO‌های مربوط به سیستم کنترل (3 مورد)

#### ✅ TODO 1: Integration با سیستم کنترل در `apply_parameter_change`
- **فایل**: `src/backend/services/control_service.py`
- **خط**: 74
- **توضیحات**: در production نیاز به integration با سیستم کنترل واقعی (PLC/SCADA/MQTT/Modbus)

#### ✅ TODO 2: Query مقدار فعلی از سیستم کنترل
- **فایل**: `src/backend/services/control_service.py`
- **خط**: 244
- **توضیحات**: Query مقدار فعلی parameter از سیستم کنترل واقعی

#### ✅ TODO 3: Integration در `apply_change` endpoint
- **فایل**: `src/backend/api/routes/control.py`
- **خط**: 185
- **توضیحات**: Integration با سیستم کنترل در متد apply_change

#### ✅ TODO 4: Integration در `approve_change` endpoint
- **فایل**: `src/backend/api/routes/control.py`
- **خط**: 362
- **توضیحات**: Integration با سیستم کنترل در متد approve_change

### 2. TODO مربوط به Email Service (1 مورد)

#### ✅ TODO 5: Email Service Integration
- **فایل**: `src/backend/api/routes/auth.py`
- **توضیحات**: بهبود و integration کامل email service برای ارسال ایمیل‌های password reset

---

## 🔧 راه‌حل‌های پیشنهادی

### 1. سیستم کنترل (Control System Integration)

#### راه‌حل 1: REST API Integration

```python
# در control_service.py
import httpx
from config import settings

CONTROL_SYSTEM_URL = settings.CONTROL_SYSTEM_URL
CONTROL_SYSTEM_TOKEN = settings.CONTROL_SYSTEM_TOKEN

def apply_parameter_change(self, rig_id, component, parameter, new_value, metadata):
    """Apply parameter change via REST API"""
    try:
        response = httpx.post(
            f"{CONTROL_SYSTEM_URL}/rigs/{rig_id}/parameters",
            json={
                "component": component,
                "parameter": parameter,
                "value": new_value,
                "metadata": metadata
            },
            headers={"Authorization": f"Bearer {CONTROL_SYSTEM_TOKEN}"},
            timeout=10.0
        )
        
        if response.status_code != 200:
            raise Exception(f"Control system error: {response.text}")
        
        result = response.json()
        return {
            "success": True,
            "message": f"Parameter {parameter} changed to {new_value} successfully",
            "applied_at": datetime.now().isoformat(),
            "control_system_response": result
        }
    except Exception as e:
        logger.error(f"Error applying parameter change: {e}")
        return {
            "success": False,
            "message": f"Failed to apply parameter change: {str(e)}",
            "applied_at": None,
            "error": str(e)
        }

def get_parameter_value(self, rig_id, component, parameter):
    """Get current parameter value from control system"""
    try:
        response = httpx.get(
            f"{CONTROL_SYSTEM_URL}/rigs/{rig_id}/parameters/{component}/{parameter}",
            headers={"Authorization": f"Bearer {CONTROL_SYSTEM_TOKEN}"},
            timeout=5.0
        )
        
        if response.status_code == 200:
            return response.json().get("value")
        return None
    except Exception as e:
        logger.warning(f"Error getting parameter value: {e}")
        return None
```

#### راه‌حل 2: MQTT Integration

```python
import paho.mqtt.client as mqtt
import json

class ControlService:
    def __init__(self):
        self.mqtt_client = None
        if settings.MQTT_ENABLED:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.connect(settings.MQTT_HOST, settings.MQTT_PORT)
    
    def apply_parameter_change(self, rig_id, component, parameter, new_value, metadata):
        """Apply parameter change via MQTT"""
        try:
            topic = f"rigs/{rig_id}/control/{component}/{parameter}"
            payload = json.dumps({
                "value": new_value,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            })
            
            result = self.mqtt_client.publish(topic, payload, qos=1)
            result.wait_for_publish()
            
            return {
                "success": True,
                "message": f"Parameter change published to MQTT",
                "applied_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error publishing to MQTT: {e}")
            return {
                "success": False,
                "message": f"Failed to publish: {str(e)}",
                "error": str(e)
            }
```

#### راه‌حل 3: Modbus Integration

```python
from pymodbus.client.sync import ModbusTcpClient

class ControlService:
    def __init__(self):
        self.modbus_clients = {}  # Cache clients per rig
    
    def _get_modbus_client(self, rig_id):
        """Get or create Modbus client for rig"""
        if rig_id not in self.modbus_clients:
            address = settings.RIG_MODBUS_ADDRESSES.get(rig_id)
            if address:
                self.modbus_clients[rig_id] = ModbusTcpClient(address['host'], address['port'])
        return self.modbus_clients.get(rig_id)
    
    def apply_parameter_change(self, rig_id, component, parameter, new_value, metadata):
        """Apply parameter change via Modbus"""
        try:
            client = self._get_modbus_client(rig_id)
            if not client:
                raise Exception(f"No Modbus client configured for rig {rig_id}")
            
            # Map parameter to Modbus register
            register_map = settings.PARAMETER_REGISTER_MAP
            register = register_map.get(parameter)
            
            if not register:
                raise Exception(f"No register mapping for parameter {parameter}")
            
            # Write value to register
            result = client.write_register(
                address=register,
                value=int(float(new_value) * 100)  # Scale as needed
            )
            
            if result.isError():
                raise Exception(f"Modbus write error: {result}")
            
            return {
                "success": True,
                "message": f"Parameter {parameter} written to Modbus register {register}",
                "applied_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error writing to Modbus: {e}")
            return {
                "success": False,
                "message": f"Failed to write to Modbus: {str(e)}",
                "error": str(e)
            }
```

---

### 2. Email Service Integration

#### بهبودهای پیشنهادی:

1. **افزودن Email Templates**: استفاده از templates برای ایمیل‌ها
2. **افزودن Email Queue**: استفاده از queue برای ارسال ایمیل‌های bulk
3. **افزودن Retry Logic**: Retry برای ارسال ایمیل‌های ناموفق
4. **افزودن Email Tracking**: Tracking باز شدن و کلیک ایمیل‌ها

```python
# بهبود email_service.py
from typing import Optional, Dict, Any, List
import asyncio
from datetime import datetime, timedelta
import logging

class EmailService:
    """Enhanced Email Service with queue and retry support"""
    
    def __init__(self):
        self.enabled = SMTP_ENABLED and EMAIL_LIBS_AVAILABLE
        self.email_queue = asyncio.Queue()
        self.max_retries = 3
        self.retry_delay = 60  # seconds
    
    async def send_email_async(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """Send email asynchronously with retry logic"""
        retries = 0
        
        while retries < self.max_retries:
            try:
                result = self._send_email(
                    to_email=to_email,
                    subject=subject,
                    text_body=text_body or self._html_to_text(html_body),
                    html_body=html_body
                )
                
                if result["success"]:
                    logger.info(f"Email sent successfully to {to_email}")
                    return result
                else:
                    retries += 1
                    if retries < self.max_retries:
                        logger.warning(
                            f"Email send failed (attempt {retries}/{self.max_retries}), "
                            f"retrying in {self.retry_delay}s: {result.get('error')}"
                        )
                        await asyncio.sleep(self.retry_delay)
                    else:
                        logger.error(f"Email send failed after {self.max_retries} attempts")
                        return result
                        
            except Exception as e:
                retries += 1
                if retries < self.max_retries:
                    logger.warning(f"Email send exception (attempt {retries}/{self.max_retries}): {e}")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"Email send exception after {self.max_retries} attempts: {e}")
                    return {
                        "success": False,
                        "error": str(e)
                    }
        
        return {
            "success": False,
            "error": "Max retries exceeded"
        }
    
    def send_password_reset_email(
        self,
        email: str,
        reset_token: str,
        username: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send password reset email with improved template"""
        try:
            reset_link = f"{self.frontend_url}/auth/password/reset/confirm?token={reset_token}"
            
            # Use template for email
            html_body = self._get_password_reset_template(
                username=username,
                reset_link=reset_link,
                expiry_hours=24
            )
            
            # Send email (sync version for compatibility)
            if self.enabled:
                result = self._send_email(
                    to_email=email,
                    subject="Password Reset Request - i-Drill",
                    text_body=self._html_to_text(html_body),
                    html_body=html_body
                )
                
                return result
            else:
                # Log in development
                logger.info(f"Password reset email (not sent): {email} - Link: {reset_link}")
                return {
                    "success": True,
                    "email_logged": True,
                    "reset_link": reset_link
                }
                
        except Exception as e:
            logger.error(f"Error sending password reset email: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_password_reset_template(
        self,
        username: Optional[str],
        reset_link: str,
        expiry_hours: int
    ) -> str:
        """Get password reset email template"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #0891b2; color: white; padding: 20px; text-align: center; }}
                .content {{ background-color: #f9fafb; padding: 30px; }}
                .button {{ display: inline-block; padding: 12px 24px; background-color: #0891b2; 
                          color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
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
                    <p>Hello{' ' + username if username else 'User'},</p>
                    <p>You have requested to reset your password for your i-Drill account.</p>
                    <p>Click the button below to reset your password:</p>
                    <p style="text-align: center;">
                        <a href="{reset_link}" class="button">Reset Password</a>
                    </p>
                    <p>Or copy and paste this link into your browser:</p>
                    <p style="word-break: break-all; color: #0891b2;">{reset_link}</p>
                    <p class="warning">This link will expire in {expiry_hours} hours.</p>
                    <p>If you did not request this password reset, please ignore this email.</p>
                </div>
                <div class="footer">
                    <p>&copy; {datetime.now().year} i-Drill. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text"""
        # Simple HTML to text conversion
        import re
        text = re.sub('<[^<]+?>', '', html)
        text = text.replace('&nbsp;', ' ')
        return text.strip()
```

---

## 📋 مراحل پیاده‌سازی

### فاز 1: کنترل سیستم

1. **انتخاب روش Integration**:
   - REST API (ساده‌ترین)
   - MQTT (برای real-time)
   - Modbus (برای PLC)
   
2. **تنظیم Environment Variables**:
   ```env
   CONTROL_SYSTEM_TYPE=REST|MQTT|MODBUS
   CONTROL_SYSTEM_URL=http://control-system:8080/api
   CONTROL_SYSTEM_TOKEN=your-token
   ```

3. **به‌روزرسانی `control_service.py`**:
   - پیاده‌سازی integration methods
   - افزودن error handling
   - افزودن logging

4. **تست Integration**:
   - Unit tests
   - Integration tests
   - End-to-end tests

### فاز 2: Email Service

1. **بهبود Email Templates**:
   - Template برای password reset
   - Template برای notifications
   - Template برای alerts

2. **افزودن Retry Logic**:
   - Retry mechanism
   - Exponential backoff
   - Error logging

3. **افزودن Email Queue** (اختیاری):
   - استفاده از Celery یا background task
   - Queue برای bulk emails
   - Priority queue

4. **تست Email Service**:
   - Unit tests
   - Integration tests با SMTP server mock

---

## 🔧 تنظیمات Environment Variables

```env
# Control System Integration
CONTROL_SYSTEM_TYPE=REST
CONTROL_SYSTEM_URL=http://localhost:8080/api/v1
CONTROL_SYSTEM_TOKEN=your-api-token
CONTROL_SYSTEM_TIMEOUT=10

# Email Service
SMTP_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-password
SMTP_FROM_EMAIL=noreply@i-drill.local
SMTP_FROM_NAME=i-Drill System
SMTP_USE_TLS=true
FRONTEND_URL=http://localhost:3001
```

---

## ✅ Checklist پیاده‌سازی

### Control System Integration
- [ ] انتخاب روش integration (REST/MQTT/Modbus)
- [ ] افزودن environment variables
- [ ] پیاده‌سازی `apply_parameter_change` در control_service.py
- [ ] پیاده‌سازی `get_parameter_value` در control_service.py
- [ ] به‌روزرسانی control.py endpoints
- [ ] افزودن error handling
- [ ] افزودن logging
- [ ] نوشتن unit tests
- [ ] نوشتن integration tests

### Email Service
- [ ] بهبود email templates
- [ ] افزودن retry logic
- [ ] بهبود error handling
- [ ] افزودن async support (اختیاری)
- [ ] تست با SMTP server
- [ ] تست email templates
- [ ] نوشتن unit tests

---

## 📚 منابع و مستندات

- [FastAPI Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)
- [Python SMTP Documentation](https://docs.python.org/3/library/smtplib.html)
- [MQTT Python Client](https://pypi.org/project/paho-mqtt/)
- [Modbus Python Library](https://pypi.org/project/pymodbus/)
- [Email Templates Best Practices](https://www.campaignmonitor.com/dev-resources/guides/coding/)

---

## 💡 نکات مهم

1. **Security**: 
   - استفاده از tokens/API keys
   - Encrypt sensitive data
   - Rate limiting برای API calls

2. **Error Handling**:
   - Graceful degradation
   - Retry logic
   - Proper logging

3. **Performance**:
   - Connection pooling
   - Async operations
   - Caching

4. **Testing**:
   - Mock external services
   - Integration tests
   - End-to-end tests

---

**آیا می‌خواهید که این TODO‌ها را پیاده‌سازی کنم؟**

