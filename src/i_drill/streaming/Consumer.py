# src/i_drill/streaming/consumer.py

import json
from confluent_kafka import Consumer, KafkaException
import logging

# وارد کردن ماژول DVR از مسیر صحیح در ساختار جدید
from i_drill.processing.dvr_controller import process_data

# --- راه‌اندازی لاگینگ برای نمایش بهتر خروجی ---
# این به ما کمک می‌کند خروجی‌های consumer و dvr_controller را یکجا ببینیم
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- اصلاح تنظیمات کافکا برای کار با داکر ---
KAFKA_BROKER = 'kafka:9092'  # استفاده از نام سرویس در docker-compose
TOPIC_NAME = 'oil_rig_sensor_data' # استفاده از نام تاپیک جدید و درست
CONSUMER_GROUP = 'dvr_processing_group'

def create_consumer():
    """یک Kafka consumer جدید ایجاد و آن را برای دریافت پیام‌ها آماده می‌کند."""
    consumer_conf = {
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': CONSUMER_GROUP,
        'auto.offset.reset': 'earliest',
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe([TOPIC_NAME])
    return consumer

def consume_and_process():
    """
    به صورت پیوسته پیام‌ها را از کافکا دریافت کرده، آن‌ها را با ماژول DVR
    پردازش می‌کند و نتیجه را نمایش می‌دهد.
    """
    consumer = create_consumer()
    logger.info(f"📥 Listening to Kafka topic '{TOPIC_NAME}' for sensor data...")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())

            try:
                # دریافت و دیکود کردن پیام از کافکا
                data = json.loads(msg.value().decode('utf-8'))
                rig_id = data.get('rig_id', 'UNKNOWN_RIG')
                timestamp = data.get('timestamp', 'UNKNOWN_TIMESTAMP')
                logger.info(f"Received raw message from {rig_id} at {timestamp}")

                # >>>>>>>> بخش اصلی تغییرات: استفاده از DVR <<<<<<<<
                # پیام خام به ماژول DVR برای اعتبارسنجی و تصحیح ارسال می‌شود
                reconciled_record = process_data(data)

                # بررسی خروجی ماژول DVR
                if reconciled_record:
                    # اگر داده معتبر و تصحیح شده بود، آن را چاپ می‌کنیم
                    logger.info("✅ Record is VALID. Reconciled data:")
                    # چاپ داده‌های تمیز به صورت خوانا
                    print(json.dumps(reconciled_record, indent=2))
                else:
                    # اگر داده نامعتبر بود، dvr_controller از قبل دلیل آن را لاگ کرده است
                    # ما در اینجا فقط یک پیام کلی ثبت می‌کنیم
                    logger.warning("❌ Record is INVALID and was discarded.")

            except json.JSONDecodeError:
                logger.error("Failed to decode JSON from message.")
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")

    except KeyboardInterrupt:
        logger.info("🛑 User stopped the consumer.")
    finally:
        consumer.close()
        logger.info("Kafka consumer closed.")

if __name__ == "__main__":
    consume_and_process()