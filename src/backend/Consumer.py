"""
Kafka Consumer with Advanced Data Quality Validation
این ماژول داده‌های سنسور را از Kafka دریافت کرده و با استفاده از سیستم
اعتبارسنجی پیشرفته پردازش می‌کند.
"""

from confluent_kafka import Consumer, KafkaException
from src.processing.dvr_controller import process_data, get_validation_report
from src.processing.dvr import insert_message
import json
from datetime import datetime
import pandas as pd
from collections import defaultdict
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Kafka configuration
KAFKA_BROKER = 'localhost:29092'
TOPIC_NAME = 'oil_rig_sensor_data'
CONSUMER_GROUP = 'oil_rig_analytics'

def create_consumer():
    """Create and return a Confluent Kafka consumer with JSON deserializer"""
    consumer_conf = {
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': CONSUMER_GROUP,
        'auto.offset.reset': 'latest',   # Start from latest messages
        'enable.auto.commit': True
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe(["oil_rig_sensor_data"])
    return consumer


def process_message(msg, enable_validation: bool = True, save_to_db: bool = True):
    """
    Process a single Kafka message with advanced data quality validation
    
    Args:
        msg: Kafka message object
        enable_validation: If True, run comprehensive data quality validation
        save_to_db: If True, save validated data to database
    
    Returns:
        Processed data dictionary or None if validation fails
    """
    try:
        data = json.loads(msg.value().decode('utf-8'))
    except Exception as e:
        logger.error(f"⚠️ Error decoding message: {e}")
        return None

    rig_id = data.get('rig_id', 'UNKNOWN')
    timestamp_str = data.get('timestamp', datetime.now().isoformat())
    
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
    except:
        timestamp = datetime.now()

    logger.info(f"\n📥 Processing data from {rig_id} at {timestamp}")

    # Maintenance flag alerts
    if data.get('maintenance_flag') == 1:
        failure_type = data.get('failure_type', 'Unknown')
        logger.warning(f"🚨 MAINTENANCE ALERT: {rig_id} has {failure_type}")

    # Apply advanced data quality validation
    if enable_validation:
        try:
            # Process data with validation and automatic corrections
            processed_data = process_data(data, use_corrections=True)
            
            if processed_data is None:
                logger.error(f"❌ Data validation FAILED for {rig_id} - record rejected")
                return None
            
            # Extract validation metadata
            validation_meta = processed_data.pop("_validation_metadata", {})
            quality_score = validation_meta.get("quality_score", 0.0)
            is_valid = validation_meta.get("is_valid", False)
            issue_count = validation_meta.get("issue_count", 0)
            
            # Log validation results
            if issue_count > 0:
                logger.warning(
                    f"⚠️ {rig_id}: Quality score={quality_score:.2f}, "
                    f"Issues={issue_count}, Valid={is_valid}"
                )
            else:
                logger.info(
                    f"✅ {rig_id}: Quality score={quality_score:.2f}, "
                    f"All checks passed"
                )
            
            # Save to database if enabled
            if save_to_db and processed_data:
                try:
                    insert_message(processed_data)
                    logger.debug(f"💾 Saved validated data for {rig_id} to database")
                except Exception as e:
                    logger.error(f"❌ Failed to save data to database: {e}")
            
            # Add back validation metadata for analysis
            processed_data["_validation_metadata"] = validation_meta
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Validation error for {rig_id}: {e}", exc_info=True)
            # Fallback: return original data without validation
            return data if not enable_validation else None
    else:
        # No validation, just return original data
        if save_to_db:
            try:
                insert_message(data)
            except Exception as e:
                logger.error(f"Failed to save data: {e}")
        return data


def aggregate_data(consumer, duration_seconds=60):
    """Aggregate data over a time window and display summary"""
    print(f"\n🚀 Starting consumer. Aggregating data in {duration_seconds}-second windows...")

    try:
        window_start = time.time()
        window_data = defaultdict(list)

        while True:
            msg = consumer.poll(timeout=1.0)  # Wait for messages
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())

            data = process_message(msg)
            if not data:
                continue

            rig_id = data['rig_id']
            window_data[rig_id].append(data)

            # Check if the time window has elapsed
            if time.time() - window_start >= duration_seconds:
                analyze_window(window_data)
                window_data = defaultdict(list)  # Reset
                window_start = time.time()

    except KeyboardInterrupt:
        print("\n🛑 Stopping consumer...")
    finally:
        consumer.close()


def analyze_window(window_data):
    """Analyze aggregated data for a time window with data quality metrics"""
    logger.info("\n" + "=" * 50)
    logger.info("📊 WINDOW SUMMARY ANALYSIS")
    logger.info("=" * 50)

    for rig_id, records in window_data.items():
        if not records:
            continue

        df = pd.DataFrame(records)
        numeric_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'rig_id', 'failure_type', '_validation_metadata']]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        logger.info(f"\n{rig_id} - {len(records)} records")
        logger.info("Average values:")
        logger.info(df[numeric_cols].mean().to_string())

        # Data quality metrics
        if '_validation_metadata' in df.columns:
            # Extract quality scores from metadata
            quality_scores = []
            invalid_count = 0
            issue_counts = []
            
            for idx, row in df.iterrows():
                meta = row.get('_validation_metadata', {})
                if isinstance(meta, dict):
                    quality_scores.append(meta.get('quality_score', 1.0))
                    if not meta.get('is_valid', True):
                        invalid_count += 1
                    issue_counts.append(meta.get('issue_count', 0))
            
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                avg_issues = sum(issue_counts) / len(issue_counts) if issue_counts else 0
                logger.info(f"\n📊 Data Quality Metrics for {rig_id}:")
                logger.info(f"   Average Quality Score: {avg_quality:.2f}")
                logger.info(f"   Invalid Records: {invalid_count}/{len(records)}")
                logger.info(f"   Average Issues per Record: {avg_issues:.2f}")

        # Maintenance events in this window
        maintenance_count = df['maintenance_flag'].sum() if 'maintenance_flag' in df.columns else 0
        if maintenance_count > 0:
            logger.warning(f"\n🔴 {maintenance_count} maintenance events detected:")
            maintenance_df = df[df['maintenance_flag'] == 1][['timestamp', 'failure_type']]
            logger.warning(maintenance_df.to_string(index=False))


if __name__ == "__main__":
    consumer = create_consumer()
    aggregate_data(consumer, duration_seconds=60)
