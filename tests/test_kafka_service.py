"""
Unit tests for KafkaService
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
import json
import os

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src" / "backend"
sys.path.insert(0, str(src_path))

from services.kafka_service import KafkaService, KAFKA_AVAILABLE


@pytest.fixture
def mock_kafka_available():
    """Mock Kafka availability"""
    return True


@pytest.fixture
def mock_producer():
    """Mock Kafka producer"""
    producer = Mock()
    producer.produce = Mock()
    producer.poll = Mock()
    producer.flush = Mock()
    producer.list_topics = Mock(return_value=Mock())
    return producer


@pytest.fixture
def mock_consumer():
    """Mock Kafka consumer"""
    consumer = Mock()
    consumer.poll = Mock(return_value=None)
    consumer.subscribe = Mock()
    consumer.close = Mock()
    return consumer


@pytest.fixture
def kafka_service(mock_producer, mock_consumer):
    """Create KafkaService instance with mocked Kafka components"""
    with patch('services.kafka_service.KAFKA_AVAILABLE', True):
        with patch('services.kafka_service.Producer', return_value=mock_producer):
            with patch('services.kafka_service.Consumer', return_value=mock_consumer):
                with patch('services.kafka_service.config_loader') as mock_config:
                    mock_config.get_kafka_config.return_value = {
                        'bootstrap_servers': 'localhost:9092'
                    }
                    service = KafkaService()
                    service.producer = mock_producer
                    service.consumer = mock_consumer
                    return service


class TestKafkaService:
    """Test suite for KafkaService"""

    def test_init_with_kafka_available(self, mock_producer):
        """Test initialization when Kafka is available"""
        with patch('services.kafka_service.KAFKA_AVAILABLE', True):
            with patch('services.kafka_service.Producer', return_value=mock_producer):
                with patch('services.kafka_service.config_loader') as mock_config:
                    mock_config.get_kafka_config.return_value = {
                        'bootstrap_servers': 'localhost:9092'
                    }
                    service = KafkaService()
                    assert service.available is True
                    assert service.producer is not None

    def test_init_without_kafka(self):
        """Test initialization when Kafka is not available"""
        with patch('services.kafka_service.KAFKA_AVAILABLE', False):
            service = KafkaService()
            assert service.available is False
            assert service.producer is None

    def test_produce_sensor_data_success(self, kafka_service, mock_producer):
        """Test producing sensor data successfully"""
        # Setup
        topic = 'sensor-data'
        data = {'rig_id': 'RIG_01', 'timestamp': '2025-01-15T10:30:00Z', 'depth': 5000.0}
        
        # Execute
        result = kafka_service.produce_sensor_data(topic, data)
        
        # Assert
        assert result is True
        mock_producer.produce.assert_called_once()
        mock_producer.poll.assert_called_once()

    def test_produce_sensor_data_without_producer(self):
        """Test producing data when producer is not initialized"""
        with patch('services.kafka_service.KAFKA_AVAILABLE', False):
            service = KafkaService()
            result = service.produce_sensor_data('topic', {})
            assert result is False

    def test_produce_sensor_data_retry(self, kafka_service, mock_producer):
        """Test producing data with retry logic"""
        # Setup
        topic = 'sensor-data'
        data = {'rig_id': 'RIG_01', 'depth': 5000.0}
        
        # First call fails, second succeeds
        mock_producer.produce.side_effect = [Exception("Connection error"), None]
        
        # Execute
        result = kafka_service.produce_sensor_data(topic, data, max_retries=3)
        
        # Assert
        assert mock_producer.produce.call_count >= 2

    def test_create_consumer_success(self, kafka_service, mock_consumer):
        """Test creating consumer successfully"""
        # Setup
        consumer_id = 'test-consumer'
        topic = 'sensor-data'
        
        # Execute
        result = kafka_service.create_consumer(consumer_id, topic)
        
        # Assert
        assert result is True
        assert consumer_id in kafka_service.consumers
        mock_consumer.subscribe.assert_called_once_with([topic])

    def test_create_consumer_without_kafka(self):
        """Test creating consumer when Kafka is not available"""
        with patch('services.kafka_service.KAFKA_AVAILABLE', False):
            service = KafkaService()
            result = service.create_consumer('consumer1', 'topic1')
            assert result is False

    def test_consume_messages_success(self, kafka_service, mock_consumer):
        """Test consuming messages successfully"""
        # Setup
        consumer_id = 'test-consumer'
        topic = 'sensor-data'
        kafka_service.consumers[consumer_id] = mock_consumer
        
        # Mock message
        mock_msg = Mock()
        mock_msg.error.return_value = None
        mock_msg.value.return_value = json.dumps({'rig_id': 'RIG_01', 'depth': 5000.0}).encode('utf-8')
        mock_consumer.poll.return_value = mock_msg
        
        # Execute
        result = kafka_service.consume_messages(consumer_id, timeout=1.0)
        
        # Assert
        assert result is not None
        assert result['rig_id'] == 'RIG_01'

    def test_consume_messages_timeout(self, kafka_service, mock_consumer):
        """Test consuming messages with timeout"""
        # Setup
        consumer_id = 'test-consumer'
        kafka_service.consumers[consumer_id] = mock_consumer
        mock_consumer.poll.return_value = None
        
        # Execute
        result = kafka_service.consume_messages(consumer_id, timeout=1.0)
        
        # Assert
        assert result is None

    def test_consume_messages_no_consumer(self, kafka_service):
        """Test consuming messages when consumer doesn't exist"""
        # Execute
        result = kafka_service.consume_messages('non-existent-consumer')
        
        # Assert
        assert result is None

    def test_get_latest_messages(self, kafka_service, mock_consumer):
        """Test getting latest messages"""
        # Setup
        consumer_id = 'test-consumer'
        kafka_service.consumers[consumer_id] = mock_consumer
        
        # Mock messages
        mock_msg1 = Mock()
        mock_msg1.error.return_value = None
        mock_msg1.value.return_value = json.dumps({'rig_id': 'RIG_01', 'depth': 5000.0}).encode('utf-8')
        
        mock_msg2 = Mock()
        mock_msg2.error.return_value = None
        mock_msg2.value.return_value = json.dumps({'rig_id': 'RIG_01', 'depth': 5001.0}).encode('utf-8')
        
        mock_consumer.poll.side_effect = [mock_msg1, mock_msg2, None]
        
        # Execute
        result = kafka_service.get_latest_messages(consumer_id, count=2)
        
        # Assert
        assert len(result) == 2
        assert result[0]['depth'] == 5000.0
        assert result[1]['depth'] == 5001.0

    def test_close_consumer(self, kafka_service, mock_consumer):
        """Test closing consumer"""
        # Setup
        consumer_id = 'test-consumer'
        kafka_service.consumers[consumer_id] = mock_consumer
        
        # Execute
        kafka_service.close_consumer(consumer_id)
        
        # Assert
        assert consumer_id not in kafka_service.consumers
        mock_consumer.close.assert_called_once()

    def test_close_all(self, kafka_service, mock_producer, mock_consumer):
        """Test closing all consumers and producer"""
        # Setup
        consumer_id = 'test-consumer'
        kafka_service.consumers[consumer_id] = mock_consumer
        
        # Execute
        kafka_service.close()
        
        # Assert
        assert len(kafka_service.consumers) == 0
        mock_producer.flush.assert_called_once()

    def test_check_connection_success(self, kafka_service, mock_producer):
        """Test checking connection successfully"""
        # Setup
        mock_producer.list_topics.return_value = Mock()
        
        # Execute
        result = kafka_service.check_connection()
        
        # Assert
        assert result is True
        mock_producer.list_topics.assert_called_once()

    def test_check_connection_failure(self, kafka_service, mock_producer):
        """Test checking connection when it fails"""
        # Setup
        mock_producer.list_topics.side_effect = Exception("Connection failed")
        
        # Execute
        result = kafka_service.check_connection()
        
        # Assert
        assert result is False

    def test_check_connection_no_producer(self):
        """Test checking connection when producer is not initialized"""
        with patch('services.kafka_service.KAFKA_AVAILABLE', False):
            service = KafkaService()
            result = service.check_connection()
            assert result is False

    def test_is_available_true(self, kafka_service, mock_producer):
        """Test is_available returns True when service is available"""
        kafka_service.available = True
        kafka_service.producer = mock_producer
        assert kafka_service.is_available() is True

    def test_is_available_false(self, kafka_service):
        """Test is_available returns False when service is not available"""
        kafka_service.available = False
        kafka_service.producer = None
        assert kafka_service.is_available() is False

    def test_delivery_callback_success(self, kafka_service):
        """Test delivery callback on success"""
        # Setup
        mock_msg = Mock()
        mock_msg.topic.return_value = 'sensor-data'
        mock_msg.partition.return_value = 0
        mock_msg.offset.return_value = 100
        
        # Execute
        kafka_service._delivery_callback(None, mock_msg)
        
        # Assert - should not raise exception

    def test_delivery_callback_error(self, kafka_service):
        """Test delivery callback on error"""
        # Setup
        error = Mock()
        error.__str__ = Mock(return_value="Delivery failed")
        
        # Execute
        kafka_service._delivery_callback(error, None)
        
        # Assert - should not raise exception

    def test_initialize_producer_with_retry(self, mock_producer):
        """Test initializing producer with retry logic"""
        with patch('services.kafka_service.KAFKA_AVAILABLE', True):
            with patch('services.kafka_service.Producer') as mock_producer_class:
                # First call fails, second succeeds
                mock_producer_class.side_effect = [Exception("Connection error"), mock_producer]
                
                with patch('services.kafka_service.config_loader') as mock_config:
                    mock_config.get_kafka_config.return_value = {
                        'bootstrap_servers': 'localhost:9092'
                    }
                    with patch('time.sleep'):  # Mock sleep to speed up test
                        service = KafkaService()
                        # After retry, producer should be initialized
                        assert service.producer is not None

    def test_produce_with_authentication(self, mock_producer):
        """Test producing data with Kafka authentication"""
        with patch('services.kafka_service.KAFKA_AVAILABLE', True):
            with patch('services.kafka_service.Producer', return_value=mock_producer):
                with patch('services.kafka_service.config_loader') as mock_config:
                    with patch.dict(os.environ, {
                        'KAFKA_USERNAME': 'test_user',
                        'KAFKA_PASSWORD': 'test_password'
                    }):
                        mock_config.get_kafka_config.return_value = {
                            'bootstrap_servers': 'localhost:9092'
                        }
                        service = KafkaService()
                        # Verify producer was created with auth config
                        assert service.producer is not None

