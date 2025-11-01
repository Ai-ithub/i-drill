#!/usr/bin/env python3
"""
Main entry point for the drilling data processing system.
Runs both Producer and Consumer in separate threads.
"""

import threading
import time
import signal
import sys
import logging
from config_loader import ConfigLoader

# Import the main functions from Producer and Consumer
from Producer import main as producer_main
from Consumer import main as consumer_main

# Global flag for graceful shutdown
shutdown_flag = threading.Event()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logging.info("Received shutdown signal. Stopping all processes...")
    shutdown_flag.set()

def run_producer():
    """Run the producer in a separate thread"""
    try:
        logging.info("Starting Producer thread...")
        producer_main()
    except Exception as e:
        logging.error(f"Producer thread error: {e}")
        shutdown_flag.set()

def run_consumer():
    """Run the consumer in a separate thread"""
    try:
        logging.info("Starting Consumer thread...")
        consumer_main()
    except Exception as e:
        logging.error(f"Consumer thread error: {e}")
        shutdown_flag.set()

def main():
    """Main function to orchestrate the drilling data system"""
    # Setup logging
    config_loader = ConfigLoader()
    logging.basicConfig(
        level=getattr(logging, config_loader.get_logging_config()['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Drilling Data Processing System...")
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start threads
    producer_thread = threading.Thread(target=run_producer, name="Producer")
    consumer_thread = threading.Thread(target=run_consumer, name="Consumer")
    
    producer_thread.daemon = True
    consumer_thread.daemon = True
    
    try:
        # Start both threads
        producer_thread.start()
        time.sleep(2)  # Give producer a head start
        consumer_thread.start()
        
        logger.info("Both Producer and Consumer threads started successfully")
        
        # Wait for shutdown signal or thread completion
        while not shutdown_flag.is_set():
            if not producer_thread.is_alive() or not consumer_thread.is_alive():
                logger.warning("One of the threads has stopped unexpectedly")
                shutdown_flag.set()
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        shutdown_flag.set()
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        shutdown_flag.set()
    finally:
        logger.info("Shutting down system...")
        
        # Wait for threads to finish (with timeout)
        if producer_thread.is_alive():
            producer_thread.join(timeout=5)
        if consumer_thread.is_alive():
            consumer_thread.join(timeout=5)
            
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()