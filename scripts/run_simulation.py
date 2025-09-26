# api/app.py
import sys
import os

# Get the path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the project root to the Python path
sys.path.insert(0, project_root) 
import time
import threading
from src.kafka.producer import TitanicDataProducer
from src.kafka.consumer import TitanicDataConsumer
from src.kafka.stream_processor import StreamProcessor

def run_producer(duration=60):
    """Run the Kafka producer"""
    producer = TitanicDataProducer()
    try:
        print("Starting producer...")
        producer.start_streaming(interval=2, duration=duration)
    except KeyboardInterrupt:
        print("Producer stopped")
    finally:
        producer.close()

def run_consumer():
    """Run the Kafka consumer with stream processor"""
    processor = StreamProcessor('models/saved_models/titanic_model.pkl')
    consumer = TitanicDataConsumer()
    
    def process_message(passenger_data, passenger_id):
        result = processor.process_passenger_data(passenger_data, passenger_id)
        
        # Print statistics every 10 processed passengers
        if processor.processed_count % 10 == 0 and processor.processed_count > 0:
            stats = processor.get_statistics()
            print(f"\n--- STATISTICS ---")
            print(f"Processed: {stats['processed_count']} passengers")
            print(f"Survival Rate: {stats['survival_rate']:.3f}")
            print(f"Avg Survival Probability: {stats['average_survival_probability']:.3f}")
            print("------------------\n")
    
    try:
        print("Starting consumer and stream processor...")
        consumer.consume_messages(callback_function=process_message)
    except KeyboardInterrupt:
        print("Consumer stopped")

def main():
    """Run the real-time simulation"""
    print("=== TITANIC REAL-TIME PREDICTION SIMULATION ===")
    print("Starting Kafka simulation...")
    print("Note: Make sure Kafka is running (docker-compose up -d)")
    
    # Start consumer in a separate thread
    consumer_thread = threading.Thread(target=run_consumer)
    consumer_thread.daemon = True
    consumer_thread.start()
    
    # Wait a moment for consumer to start
    time.sleep(2)
    
    # Start producer
    run_producer(duration=120)  # Run for 2 minutes
    
    print("Simulation completed!")

if __name__ == "__main__":
    main()