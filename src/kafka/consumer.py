# src/kafka/consumer.py
from kafka import KafkaConsumer
import json

class TitanicDataConsumer:
    def __init__(self, bootstrap_servers='localhost:9092', topic='titanic-predictions'):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            key_deserializer=lambda x: x.decode('utf-8') if x else None,
            group_id='titanic-prediction-group',
            auto_offset_reset='latest'
        )
    
    def consume_messages(self, callback_function=None):
        """Consume messages and process them"""
        try:
            for message in self.consumer:
                passenger_data = message.value
                passenger_id = message.key
                
                print(f"Received data for passenger {passenger_id}: {passenger_data}")
                
                if callback_function:
                    callback_function(passenger_data, passenger_id)
                    
        except KeyboardInterrupt:
            print("Stopping consumer...")
        finally:
            self.consumer.close()

