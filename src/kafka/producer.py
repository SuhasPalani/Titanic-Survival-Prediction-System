# src/kafka/producer.py
import json
import time
import random
import pandas as pd
from kafka import KafkaProducer
from datetime import datetime

class TitanicDataProducer:
    def __init__(self, bootstrap_servers='localhost:9092', topic='titanic-predictions'):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            key_serializer=lambda x: x.encode('utf-8') if x else None
        )
        self.topic = topic
    
    def generate_passenger_data(self):
        """Generate random passenger data"""
        passenger_data = {
            'PassengerId': random.randint(1000, 9999),
            'Pclass': random.choice([1, 2, 3]),
            'Sex': random.choice(['male', 'female']),
            'Age': random.randint(1, 80),
            'SibSp': random.randint(0, 5),
            'Parch': random.randint(0, 5),
            'Fare': round(random.uniform(5, 500), 2),
            'Embarked': random.choice(['C', 'Q', 'S']),
            'timestamp': datetime.now().isoformat()
        }
        return passenger_data
    
    def send_passenger_data(self, passenger_data):
        """Send passenger data to Kafka topic"""
        try:
            key = str(passenger_data['PassengerId'])
            self.producer.send(self.topic, key=key, value=passenger_data)
            self.producer.flush()
            print(f"Sent data for passenger {passenger_data['PassengerId']}")
        except Exception as e:
            print(f"Error sending data: {e}")
    
    def start_streaming(self, interval=1, duration=60):
        """Start streaming passenger data"""
        start_time = time.time()
        while time.time() - start_time < duration:
            passenger_data = self.generate_passenger_data()
            self.send_passenger_data(passenger_data)
            time.sleep(interval)
    
    def close(self):
        """Close producer"""
        self.producer.close()

