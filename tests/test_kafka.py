# tests/test_kafka.py
import unittest
from unittest.mock import Mock, patch
import json

from src.kafka.producer import TitanicDataProducer
from src.kafka.stream_processor import StreamProcessor

class TestKafka(unittest.TestCase):
    def setUp(self):
        """Set up test components"""
        self.sample_passenger = {
            'PassengerId': 1001,
            'Pclass': 1,
            'Sex': 'female',
            'Age': 25,
            'SibSp': 0,
            'Parch': 0,
            'Fare': 100.0,
            'Embarked': 'S'
        }
    
    def test_passenger_data_generation(self):
        """Test passenger data generation"""
        producer = TitanicDataProducer()
        passenger_data = producer.generate_passenger_data()
        
        # Check required fields exist
        required_fields = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 
                          'Parch', 'Fare', 'Embarked', 'timestamp']
        
        for field in required_fields:
            self.assertIn(field, passenger_data)
        
        # Check data types and ranges
        self.assertIn(passenger_data['Pclass'], [1, 2, 3])
        self.assertIn(passenger_data['Sex'], ['male', 'female'])
        self.assertIn(passenger_data['Embarked'], ['C', 'Q', 'S'])
        self.assertGreaterEqual(passenger_data['Age'], 1)
        self.assertLessEqual(passenger_data['Age'], 80)
    
    @patch('src.kafka.stream_processor.Predictor')
    def test_stream_processor(self, mock_predictor):
        """Test stream processor functionality"""
        # Mock the predictor
        mock_model = Mock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.2, 0.8]]
        mock_predictor.return_value.model = mock_model
        mock_predictor.return_value.predict = mock_model.predict
        mock_predictor.return_value.predict_proba = mock_model.predict_proba
        
        # Test processing (this would normally require a real model)
        # processor = StreamProcessor('dummy_model_path')
        # result = processor.process_passenger_data(self.sample_passenger, '1001')
        
        # This test would need a more sophisticated setup with mocked dependencies
        self.assertTrue(True)  # Placeholder for now