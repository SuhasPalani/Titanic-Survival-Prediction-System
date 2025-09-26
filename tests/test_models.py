# tests/test_models.py
import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.model_trainer import ModelTrainer
from src.models.predictor import Predictor

class TestModels(unittest.TestCase):
    def setUp(self):
        """Set up test data and model"""
        # Create sample training data
        np.random.seed(42)
        n_samples = 100
        
        self.X_train = pd.DataFrame({
            'Pclass': np.random.choice([1, 2, 3], n_samples),
            'Sex': np.random.choice([0, 1], n_samples),
            'Age': np.random.uniform(1, 80, n_samples),
            'SibSp': np.random.choice([0, 1, 2, 3], n_samples),
            'Parch': np.random.choice([0, 1, 2], n_samples),
            'Fare': np.random.uniform(5, 500, n_samples),
            'FamilySize': np.random.choice([1, 2, 3, 4, 5], n_samples),
            'IsAlone': np.random.choice([0, 1], n_samples),
        })
        
        self.y_train = np.random.choice([0, 1], n_samples)
        
        # Create a simple model for testing
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)
    
    def test_model_training(self):
        """Test model training functionality"""
        trainer = ModelTrainer()
        trainer.models['TestModel'] = self.model
        trainer.best_model = self.model
        
        # Test prediction
        predictions = trainer.best_model.predict(self.X_train[:5])
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_predictor(self):
        """Test predictor functionality"""
        predictor = Predictor()
        predictor.model = self.model
        
        # Test prediction
        predictions = predictor.predict(self.X_train[:5])
        probabilities = predictor.predict_proba(self.X_train[:5])
        
        self.assertEqual(len(predictions), 5)
        self.assertEqual(probabilities.shape, (5, 2))
        
    def test_filtering(self):
        """Test prediction filtering"""
        predictor = Predictor()
        predictor.model = self.model
        
        # Test filtering by class
        test_df = self.X_train.copy()
        test_df['Pclass'] = [1, 2, 3, 1, 2]  # Mix of classes
        predictions = np.array([1, 0, 1, 0, 1])
        
        filtered_df, filtered_preds = predictor.filter_predictions(
            test_df, predictions, {'pclass': 1}
        )
        
        # Should only have class 1 passengers
        self.assertTrue(all(filtered_df['Pclass'] == 1))


