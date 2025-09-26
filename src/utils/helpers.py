
# src/utils/helpers.py
import pandas as pd
import numpy as np
import pickle
import logging
import os
from datetime import datetime

def setup_logging(log_file='logs/app.log', level=logging.INFO):
    """Setup logging configuration"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_pickle(filepath):
    """Load pickle file safely"""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading pickle file {filepath}: {e}")

def save_pickle(obj, filepath):
    """Save object to pickle file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise Exception(f"Error saving pickle file {filepath}: {e}")

def validate_passenger_data(data):
    """Validate passenger data for prediction"""
    required_fields = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate data types and ranges
    if not isinstance(data['Pclass'], int) or data['Pclass'] not in [1, 2, 3]:
        raise ValueError("Pclass must be 1, 2, or 3")
    
    if data['Sex'] not in ['male', 'female']:
        raise ValueError("Sex must be 'male' or 'female'")
    
    if not isinstance(data['Age'], (int, float)) or data['Age'] < 0 or data['Age'] > 120:
        raise ValueError("Age must be between 0 and 120")
    
    if not isinstance(data['SibSp'], int) or data['SibSp'] < 0:
        raise ValueError("SibSp must be a non-negative integer")
    
    if not isinstance(data['Parch'], int) or data['Parch'] < 0:
        raise ValueError("Parch must be a non-negative integer")
    
    if not isinstance(data['Fare'], (int, float)) or data['Fare'] < 0:
        raise ValueError("Fare must be non-negative")
    
    if data['Embarked'] not in ['C', 'Q', 'S']:
        raise ValueError("Embarked must be 'C', 'Q', or 'S'")
    
    return True

def calculate_model_metrics(y_true, y_pred, y_prob=None):
    """Calculate comprehensive model metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    return metrics

def format_prediction_result(passenger_data, prediction, probability):
    """Format prediction result for display"""
    return {
        'passenger_info': {
            'class': passenger_data.get('Pclass'),
            'sex': passenger_data.get('Sex'),
            'age': passenger_data.get('Age'),
            'fare': passenger_data.get('Fare')
        },
        'prediction': {
            'survived': bool(prediction),
            'survival_probability': float(probability[1]) if len(probability) > 1 else float(probability),
            'confidence': 'High' if max(probability) > 0.8 else 'Medium' if max(probability) > 0.6 else 'Low'
        },
        'timestamp': datetime.now().isoformat()
    }