# tests/test_data_processing.py
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_cleaner import DataCleaner
from src.data.feature_engineer import FeatureEngineer

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.sample_data = pd.DataFrame({
            'PassengerId': [1, 2, 3],
            'Pclass': [1, 2, 3],
            'Name': ['Smith, Mr. John', 'Johnson, Mrs. Mary', 'Brown, Miss. Sarah'],
            'Sex': ['male', 'female', 'female'],
            'Age': [25, np.nan, 30],
            'SibSp': [0, 1, 0],
            'Parch': [0, 1, 2],
            'Ticket': ['A123', 'B456', 'C789'],
            'Fare': [50.0, np.nan, 25.0],
            'Cabin': ['C85', np.nan, 'E46'],
            'Embarked': ['S', 'C', np.nan],
            'Survived': [0, 1, 1]
        })
        
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
    
    def test_data_cleaning(self):
        """Test data cleaning functionality"""
        cleaned_data = self.cleaner.clean_data(self.sample_data)
        
        # Check that missing values are handled
        self.assertFalse(cleaned_data['Age'].isna().any())
        self.assertFalse(cleaned_data['Fare'].isna().any())
        self.assertFalse(cleaned_data['Embarked'].isna().any())
    
    def test_feature_engineering(self):
        """Test feature engineering"""
        cleaned_data = self.cleaner.clean_data(self.sample_data)
        engineered_data = self.engineer.engineer_features(cleaned_data)
        
        # Check new features exist
        self.assertIn('FamilySize', engineered_data.columns)
        self.assertIn('IsAlone', engineered_data.columns)
        self.assertIn('Title', engineered_data.columns)
        self.assertIn('AgeGroup', engineered_data.columns)
        
        # Check FamilySize calculation
        expected_family_sizes = [1, 3, 3]  # SibSp + Parch + 1
        self.assertEqual(engineered_data['FamilySize'].tolist(), expected_family_sizes)
    
    def test_title_extraction(self):
        """Test title extraction from names"""
        titles = self.sample_data['Name'].apply(self.engineer.extract_title)
        expected_titles = ['Mr', 'Mrs', 'Miss']
        self.assertEqual(titles.tolist(), expected_titles)

