# src/data/feature_engineer.py
import re
import pandas as pd
import numpy as np


# src/data/feature_engineer.py
import re


class FeatureEngineer:
    def __init__(self):
        pass
    
    def extract_title(self, name):
        """Extract title from name"""
        title = re.search(' ([A-Za-z]+)\.', name)
        if title:
            return title.group(1)
        return 'Unknown'
    
    def engineer_features(self, df):
        """Create new features"""
        df_engineered = df.copy()
        
        # Create FamilySize feature
        df_engineered['FamilySize'] = df_engineered['SibSp'] + df_engineered['Parch'] + 1
        
        # Create IsAlone feature
        df_engineered['IsAlone'] = (df_engineered['FamilySize'] == 1).astype(int)
        
        # Extract titles from names (handle missing names)
        if 'Name' in df_engineered.columns:
            df_engineered['Title'] = df_engineered['Name'].apply(self.extract_title)
        else:
            # If no Name column, infer title from Sex and Age
            def infer_title(row):
                if row['Sex'] == 'male':
                    return 'Master' if row['Age'] < 16 else 'Mr'
                else:
                    return 'Miss' if row['Age'] < 35 else 'Mrs'
            
            df_engineered['Title'] = df_engineered.apply(infer_title, axis=1)
        
        # Group rare titles
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
            'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
            'Capt': 'Rare', 'Sir': 'Rare'
        }
        df_engineered['Title'] = df_engineered['Title'].map(title_mapping)
        df_engineered['Title'].fillna('Rare', inplace=True)
        
        # Create age groups
        df_engineered['AgeGroup'] = pd.cut(df_engineered['Age'], 
                                         bins=[0, 12, 18, 35, 60, 100], 
                                         labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        
        # Create fare groups (handle single data point case)
        try:
            if len(df_engineered) > 1:
                df_engineered['FareGroup'] = pd.qcut(df_engineered['Fare'], 
                                               q=4, 
                                               labels=['Low', 'Medium', 'High', 'Very High'],
                                               duplicates='drop')
            else:
                # For single predictions, assign based on fare value ranges
                fare_value = df_engineered['Fare'].iloc[0]
                if fare_value <= 7.91:
                    fare_group = 'Low'
                elif fare_value <= 14.45:
                    fare_group = 'Medium'
                elif fare_value <= 31.0:
                    fare_group = 'High'
                else:
                    fare_group = 'Very High'
                df_engineered['FareGroup'] = fare_group
        except Exception:
            # Fallback: assign based on fare value
            fare_value = df_engineered['Fare'].iloc[0]
            if fare_value <= 10:
                fare_group = 'Low'
            elif fare_value <= 25:
                fare_group = 'Medium'
            elif fare_value <= 50:
                fare_group = 'High'
            else:
                fare_group = 'Very High'
            df_engineered['FareGroup'] = fare_group
        
        return df_engineered
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        # Label encoding for binary variables
        df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})
        
        # One-hot encoding for multi-class variables
        if 'Embarked' in df_encoded.columns:
            embarked_dummies = pd.get_dummies(df_encoded['Embarked'], prefix='Embarked')
            df_encoded = pd.concat([df_encoded, embarked_dummies], axis=1)
        
        title_dummies = pd.get_dummies(df_encoded['Title'], prefix='Title')
        df_encoded = pd.concat([df_encoded, title_dummies], axis=1)
        
        age_group_dummies = pd.get_dummies(df_encoded['AgeGroup'], prefix='AgeGroup')
        df_encoded = pd.concat([df_encoded, age_group_dummies], axis=1)
        
        fare_group_dummies = pd.get_dummies(df_encoded['FareGroup'], prefix='FareGroup')
        df_encoded = pd.concat([df_encoded, fare_group_dummies], axis=1)
        
        return df_encoded
