# src/kafka/stream_processor.py
import pandas as pd
import numpy as np
from src.models.predictor import Predictor
from src.data.data_cleaner import DataCleaner
from src.data.feature_engineer import FeatureEngineer

class StreamProcessor:
    def __init__(self, model_path):
        self.predictor = Predictor(model_path)
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.processed_count = 0
        self.predictions = []
    
    def process_passenger_data(self, passenger_data, passenger_id):
        """Process incoming passenger data and make prediction"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([passenger_data])
            
            # Clean data
            df_cleaned = self.cleaner.clean_data(df)
            
            # Engineer features
            df_engineered = self.engineer.engineer_features(df_cleaned)
            df_encoded = self.engineer.encode_categorical(df_engineered)
            
            # Prepare features - ensure all expected columns are present
            base_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
            
            # All possible one-hot encoded columns that could exist
            all_possible_features = base_features + [
                'Embarked_C', 'Embarked_Q', 'Embarked_S',
                'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare',
                'AgeGroup_Child', 'AgeGroup_Teen', 'AgeGroup_Adult', 'AgeGroup_Middle', 'AgeGroup_Senior',
                'FareGroup_Low', 'FareGroup_Medium', 'FareGroup_High', 'FareGroup_Very High'
            ]
            
            # Create a DataFrame with all possible columns, initialized to 0
            feature_df = pd.DataFrame(0, index=[0], columns=all_possible_features)
            
            # Fill in the values we have from df_encoded
            for col in df_encoded.columns:
                if col in all_possible_features:
                    feature_df[col] = df_encoded[col].iloc[0]
            
            X = feature_df
            
            # Make prediction
            prediction = self.predictor.predict(X)[0]
            probability = self.predictor.predict_proba(X)[0]
            
            # Store result
            result = {
                'passenger_id': passenger_id,
                'prediction': int(prediction),
                'survival_probability': float(probability[1]),
                'passenger_data': passenger_data,
                'processed_at': pd.Timestamp.now().isoformat()
            }
            
            self.predictions.append(result)
            self.processed_count += 1
            
            # Print result
            survival_status = "SURVIVED" if prediction == 1 else "DID NOT SURVIVE"
            print(f"Passenger {passenger_id}: {survival_status} (Probability: {probability[1]:.3f})")
            
            return result
            
        except Exception as e:
            print(f"Error processing passenger {passenger_id}: {e}")
            return None
    
    def get_statistics(self):
        """Get processing statistics"""
        if not self.predictions:
            return {
                "processed_count": 0, 
                "survival_rate": 0.0,
                "average_survival_probability": 0.0,
                "predictions": []
            }
        
        predictions_df = pd.DataFrame(self.predictions)
        stats = {
            'processed_count': self.processed_count,
            'survival_rate': predictions_df['prediction'].mean(),
            'average_survival_probability': predictions_df['survival_probability'].mean(),
            'last_processed': predictions_df['processed_at'].iloc[-1] if len(predictions_df) > 0 else None
        }
        return stats