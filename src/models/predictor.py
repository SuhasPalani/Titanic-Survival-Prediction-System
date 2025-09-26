# src/models/predictor.py
import numpy as np
import pickle
class Predictor:
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)
    
    def filter_predictions(self, df, predictions, filters):
        """Filter predictions based on customer preferences"""
        filtered_df = df.copy()
        filtered_predictions = predictions.copy()
        
        for filter_key, filter_value in filters.items():
            if filter_key == 'pclass':
                mask = filtered_df['Pclass'].isin(filter_value) if isinstance(filter_value, list) else filtered_df['Pclass'] == filter_value
            elif filter_key == 'sex':
                mask = filtered_df['Sex'] == (0 if filter_value == 'male' else 1)
            elif filter_key == 'age_range':
                mask = (filtered_df['Age'] >= filter_value[0]) & (filtered_df['Age'] <= filter_value[1])
            elif filter_key == 'survival_prediction':
                mask = predictions == filter_value
            else:
                continue
            
            filtered_df = filtered_df[mask]
            filtered_predictions = filtered_predictions[mask]
        
        return filtered_df, filtered_predictions
    
    def sort_predictions(self, df, predictions, sort_by='survival_probability', ascending=False):
        """Sort predictions based on criteria"""
        if sort_by == 'survival_probability' and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(df)[:, 1]
            sorted_indices = np.argsort(probabilities)
            if not ascending:
                sorted_indices = sorted_indices[::-1]
        elif sort_by in df.columns:
            sorted_indices = df[sort_by].argsort()
            if not ascending:
                sorted_indices = sorted_indices[::-1]
        else:
            sorted_indices = np.arange(len(df))
        
        return df.iloc[sorted_indices], predictions[sorted_indices]