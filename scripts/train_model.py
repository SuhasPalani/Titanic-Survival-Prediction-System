import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.data.feature_engineer import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.visualization.plots import TitanicVisualizer

def main():
    print("=== TITANIC SURVIVAL PREDICTION MODEL TRAINING ===\n")
    
    # Load data
    print("Loading data...")
    loader = DataLoader()
    train_data = loader.load_train_data()
    print(f"Loaded {len(train_data)} training samples")
    
    # Initialize processors
    cleaner = DataCleaner()
    engineer = FeatureEngineer()
    visualizer = TitanicVisualizer()
    
    # Clean data
    print("Cleaning data...")
    train_cleaned = cleaner.clean_data(train_data)
    
    # Exploratory Data Analysis
    print("Performing EDA...")
    visualizer.comprehensive_eda(train_cleaned)
    
    # Feature engineering
    print("Engineering features...")
    train_engineered = engineer.engineer_features(train_cleaned)
    train_encoded = engineer.encode_categorical(train_engineered)
    
    # Prepare features and target
    feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                      'FamilySize', 'IsAlone'] + \
                     [col for col in train_encoded.columns if col.startswith(('Embarked_', 'Title_', 'AgeGroup_', 'FareGroup_'))]
    
    X = train_encoded[feature_columns]
    y = train_encoded['Survived']
    
    print(f"Features used: {len(feature_columns)}")
    print(f"Feature names: {feature_columns}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train models
    print("\nTraining models...")
    trainer = ModelTrainer()
    trainer.train_models(X_train, y_train)
    
    # Evaluate best model
    print("\nEvaluating best model...")
    evaluator = ModelEvaluator()
    evaluation_results = evaluator.evaluate_model(trainer.best_model, X_test, y_test)
    
    # Save model and preprocessing pipeline
    print("Saving model...")
    os.makedirs('models/saved_models', exist_ok=True)
    
    # Save the trained model
    model_path = 'models/saved_models/titanic_model.pkl'
    trainer.save_model(model_path)
    
    # Save preprocessing components
    preprocessing_pipeline = {
        'cleaner': cleaner,
        'engineer': engineer,
        'feature_columns': feature_columns
    }
    
    with open('models/saved_models/preprocessing_pipeline.pkl', 'wb') as f:
        pickle.dump(preprocessing_pipeline, f)
    
    print(f"Model saved to: {model_path}")
    print(f"Preprocessing pipeline saved")
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    train_encoded.to_csv('data/processed/train_processed.csv', index=False)
    print("Processed training data saved")
    
    print("\n=== TRAINING COMPLETED ===")
    return trainer.best_model, evaluation_results

if __name__ == "__main__":
    main()
