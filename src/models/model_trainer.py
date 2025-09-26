# src/models/model_trainer.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """Prepare features for training"""
        # Select relevant features
        feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                          'FamilySize', 'IsAlone'] + \
                         [col for col in df.columns if col.startswith(('Embarked_', 'Title_', 'AgeGroup_', 'FareGroup_'))]
        
        X = df[feature_columns]
        return X
    
    def train_models(self, X_train, y_train):
        """Train multiple models"""
        # Random Forest
        rf = RandomForestClassifier(random_state=42)
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        self.models['RandomForest'] = rf_grid.best_estimator_
        
        # Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr_params = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        
        lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='accuracy', n_jobs=-1)
        lr_grid.fit(X_train, y_train)
        self.models['LogisticRegression'] = lr_grid.best_estimator_
        
        # SVM
        svm = SVC(random_state=42, probability=True)
        svm_params = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
        
        svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='accuracy', n_jobs=-1)
        svm_grid.fit(X_train, y_train)
        self.models['SVM'] = svm_grid.best_estimator_
        
        # Select best model
        best_score = 0
        for name, model in self.models.items():
            score = cross_val_score(model, X_train, y_train, cv=5).mean()
            print(f"{name}: {score:.4f}")
            if score > best_score:
                best_score = score
                self.best_model = model
                self.best_model_name = name
        
        print(f"Best model: {self.best_model_name} with score: {best_score:.4f}")
        
    def save_model(self, filepath):
        """Save the best model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model, f)