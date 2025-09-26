# src/data/data_cleaner.py
import pandas as pd

class DataCleaner:
    def __init__(self):
        pass
    
    def clean_data(self, df):
        """Clean the dataset"""
        df_cleaned = df.copy()
        
        # Handle missing values in Age - fill with median by Pclass and Sex
        age_median = df_cleaned.groupby(['Pclass', 'Sex'])['Age'].median()
        for idx, row in df_cleaned.iterrows():
            if pd.isna(row['Age']):
                df_cleaned.at[idx, 'Age'] = age_median[row['Pclass'], row['Sex']]
        
        # Handle missing values in Embarked - fill with mode
        if 'Embarked' in df_cleaned.columns:
            df_cleaned['Embarked'].fillna(df_cleaned['Embarked'].mode()[0], inplace=True)
        
        # Handle missing values in Fare - fill with median by Pclass
        if df_cleaned['Fare'].isna().any():
            fare_median = df_cleaned.groupby('Pclass')['Fare'].median()
            for idx, row in df_cleaned.iterrows():
                if pd.isna(row['Fare']):
                    df_cleaned.at[idx, 'Fare'] = fare_median[row['Pclass']]
        
        return df_cleaned