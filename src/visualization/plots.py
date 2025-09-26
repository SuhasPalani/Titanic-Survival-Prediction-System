# src/visualization/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class TitanicVisualizer:
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_survival_by_class(self, df):
        """Plot survival rate by passenger class"""
        plt.figure(figsize=(10, 6))
        survival_by_class = df.groupby('Pclass')['Survived'].mean()
        
        plt.subplot(1, 2, 1)
        survival_by_class.plot(kind='bar')
        plt.title('Survival Rate by Passenger Class')
        plt.xlabel('Passenger Class')
        plt.ylabel('Survival Rate')
        plt.xticks(rotation=0)
        
        plt.subplot(1, 2, 2)
        class_counts = df['Pclass'].value_counts().sort_index()
        plt.pie(class_counts.values, labels=[f'Class {i}' for i in class_counts.index], autopct='%1.1f%%')
        plt.title('Distribution of Passenger Classes')
        
        plt.tight_layout()
        plt.show()
    
    def plot_survival_by_gender(self, df):
        """Plot survival rate by gender"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        survival_by_gender = df.groupby('Sex')['Survived'].mean()
        survival_by_gender.plot(kind='bar', color=['skyblue', 'pink'])
        plt.title('Survival Rate by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Survival Rate')
        plt.xticks(rotation=0)
        
        plt.subplot(1, 2, 2)
        crosstab = pd.crosstab(df['Sex'], df['Survived'])
        crosstab.plot(kind='bar', stacked=True)
        plt.title('Survival Count by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.legend(['Died', 'Survived'])
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.show()
    
    def plot_age_distribution(self, df):
        """Plot age distribution and survival by age"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(df['Age'].dropna(), bins=30, alpha=0.7, edgecolor='black')
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 2)
        survived = df[df['Survived'] == 1]['Age'].dropna()
        died = df[df['Survived'] == 0]['Age'].dropna()
        plt.hist([died, survived], bins=30, alpha=0.7, label=['Died', 'Survived'], color=['red', 'green'])
        plt.title('Age Distribution by Survival')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        age_groups = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        survival_by_age = df.groupby(age_groups)['Survived'].mean()
        survival_by_age.plot(kind='bar')
        plt.title('Survival Rate by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Survival Rate')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_family_size_analysis(self, df):
        """Plot family size analysis"""
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        family_survival = df.groupby('FamilySize')['Survived'].mean()
        family_survival.plot(kind='bar')
        plt.title('Survival Rate by Family Size')
        plt.xlabel('Family Size')
        plt.ylabel('Survival Rate')
        
        plt.subplot(1, 3, 2)
        alone_survival = df.groupby('IsAlone')['Survived'].mean()
        alone_survival.plot(kind='bar', color=['orange', 'purple'])
        plt.title('Survival Rate: Alone vs With Family')
        plt.xlabel('Is Alone (0=No, 1=Yes)')
        plt.ylabel('Survival Rate')
        plt.xticks(rotation=0)
        
        plt.subplot(1, 3, 3)
        plt.scatter(df['SibSp'], df['Parch'], c=df['Survived'], alpha=0.6, cmap='RdYlGn')
        plt.title('Survival by Siblings/Spouses vs Parents/Children')
        plt.xlabel('Siblings/Spouses')
        plt.ylabel('Parents/Children')
        plt.colorbar(label='Survived')
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, df):
        """Plot correlation heatmap"""
        plt.figure(figsize=(12, 10))
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()
    
    def plot_fare_analysis(self, df):
        """Plot fare analysis"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(df['Fare'], bins=50, alpha=0.7, edgecolor='black')
        plt.title('Fare Distribution')
        plt.xlabel('Fare')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 2)
        fare_by_class = df.groupby('Pclass')['Fare'].mean()
        fare_by_class.plot(kind='bar', color=['gold', 'silver', 'brown'])
        plt.title('Average Fare by Class')
        plt.xlabel('Passenger Class')
        plt.ylabel('Average Fare')
        plt.xticks(rotation=0)
        
        plt.subplot(1, 3, 3)
        fare_groups = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        survival_by_fare = df.groupby(fare_groups)['Survived'].mean()
        survival_by_fare.plot(kind='bar')
        plt.title('Survival Rate by Fare Group')
        plt.xlabel('Fare Group')
        plt.ylabel('Survival Rate')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_embarkation_analysis(self, df):
        """Plot embarkation port analysis"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        embark_counts = df['Embarked'].value_counts()
        embark_counts.plot(kind='bar', color=['lightblue', 'lightgreen', 'lightcoral'])
        plt.title('Passenger Count by Embarkation Port')
        plt.xlabel('Embarkation Port')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        plt.subplot(1, 2, 2)
        survival_by_embark = df.groupby('Embarked')['Survived'].mean()
        survival_by_embark.plot(kind='bar', color=['lightblue', 'lightgreen', 'lightcoral'])
        plt.title('Survival Rate by Embarkation Port')
        plt.xlabel('Embarkation Port')
        plt.ylabel('Survival Rate')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.show()
    
    def comprehensive_eda(self, df):
        """Run comprehensive EDA"""
        print("=== TITANIC SURVIVAL ANALYSIS ===\n")
        print(f"Dataset shape: {df.shape}")
        print(f"Overall survival rate: {df['Survived'].mean():.3f}")
        print("\n" + "="*50 + "\n")
        
        self.plot_survival_by_class(df)
        self.plot_survival_by_gender(df)
        self.plot_age_distribution(df)
        self.plot_family_size_analysis(df)
        self.plot_fare_analysis(df)
        self.plot_embarkation_analysis(df)
        self.plot_correlation_heatmap(df)