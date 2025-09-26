import pandas as pd
import numpy as np
from pathlib import Path



class DataLoader:
    def __init__(self, data_path="data/raw"):
        self.data_path = Path(data_path)
    
    def load_train_data(self):
        """Load training data"""
        return pd.read_csv(self.data_path / "train.csv")
    
    def load_test_data(self):
        """Load test data"""
        return pd.read_csv(self.data_path / "test.csv")