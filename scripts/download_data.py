# scripts/download_data.py
import pandas as pd
import requests
import os


def download_titanic_data():
    """Download Titanic dataset from a reliable source"""
    print("Downloading Titanic dataset...")

    # Create data directory
    os.makedirs("data/raw", exist_ok=True)

    # URLs for Titanic dataset (using a reliable public source)
    train_url = (
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )

    try:
        # Download training data
        response = requests.get(train_url)
        response.raise_for_status()

        with open("data/raw/train.csv", "wb") as f:
            f.write(response.content)

        print("✅ Training data downloaded successfully")

        # Load and inspect the data
        df = pd.read_csv("data/raw/train.csv")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())

        # Create a test set by splitting the data
        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["survived"]
        )

        # Save test set (without target variable for realistic testing)
        test_features = test_df.drop("survived", axis=1)
        test_features.to_csv("data/raw/test.csv", index=False)

        # Update train set
        train_df.to_csv("data/raw/train.csv", index=False)

        print("✅ Data split into train and test sets")

    except requests.RequestException as e:
        print(f"❌ Error downloading data: {e}")
        print("Please download the Titanic dataset manually from Kaggle")
        print("https://www.kaggle.com/c/titanic/data")


if __name__ == "__main__":
    download_titanic_data()
