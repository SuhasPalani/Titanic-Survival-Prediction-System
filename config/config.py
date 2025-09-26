import os
from pathlib import Path


class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = BASE_DIR / "models" / "saved_models"
    LOGS_DIR = BASE_DIR / "logs"

    # Data files
    TRAIN_FILE = RAW_DATA_DIR / "train.csv"
    TEST_FILE = RAW_DATA_DIR / "test.csv"
    PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train_processed.csv"
    PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "test_processed.csv"

    # Model files
    MODEL_FILE = MODELS_DIR / "titanic_model.pkl"
    PREPROCESSING_PIPELINE_FILE = MODELS_DIR / "preprocessing_pipeline.pkl"

    # Kafka configuration
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "titanic-predictions")

    # API configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 5000))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"

    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5

    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for directory in [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
