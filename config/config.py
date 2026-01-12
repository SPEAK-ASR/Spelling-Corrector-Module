"""
Configuration module for Sinhala Spell Correction Project
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "dataset_id": "hasinduOnline/akura-sinhala-dyslexia-dataset",
    "train_split": "train",
    "test_split": "test",
}

# Processing configuration
PROCESSING_CONFIG = {
    "save_local_csv": True,
    "save_format": "csv",  # Options: 'csv', 'parquet', 'json'
}

# Hugging Face upload configuration
# Read from .env file if available, otherwise use defaults
HF_REPO_ID = os.getenv("HF_REPO_ID", "sinhala-dyslexia-cleaned")
HF_PRIVATE = os.getenv("HF_PRIVATE", "False").lower() == "true"

HF_UPLOAD_CONFIG = {
    "upload_to_hub": False,  # Set to True if you want to upload
    "repo_name": HF_REPO_ID,  # Now reads from .env HF_REPO_ID
    "repo_id": HF_REPO_ID,  # Also store as repo_id for compatibility
    "private": HF_PRIVATE,  # Now reads from .env HF_PRIVATE
}

# Model fine-tuning configuration (for future use)
MODEL_CONFIG = {
    "model_name": "bert-base-multilingual-cased",
    "max_length": 512,
    "batch_size": 32,
    "epochs": 3,
    "learning_rate": 2e-5,
}
