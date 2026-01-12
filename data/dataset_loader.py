"""
Dataset Loader Module for Sinhala Dyslexia Dataset

This module handles:
- Loading the akura-sinhala-dyslexia-dataset from Hugging Face
- Transforming the data into the desired table format
- Saving locally (CSV, Parquet, JSON)
- Uploading back to Hugging Face Hub
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from datasets import load_dataset, DatasetDict, concatenate_datasets
from huggingface_hub import login

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Load and process the Sinhala Dyslexia Dataset from Hugging Face
    """
    
    def __init__(self, dataset_id: str = "hasinduOnline/akura-sinhala-dyslexia-dataset"):
        """
        Initialize the dataset loader
        
        Args:
            dataset_id: The Hugging Face dataset ID
        """
        self.dataset_id = dataset_id
        self.dataset = None
        self.processed_data = {}
        logger.info(f"Initialized DatasetLoader for {dataset_id}")
    
    def load_dataset(self) -> DatasetDict:
        """
        Load the dataset from Hugging Face
        
        Returns:
            DatasetDict: The loaded dataset with splits
        """
        try:
            logger.info(f"Loading dataset: {self.dataset_id}")
            self.dataset = load_dataset(self.dataset_id)
            logger.info(f"Successfully loaded dataset. Splits: {list(self.dataset.keys())}")
            self._inspect_dataset()
            return self.dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def _inspect_dataset(self) -> None:
        """
        Inspect and log dataset structure and sample
        """
        if self.dataset is None:
            logger.warning("No dataset loaded. Call load_dataset() first.")
            return
        
        logger.info("\n" + "="*50)
        logger.info("DATASET STRUCTURE:")
        logger.info("="*50)
        logger.info(f"Dataset: {self.dataset}")
        
        for split_name in self.dataset.keys():
            logger.info(f"\nSplit: '{split_name}'")
            logger.info(f"  Number of samples: {len(self.dataset[split_name])}")
            logger.info(f"  Columns: {self.dataset[split_name].column_names}")
            logger.info(f"\nSample from '{split_name}':")
            logger.info(f"  {self.dataset[split_name][0]}")
        logger.info("="*50 + "\n")
    
    @staticmethod
    def get_error_type(analysis: List[Dict]) -> str:
        """
        Extract error type from analysis field
        
        Args:
            analysis: List of analysis dictionaries from the dataset
        
        Returns:
            str: The error type or 'unknown' if not found
        """
        import json
        
        if not analysis:
            return "unknown"
        
        # Parse JSON string if needed
        if isinstance(analysis, str):
            try:
                analysis = json.loads(analysis)
            except (json.JSONDecodeError, TypeError):
                return "unknown"
        
        # Handle list format
        if isinstance(analysis, list) and len(analysis) > 0:
            first = analysis[0]
        else:
            return "unknown"
        
        # Handle dict format
        if isinstance(first, dict):
            error_type = (
                first.get("error_type")
                or first.get("error")
                or first.get("pattern")
                or first.get("type")
                or "unknown"
            )
            return error_type
        
        return "unknown"
    
    def transform_data(self) -> Any:
        """
        Transform dataset into the desired table format and combine all splits:
        - clean_sentence (from corrected_text)
        - dyslexic_sentence (from input_text)
        - error_type (from analysis)
        
        Returns:
            Dataset: Single combined dataset with all data
        """
        if self.dataset is None:
            logger.error("Dataset not loaded. Call load_dataset() first.")
            raise ValueError("Dataset not loaded")
        
        logger.info("Starting data transformation...")
        
        def convert_example(example):
            """Convert a single example to the desired format"""
            return {
                "clean_sentence": example["corrected_text"].strip(),
                "dyslexic_sentence": example["input_text"].strip(),
                "error_type": self.get_error_type(example.get("analysis", []))
            }
        
        # Transform each split
        transformed_splits = []
        total_split_samples = 0
        for split_name in self.dataset.keys():
            split_size = len(self.dataset[split_name])
            logger.info(f"Transforming '{split_name}' split ({split_size} rows)...")
            transformed = self.dataset[split_name].map(
                convert_example,
                remove_columns=self.dataset[split_name].column_names
            )
            transformed_splits.append(transformed)
            logger.info(f"  ✓ Transformed {len(transformed)} samples from '{split_name}'")
            total_split_samples += len(transformed)
        
        # Combine all splits into one dataset
        logger.info("\n" + "="*60)
        logger.info("COMBINING ALL SPLITS INTO SINGLE DATASET")
        logger.info("="*60)
        combined_data = concatenate_datasets(transformed_splits)
        
        self.processed_data["all"] = combined_data
        logger.info(f"✓ Total samples in combined dataset: {len(combined_data)}")
        logger.info(f"✓ Expected total: {total_split_samples}")
        logger.info(f"✓ All rows loaded and combined successfully!")
        logger.info("="*60 + "\n")
        
        # Log a sample
        if len(combined_data) > 0:
            logger.info(f"Sample from combined dataset:")
            logger.info(f"  {combined_data[0]}")
        
        logger.info("Data transformation completed!")
        return combined_data
    
    def save_as_csv(self, output_dir: Path) -> Path:
        """
        Save processed data as CSV file
        
        Args:
            output_dir: Directory to save CSV file
        
        Returns:
            Path: Path to the saved file
        """
        if not self.processed_data:
            logger.error("No processed data found. Call transform_data() first.")
            raise ValueError("No processed data")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("\n" + "="*60)
        logger.info("SAVING DATASET AS CSV")
        logger.info("="*60)
        
        dataset = self.processed_data["all"]
        df = dataset.to_pandas()
        file_path = output_dir / "dataset.csv"
        df.to_csv(file_path, index=False, encoding='utf-8')
        logger.info(f"✓ Saved {len(df)} rows to {file_path}")
        logger.info(f"✓ File size: {file_path.stat().st_size / (1024*1024):.2f} MB")
        logger.info("="*60 + "\n")
        
        return file_path
    
    def save_as_parquet(self, output_dir: Path) -> Path:
        """
        Save processed data as Parquet file
        
        Args:
            output_dir: Directory to save Parquet file
        
        Returns:
            Path: Path to the saved file
        """
        if not self.processed_data:
            logger.error("No processed data found. Call transform_data() first.")
            raise ValueError("No processed data")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving data as Parquet to {output_dir}")
        
        dataset = self.processed_data["all"]
        file_path = output_dir / "dataset.parquet"
        dataset.to_parquet(str(file_path))
        logger.info(f"  Saved to {file_path}")
        
        return file_path
    
    def save_as_json(self, output_dir: Path) -> Path:
        """
        Save processed data as JSON file
        
        Args:
            output_dir: Directory to save JSON file
        
        Returns:
            Path: Path to the saved file
        """
        if not self.processed_data:
            logger.error("No processed data found. Call transform_data() first.")
            raise ValueError("No processed data")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving data as JSON to {output_dir}")
        
        dataset = self.processed_data["all"]
        file_path = output_dir / "dataset.json"
        dataset.to_json(str(file_path))
        logger.info(f"  Saved to {file_path}")
        
        return file_path
    
    def upload_to_hub(
        self,
        repo_name: str,
        private: bool = False,
        hf_token: Optional[str] = None
    ) -> None:
        """
        Upload processed dataset to Hugging Face Hub
        
        Args:
            repo_name: Name for the new repository (format: username/repo-name)
            private: Whether to make the dataset private
            hf_token: Hugging Face API token (optional, can use environment variable)
        """
        if not self.processed_data:
            logger.error("No processed data found. Call transform_data() first.")
            raise ValueError("No processed data")
        
        try:
            logger.info("Logging into Hugging Face Hub...")
            if hf_token:
                login(token=hf_token)
            else:
                login()  # Uses stored token or environment variable
            
            logger.info(f"Uploading dataset as '{repo_name}'...")
            
            dataset = self.processed_data["all"]
            logger.info(f"  Uploading combined dataset...")
            dataset.push_to_hub(
                repo_id=repo_name,
                split="train",
                private=private
            )
            logger.info(f"    Successfully uploaded")
            
            logger.info(f"Dataset successfully uploaded to https://huggingface.co/datasets/{repo_name}")
            
        except Exception as e:
            logger.error(f"Error uploading to Hugging Face Hub: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the processed data
        
        Returns:
            dict: Statistics including counts and unique error types
        """
        if not self.processed_data:
            logger.warning("No processed data found.")
            return {}
        
        dataset = self.processed_data["all"]
        df = dataset.to_pandas()
        
        stats = {
            "total_samples": len(df),
            "unique_error_types": df["error_type"].nunique(),
            "error_type_distribution": df["error_type"].value_counts().to_dict(),
            "avg_clean_length": df["clean_sentence"].str.len().mean(),
            "avg_dyslexic_length": df["dyslexic_sentence"].str.len().mean(),
        }
        
        return stats
    
    def print_statistics(self) -> None:
        """Print formatted statistics"""
        stats = self.get_statistics()
        
        logger.info("\n" + "="*50)
        logger.info("DATASET STATISTICS:")
        logger.info("="*50)
        
        logger.info(f"\nCombined Dataset:")
        logger.info(f"  Total samples: {stats['total_samples']}")
        logger.info(f"  Unique error types: {stats['unique_error_types']}")
        logger.info(f"  Avg clean sentence length: {stats['avg_clean_length']:.2f} chars")
        logger.info(f"  Avg dyslexic sentence length: {stats['avg_dyslexic_length']:.2f} chars")
        logger.info(f"  Error type distribution:")
        for error_type, count in stats['error_type_distribution'].items():
            logger.info(f"    - {error_type}: {count}")
        
        logger.info("="*50 + "\n")
