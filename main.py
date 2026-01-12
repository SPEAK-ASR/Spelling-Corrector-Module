"""
Main script for dataset creation and processing

Usage:
    python main.py --load              # Just load the dataset
    python main.py --process           # Load and transform the dataset
    python main.py --save-csv          # Load, transform, and save as CSV
    python main.py --save-parquet      # Load, transform, and save as Parquet
    python main.py --stats             # Show statistics
    python main.py --all               # Do everything (except upload)
"""

import argparse
import logging
from pathlib import Path
from config.config import (
    DATASET_CONFIG,
    PROCESSING_CONFIG,
    HF_UPLOAD_CONFIG,
    DATA_DIR,
    RAW_DATA_DIR,
)
from data.dataset_loader import DatasetLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run dataset processing pipeline"""
    
    parser = argparse.ArgumentParser(
        description="Sinhala Spell Correction Dataset Processor"
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load the dataset from Hugging Face"
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Load and transform the dataset"
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save processed data as CSV"
    )
    parser.add_argument(
        "--save-parquet",
        action="store_true",
        help="Save processed data as Parquet"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show dataset statistics"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to Hugging Face Hub (requires auth)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Do everything: load, process, and save (except upload)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face API token for upload (optional)"
    )
    
    args = parser.parse_args()
    
    # Initialize dataset loader
    logger.info("Initializing Dataset Loader...")
    loader = DatasetLoader(dataset_id=DATASET_CONFIG["dataset_id"])
    
    try:
        # If no arguments provided, show help
        if not any([args.load, args.process, args.save_csv, args.save_parquet, args.stats, args.upload, args.all]):
            parser.print_help()
            return
        
        # Load dataset
        if args.load or args.process or args.save_csv or args.save_parquet or args.stats or args.all:
            logger.info("\n" + "="*60)
            logger.info("STEP 1: LOADING DATASET")
            logger.info("="*60)
            loader.load_dataset()
        
        # Transform data
        if args.process or args.save_csv or args.save_parquet or args.stats or args.all:
            logger.info("\n" + "="*60)
            logger.info("STEP 2: TRANSFORMING DATA")
            logger.info("="*60)
            loader.transform_data()
        
        # Save as CSV
        if args.save_csv or args.all:
            logger.info("\n" + "="*60)
            logger.info("STEP 3: SAVING AS CSV")
            logger.info("="*60)
            csv_dir = DATA_DIR / "csv"
            loader.save_as_csv(csv_dir)
        
        # Save as Parquet
        if args.save_parquet or args.all:
            logger.info("\n" + "="*60)
            logger.info("STEP 4: SAVING AS PARQUET")
            logger.info("="*60)
            parquet_dir = DATA_DIR / "parquet"
            loader.save_as_parquet(parquet_dir)
        
        # Show statistics
        if args.stats or args.all:
            logger.info("\n" + "="*60)
            logger.info("STEP 5: DISPLAYING STATISTICS")
            logger.info("="*60)
            loader.print_statistics()
        
        # Upload to Hugging Face Hub
        if args.upload:
            logger.info("\n" + "="*60)
            logger.info("STEP 6: UPLOADING TO HUGGING FACE HUB")
            logger.info("="*60)
            repo_name = HF_UPLOAD_CONFIG["repo_name"]
            logger.info(f"Repository name: {repo_name}")
            loader.upload_to_hub(
                repo_name=repo_name,
                private=HF_UPLOAD_CONFIG["private"],
                hf_token=args.hf_token
            )
        
        logger.info("\n" + "="*60)
        logger.info("DATASET PROCESSING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
