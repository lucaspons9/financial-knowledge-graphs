"""
Stanford OpenIE Extraction Runner

This script runs the Stanford OpenIE extraction on sample data and stores
the extracted triples as ground truth in versioned directories.
"""

import os
from typing import List, Dict, Any

from src.utils.file_utils import load_yaml
from src.utils.ground_truth import StanfordOpenIEExtractor
from src.utils.logging_utils import setup_logging, get_logger

# Initialize logger
logger = get_logger(__name__)

def main():
    """Main function to run Stanford OpenIE extraction."""
    # Setup logging
    setup_logging(log_level="INFO")
    logger.info("Starting Stanford OpenIE ground truth extraction")
    
    try:
        # Load configuration
        config = load_yaml("configs/config_stanford_openie.yaml")
        data_path = config.get("data_path", "data/raw/sample_for_openie.yaml")
        
        # Load sample data
        sample_data = load_yaml(data_path)
        sentence_ids = list(sample_data.keys())
        texts = list(sample_data.values())
        
        logger.info(f"Loaded {len(texts)} sample texts from {data_path}")
        
        # Initialize Stanford OpenIE extractor
        logger.info("Initializing Stanford OpenIE extractor...")
        openie_extractor = StanfordOpenIEExtractor(config_path="configs/config_stanford_openie.yaml")
        
        # Extract ground truth triples
        logger.info("Extracting ground truth triples...")
        
        # Process all texts with their IDs
        triples_batch = openie_extractor.batch_extract(texts, sentence_ids)
        
        # Print summary of extracted triples
        total_triples = sum(len(triples) for triples in triples_batch)
        logger.info(f"Extracted {total_triples} triples as ground truth")
        
        if openie_extractor.store_results:
            logger.info(f"Ground truth data saved to {openie_extractor.output_dir}")
            
        logger.info("Ground truth extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during ground truth extraction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 