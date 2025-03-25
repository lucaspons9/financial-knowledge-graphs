"""
Script for running LLM-based tasks.

This script executes various LLM tasks based on configuration,
processes results, and stores them in versioned directories.
"""

import os
import re
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Union, Optional

from src.utils.reading_files import load_yaml, load_csv_news
from src.utils.file_utils import ensure_dir, find_next_versioned_dir, save_json, create_run_summary
from src.utils.logging_utils import setup_logging, get_logger
from src.llm.batch_processor import BatchProcessor

# Initialize logger
logger = get_logger(__name__)

def extract_json_from_output(output_content: str) -> List[Dict[str, str]]:
    """Extract JSON data from the model output."""
    try:
        # Look for content between ```json and ``` markers if present
        if "```json" in output_content and "```" in output_content.split("```json")[1]:
            json_str = output_content.split("```json")[1].split("```")[0].strip()
        # Otherwise, try to parse the whole content as JSON
        else:
            json_str = output_content.strip()
        
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON from output: {output_content}")
        # Try to handle other formats here if needed
        return []

def save_results(results: Union[List[Dict[str, str]], str], test_dir: str, sentence_id: str) -> str:
    """Save results to a JSON file and return the file path."""
    # If results is a string, try to parse it as JSON
    if isinstance(results, str):
        try:
            results = extract_json_from_output(results)
        except Exception:
            # If parsing fails, store as is in a dict
            results = {"raw_output": results}
    
    # Save to file
    file_path = os.path.join(test_dir, f"{sentence_id}.json")
    save_json(results, file_path)
    
    return file_path

def main():
    """Main function to run LLM tasks."""
    # Setup logging
    setup_logging(log_level="INFO")
    logger.info("Starting LLM task execution")
    
    try:
        # Load configuration
        config = load_yaml("configs/config_llm_execution.yaml")
        
        # Check if results should be stored
        store_results = config.get("store_results", False)
        
        # Setup test folder if storing results
        test_dir = None
        if store_results:
            results_dir = config.get("results_dir", "runs")
            test_name = config.get("test_name", "test_llm")
            test_dir = find_next_versioned_dir(results_dir, test_name)
            logger.info(f"Results will be stored in: {test_dir}")
        
        # Load sample data based on file extension
        data_path = config["data_path"]
        file_extension = os.path.splitext(data_path)[1].lower()
        
        if file_extension == '.yaml' or file_extension == '.yml':
            # Load from YAML (original behavior)
            sample_data = load_yaml(data_path)
        elif file_extension == '.csv':
            # Load from CSV using the new function
            sample_data = load_csv_news(
                data_path, 
                id_column=config.get("id_column", "newsID"), 
                text_column=config.get("text_column", "story")
            )
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        # Extract text from sample data
        texts = list(sample_data.values())
        sentence_ids = list(sample_data.keys())
        
        logger.info(f"Loaded {len(texts)} sample texts")
        
        # Initialize BatchProcessor
        batch_processor = BatchProcessor()
        
        # Run batch processing
        logger.info(f"Running batch {config['prompt']}...")
        
        # Process each text individually to handle different sentence IDs
        results_files = {}
        for i, (sentence_id, text) in enumerate(zip(sentence_ids, texts), start=1):
            logger.info(f"Processing {sentence_id}...")
            
            # Run the LLM task on the text
            response = batch_processor.run_task(config["prompt"], text)
            
            # Print the result
            logger.info(f"Results for {sentence_id}: {response.content}")
            
            # Save results if configured
            if store_results and test_dir:
                if config["prompt"] == "triplet_extraction":
                    # For triplet extraction, parse the JSON output
                    triplets = extract_json_from_output(response.content)
                    results_file = save_results(triplets, test_dir, sentence_id)
                else:
                    # For other tasks, save the raw output
                    results_file = save_results(response.content, test_dir, sentence_id)
                
                results_files[sentence_id] = results_file
                logger.info(f"Saved results to {results_file}")
        
        # Save a summary file if storing results
        if store_results and test_dir:
            summary = create_run_summary(
                config={k: v for k, v in config.items() if k != "api_key"},
                stats={
                    "num_texts": len(texts),
                    "results_files": results_files
                }
            )
            
            summary_path = os.path.join(test_dir, "summary.json")
            save_json(summary, summary_path)
            logger.info(f"Summary saved to {summary_path}")
            
        logger.info("Batch processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during batch processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 