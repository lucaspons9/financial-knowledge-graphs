"""
Script for running LLM-based tasks.

This script executes various LLM tasks based on configuration,
processes results, and stores them in versioned directories.
"""

import os
import time

from src.utils.reading_files import load_yaml, load_data_by_extension
from src.utils.file_utils import find_next_versioned_dir, save_json, create_run_summary, save_results
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.text_processing import extract_json_from_output
from src.llm.model_handler import LLMHandler
from src.llm.openai_batch_processor import OpenAIBatchProcessor

# Initialize logger
logger = get_logger(__name__)

def main():
    """Main function to run LLM tasks."""
    # Setup logging
    setup_logging(log_level="INFO")
    logger.info("Starting LLM task execution")
    
    start_time = time.time()
    
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
        sample_data = load_data_by_extension(
            config["data_path"],
            id_column="newsID",
            text_column="story"
        )
        
        # Extract text from sample data
        texts = list(sample_data.values())
        sentence_ids = list(sample_data.keys())
        
        logger.info(f"Loaded {len(texts)} sample texts")
        
        # Initialize LLM Handler
        llm_handler = LLMHandler()
        
        # Check if batch processing should be used
        use_batch = config.get("use_batch", False)
        
        if use_batch and llm_handler.provider == "openai":
            # Use batch processing for OpenAI
            logger.info(f"Using OpenAI Batch API for processing {len(texts)} texts...")
            
            # Initialize the batch processor
            batch_processor = OpenAIBatchProcessor()
            
            # Get batch size from config or use default
            batch_size = config.get("batch_size", 2000)
            
            # Get parent batch directory if specified
            parent_batch_dir = config.get("parent_batch_dir")
            
            # Submit the batch
            batch_info = batch_processor.submit_batch(
                task=config["prompt"], 
                texts=texts,
                item_ids=sentence_ids,
                batch_size=batch_size,
                parent_batch_dir=parent_batch_dir
            )
            
            # Check for batch status
            if batch_info.get("status") == "skipped":
                logger.info("No new items to process, all have been processed already.")
            elif batch_info.get("status") == "failed":
                logger.error(f"Batch submission failed: {batch_info.get('error')}")
            else:
                logger.info(f"Batch submitted with ID: {batch_info.get('batch_id')}")
                
                # Store batch ID for later reference
                batch_id = batch_info.get("batch_id")
            
                # If configured to wait for results, check status and retrieve when ready
                wait_for_completion = config.get("wait_for_completion", False)
                
                if wait_for_completion:
                    if not batch_id:
                        logger.error("Batch submission failed or batch_id not returned")
                    else:
                        # Wait for batch to complete
                        logger.info("Waiting for batch to complete...")
                        
                        while True:
                            status_info = batch_processor.check_batch_status(batch_id)
                            
                            if "error" in status_info:
                                logger.error(f"Error checking batch status: {status_info['error']}")
                                break
                                    
                            if status_info.get("completed", False):
                                logger.info("Batch processing completed!")
                                
                                # Retrieve and save results
                                results_info = batch_processor.retrieve_batch_results(
                                    batch_id, 
                                    output_dir=test_dir if store_results else None
                                )
                                
                                logger.info(f"Retrieved {results_info.get('n_results', 0)} results")
                                break
                                    
                            # Wait before checking again
                            wait_time = 10  # seconds
                            logger.info(f"Batch status: {status_info.get('status', 'unknown')}. Checking again in {wait_time} seconds...")
                            time.sleep(wait_time)
                
                else:
                    logger.info(f"Batch submitted for asynchronous processing. Check status later with the batch ID: {batch_id}")
                    logger.info(f"Use: python -m src.retrieve_batch {batch_id} --check_only to check status")
                    logger.info(f"Use: python -m src.retrieve_batch {batch_id} to retrieve results when completed")
        else:
            # Process each text individually
            logger.info(f"Running task {config['prompt']} on {len(texts)} texts individually...")
            
            # Process each text individually to handle different sentence IDs
            results_files = {}
            for sentence_id, text in zip(sentence_ids, texts):
                logger.info(f"Processing {sentence_id}...")
                
                # Run the LLM task on the text
                response = llm_handler.run_task(config["prompt"], text)
                
                # Save results if configured
                if store_results and test_dir:
                    # Get the model provider
                    provider = llm_handler.provider

                    if provider == "llama3":
                        # For Llama3 models, try to extract JSON content 
                        # but also accept raw text if extraction fails
                        try:
                            json_content = extract_json_from_output(response)
                            results_file = save_results(json_content, test_dir, sentence_id)
                        except Exception:
                            # If JSON extraction fails, save the raw content
                            results_file = save_results(response, test_dir, sentence_id)
                    elif provider == "t5":
                        # For T5 models, the response is already formatted as JSON
                        results_file = save_results(response, test_dir, sentence_id)
                    elif config["prompt"] == "triplet_extraction" or config["prompt"] == "triplet_extraction_with_schema":
                        # For triplet extraction, parse the JSON output
                        triplets = extract_json_from_output(response.content)
                        results_file = save_results(triplets, test_dir, sentence_id)
                    else:
                        # For other tasks, save the raw output
                        try:
                            content = response.content if hasattr(response, 'content') else response
                            results_file = save_results(content, test_dir, sentence_id)
                        except Exception as e:
                            logger.error(f"Error saving results: {str(e)}")
                            results_file = save_results({"error": str(e)}, test_dir, sentence_id)
                    
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
            
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        execution_time = time.time() - start_time
        logger.info(f"Total execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main() 