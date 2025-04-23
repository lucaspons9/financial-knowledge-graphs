import os
import time
from typing import Dict, List, Tuple, Optional, Any

from src.utils.file_utils import load_yaml, load_data_by_extension, setup_results_directory, save_json, create_run_summary, save_results
from src.utils.batch_utils import get_execution_path, get_processed_item_ids
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.text_processing import extract_json_from_output
from src.llm.model_handler import LLMHandler
from src.llm.openai_batch_processor import OpenAIBatchProcessor

# Initialize logger
logger = get_logger(__name__)

def load_data(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Load sample data and extract texts and IDs.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple[List[str], List[str]]: Tuple of texts and their IDs
    """
    sample_data = load_data_by_extension(
        config["data_path"],
        id_column="newsID",
        text_column="story"
    )
    
    news_texts = list(sample_data.values())
    news_ids = list(sample_data.keys())
    
    logger.info(f"Loaded {len(news_texts)} sample texts")
    
    return news_texts, news_ids

def filter_processed_texts_and_ids(news_texts: List[str], news_ids: List[str], execution_dir: str) -> Tuple[List[str], List[str]]:
    # Retrieve all processed item IDs from that directory
    if execution_dir:
        processed_ids = get_processed_item_ids(execution_dir)
        logger.info(f"Filtered out {len(processed_ids)} item IDs that have already been processed.")

        # Filter out already processed texts
        texts_batch = [text for text, id in zip(news_texts, news_ids) if id not in processed_ids]
        ids_batch = [id for id in news_ids if id not in processed_ids]

        return texts_batch, ids_batch
    else:
        return news_texts, news_ids

def process_batch(
    news_texts: List[str],
    news_ids: List[str],
    config: Dict[str, Any]
) -> None:
    """Process texts using OpenAI Batch API.
    
    Args:
        texts: List of texts to process
        sentence_ids: List of corresponding text IDs
        config: Configuration dictionary
        test_dir: Directory to store results, if any
    """
    logger.info(f"Using OpenAI Batch API for processing {len(news_texts)} texts...")
    
    # Get the path to the execution directory
    execution_dir = get_execution_path(config.get("execution_id")) 

    # Initialize the batch processor
    batch_processor = OpenAIBatchProcessor()
    
    # Filter out already processed texts
    texts_batch, ids_batch = filter_processed_texts_and_ids(news_texts, news_ids, execution_dir)
    
    # Determine the batch size from the configuration and create a batch of texts to process
    batch_size = config.get("batch_size", 5000)
    if len(texts_batch) > batch_size:
        texts_batch = texts_batch[0:batch_size] 
        ids_batch = ids_batch[0:batch_size]
    logger.info(f"Processing {len(texts_batch)} texts in a single batch...")

    # Submit a single batch of unprocessed items
    batch_info = batch_processor.submit_batch(
        task=config["prompt"], 
        texts=texts_batch,
        item_ids=ids_batch,
        execution_dir=execution_dir
    )
    
    handle_batch_submission_result(batch_info)

def handle_batch_submission_result(
    batch_info: Dict[str, Any],
) -> None:
    """Handle the result of a batch submission.
    
    Args:
        batch_processor: OpenAI batch processor instance
        batch_info: Information about the submitted batch
        config: Configuration dictionary
        test_dir: Directory to store results, if any
    """
    # Check for batch status
    if batch_info.get("status") == "skipped":
        logger.info("No new items to process, all have been processed already.")
    elif batch_info.get("status") == "failed":
        logger.error(f"Batch submission failed: {batch_info.get('error')}")
    else:
        logger.info(f"Batch submitted with ID: {batch_info.get('batch_id')}")
        
def process_individually(
    llm_handler: LLMHandler,
    news_texts: List[str],
    news_ids: List[str],
    config: Dict[str, Any],
    test_dir: Optional[str]
) -> Dict[str, str]:
    """Process each text individually.
    
    Args:
        llm_handler: LLM handler instance
        texts: List of texts to process
        sentence_ids: List of corresponding text IDs
        config: Configuration dictionary
        test_dir: Directory to store results, if any
        
    Returns:
        Dict[str, str]: Dictionary mapping sentence IDs to result file paths
    """
    logger.info(f"Running task {config['prompt']} on {len(news_texts)} texts individually...")
    
    results_files: Dict[str, str] = {}
    for news_id, news_text in zip(news_ids, news_texts):
        logger.info(f"Processing {news_id}...")
        
        # Run the LLM task on the text
        response = llm_handler.run_task(config["prompt"], news_text)
        
        # Save results if configured
        store_results = config.get("store_results", False)
        if store_results and test_dir:
            results_file = save_individual_result(response, test_dir, news_id, config, llm_handler)
            results_files[news_id] = results_file
            logger.info(f"Saved results to {results_file}")
    
    return results_files

def save_individual_result(
    response: Any,
    test_dir: str,
    sentence_id: str,
    config: Dict[str, Any],
    llm_handler: LLMHandler
) -> str:
    """Save individual result based on model provider and task.
    
    Args:
        response: LLM response to save
        test_dir: Directory to store results
        sentence_id: ID of the processed text
        config: Configuration dictionary
        llm_handler: LLM handler instance
        
    Returns:
        str: Path to the saved result file
    """
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
    
    return results_file

def save_summary(config: Dict[str, Any], test_dir: str, results_files: Dict[str, str], num_texts: int) -> None:
    """Save a summary of the run.
    
    Args:
        config: Configuration dictionary
        test_dir: Directory to store results
        results_files: Dictionary mapping sentence IDs to result file paths
        num_texts: Number of processed texts
    """
    summary = create_run_summary(
        config={k: v for k, v in config.items() if k != "api_key"},
        stats={
            "num_texts": num_texts,
            "results_files": results_files
        }
    )
    
    summary_path = os.path.join(test_dir, "summary.json")
    save_json(summary, summary_path)
    logger.info(f"Summary saved to {summary_path}")

def main():
    """Main function to run LLM tasks."""
    # Setup logging
    setup_logging(log_level="INFO")
    logger.info("Starting LLM task execution")
    
    start_time = time.time()
    
    try:
        # Load configuration
        config = load_yaml("configs/config_llm_execution.yaml")
        
        # Load and prepare data
        news_texts, news_ids = load_data(config)
        
        # Initialize LLM Handler
        llm_handler = LLMHandler()
        
        if config.get("use_batch", False) and llm_handler.provider == "openai":
            process_batch(news_texts, news_ids, config)
        else:
            # Setup results directory if needed
            test_dir = setup_results_directory(config)

            results_files = process_individually(llm_handler, news_texts, news_ids, config, test_dir)
            
            # Save a summary file if storing results
            if config.get("store_results", False) and test_dir:
                save_summary(config, test_dir, results_files, len(news_texts))
        
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