#!/usr/bin/env python
"""
Batch results retrieval script.

This script retrieves and processes results from all batches in a specified execution directory,
checking which ones haven't been retrieved yet and updating their metadata accordingly.
"""

import os
import sys
from typing import Dict, Any, List
import json
from src.llm.openai_batch_processor import OpenAIBatchProcessor
from src.utils.batch_utils import get_execution_path
from src.utils.logging_utils import get_logger
from src.llm import DEFAULT_BATCH_DIR

# Initialize logger
logger = get_logger(__name__)

def retrieve_execution_batches(execution_id: str, batch_dir: str = DEFAULT_BATCH_DIR) -> Dict[str, Any]:
    """
    Retrieve results for all unretrieved batches in an execution directory.
    
    Args:
        execution_id: The execution ID to process
        batch_dir: Base directory for batch processing
        
    Returns:
        Dict containing summary of processing results
    """
    # Get the execution directory path
    execution_dir = get_execution_path(execution_id, batch_dir)
    if not execution_dir:
        logger.error(f"Execution directory not found for ID: {execution_id}")
        return {"error": "Execution directory not found", "status": "failed"}
    
    logger.info(f"Processing batches in execution directory: {execution_dir}")
    
    # Initialize the OpenAI batch processor
    processor = OpenAIBatchProcessor()
    
    # Track results
    batches_list: List[Dict[str, Any]] = []
    results: Dict[str, Any] = {
        "execution_id": execution_id,
        "processed_batches": 0,
        "already_retrieved": 0,
        "newly_retrieved": 0,
        "failed": 0,
        "pending": 0,
        "batches": batches_list
    }
    
    # Find all batch directories in the execution directory
    batch_dirs = [d for d in os.listdir(execution_dir) 
                  if os.path.isdir(os.path.join(execution_dir, d)) 
                  and d.startswith('batch_')]
    
    # Process each batch directory
    for batch_dir in batch_dirs:
        batch_path = os.path.join(execution_dir, batch_dir)
        metadata_path = os.path.join(batch_path, "metadata.json")

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Get batch ID from metadata
            batch_id = metadata.get("batch_id")
            if not batch_id:
                logger.warning(f"No batch ID found in metadata for: {batch_dir}")
                results["failed"] += 1
                batches_list.append({
                    "batch_id": batch_dir,
                    "status": "failed", 
                    "error": "No batch ID in metadata"
                })
                continue
                
            # Check if already retrieved
            if metadata.get("retrieved", False):
                logger.info(f"Batch {batch_dir} (ID: {batch_id}) already retrieved")
                results["already_retrieved"] += 1
                batches_list.append({
                    "batch_id": batch_id,
                    "status": "already_retrieved"
                })
                continue
                
            # Retrieve batch results
            logger.info(f"Retrieving results for batch {batch_dir} (ID: {batch_id})")
            
            # Call the retrieve_batch_items function
            retrieve_result = processor.retrieve_batch_items(batch_id, metadata, batch_path)
            
            # Update results based on retrieval status
            if retrieve_result.get("status") == "completed":
                logger.info(f"Successfully retrieved and processed batch {batch_id}")
                results["newly_retrieved"] += 1
                batches_list.append({
                    "batch_id": batch_id,
                    "status": "newly_retrieved",
                    "n_results": retrieve_result.get("n_results", 0)
                })
            elif retrieve_result.get("status") == "already_retrieved":
                logger.info(f"Batch {batch_dir} (ID: {batch_id}) was already retrieved")
                results["already_retrieved"] += 1
                batches_list.append({
                    "batch_id": batch_id,
                    "status": "already_retrieved"
                })
            elif not retrieve_result.get("completed", False):
                logger.warning(f"Batch {batch_dir} (ID: {batch_id}) is not completed yet (status: {retrieve_result.get('status')})")
                results["pending"] += 1
                batches_list.append({
                    "batch_id": batch_id,
                    "status": "pending",
                    "batch_status": retrieve_result.get("status")
                })
            else:
                logger.error(f"Failed to retrieve batch {batch_dir} (ID: {batch_id}): {retrieve_result.get('error')}")
                results["failed"] += 1
                batches_list.append({
                    "batch_id": batch_id,
                    "status": "failed",
                    "error": retrieve_result.get("error")
                })
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_dir} (ID: {batch_id}): {str(e)}")
            results["failed"] += 1
            batches_list.append({
                "batch_id": batch_dir,
                "status": "failed",
                "error": str(e)
            })
    
    # Update total processed count
    results["processed_batches"] = len(batch_dirs)
    
    # Print summary
    logger.info(f"Execution {execution_id} processing summary:")
    logger.info(f"Total batches: {results['processed_batches']}")
    logger.info(f"Already retrieved: {results['already_retrieved']}")
    logger.info(f"Newly retrieved: {results['newly_retrieved']}")
    logger.info(f"Pending completion: {results['pending']}")
    logger.info(f"Failed: {results['failed']}")
    
    return results

def main():
    """Main function to process command line arguments and retrieve batches."""
    # Check for execution ID argument
    # if len(sys.argv) < 2:
    #     print("Usage: python run_retrieve_batch.py <execution_id>")
    #     print("  execution_id: The ID or name of the execution directory to process")
    #     sys.exit(1)
    
    # # Get execution ID from command line arguments
    # execution_id = sys.argv[1]
    execution_id = "execution_1"
    batch_dir = DEFAULT_BATCH_DIR
    
    logger.info(f"Starting batch retrieval for execution ID: {execution_id}")
    
    # Retrieve and process batches
    results = retrieve_execution_batches(execution_id, batch_dir)
    
    # Check for errors
    if "error" in results:
        logger.error(f"Batch retrieval failed: {results['error']}")
        sys.exit(1)
    
    logger.info("Batch retrieval completed successfully")
    
    # Exit with success
    sys.exit(0)

if __name__ == "__main__":
    main()
