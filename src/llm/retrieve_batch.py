"""
Utility script to retrieve results from OpenAI Batch API jobs.

Usage:
    python -m src.retrieve_batch <batch_id> [--parent] [--output_dir <directory>]
    python -m src.retrieve_batch <batch_id> [--parent] --check_only
    python -m src.retrieve_batch <batch_id> [--parent] --wait
"""

import sys
import argparse
import time
import os
import json
import re
from typing import Dict, Any, Optional

from src.llm.openai_batch_processor import OpenAIBatchProcessor
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.file_utils import ensure_dir

# Initialize logger
logger = get_logger(__name__)

class BatchRetriever:
    """Class to handle both individual and parent batch retrieval operations."""
    
    def __init__(self, batch_dir: str = "data/batch_processing"):
        """Initialize the batch retriever.
        
        Args:
            batch_dir: Directory where batch metadata is stored
        """
        self.batch_processor = OpenAIBatchProcessor()
        self.batch_dir = batch_dir
    
    def _is_folder_name(self, batch_id: str) -> bool:
        """Check if the provided batch ID looks like a folder name.
        
        Args:
            batch_id: The batch ID to check
            
        Returns:
            bool: True if it looks like a folder name, False otherwise
        """
        # Pattern for folder names like batch_YYYYMMDD_HHMMSS_ID
        folder_pattern = r'^batch_\d{8}_\d{6}_[a-f0-9]{8}$'
        return bool(re.match(folder_pattern, batch_id))
    
    def _get_real_batch_id(self, folder_name: str) -> Optional[str]:
        """Get the real OpenAI batch ID from a folder's metadata.
        
        Args:
            folder_name: The batch folder name
            
        Returns:
            Optional[str]: The real OpenAI batch ID or None if not found
        """
        # First check directly in batch_dir
        folder_path = os.path.join(self.batch_dir, folder_name)
        metadata_path = os.path.join(folder_path, "metadata.json")
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get("batch_id")
            except Exception as e:
                logger.error(f"Error reading metadata from {metadata_path}: {str(e)}")
        
        # Check in parent batch directories
        for parent_dir in os.listdir(self.batch_dir):
            parent_path = os.path.join(self.batch_dir, parent_dir)
            if os.path.isdir(parent_path) and parent_dir.startswith("parent_batch_"):
                folder_path = os.path.join(parent_path, folder_name)
                metadata_path = os.path.join(folder_path, "metadata.json")
                
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            return metadata.get("batch_id")
                    except Exception as e:
                        logger.error(f"Error reading metadata from {metadata_path}: {str(e)}")
        
        return None
    
    def check_batch_status(self, batch_id: str, is_parent: bool = False) -> Dict[str, Any]:
        """Check the status of a batch or parent batch.
        
        Args:
            batch_id: The batch ID to check (can be folder name or OpenAI batch ID)
            is_parent: Whether this is a parent batch ID
            
        Returns:
            Dict containing status information
        """
        # If it's a parent batch, delegate to parent batch method
        if is_parent:
            return self._check_parent_batch_status(batch_id)
        
        # If the batch ID looks like a folder name, get the real batch ID from metadata
        if self._is_folder_name(batch_id):
            logger.info(f"Input appears to be a batch folder name: {batch_id}")
            real_batch_id = self._get_real_batch_id(batch_id)
            
            if real_batch_id:
                logger.info(f"Found real OpenAI batch ID: {real_batch_id}")
                batch_id = real_batch_id
            else:
                logger.warning(f"Could not find real batch ID for folder {batch_id}. Will attempt to use as-is.")
        
        # Check batch status using the batch processor
        return self.batch_processor.check_batch_status(batch_id, self.batch_dir)
    
    def retrieve_batch_results(self, batch_id: str, is_parent: bool = False, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve results from a batch or parent batch.
        
        Args:
            batch_id: The batch ID to retrieve results for (can be folder name or OpenAI batch ID)
            is_parent: Whether this is a parent batch ID
            output_dir: Directory to save results
            
        Returns:
            Dict containing results information
        """
        # If it's a parent batch, delegate to parent batch method
        if is_parent:
            return self._retrieve_parent_batch_results(batch_id, output_dir)
        
        # If the batch ID looks like a folder name, get the real batch ID from metadata
        original_batch_id = batch_id
        if self._is_folder_name(batch_id):
            logger.info(f"Input appears to be a batch folder name: {batch_id}")
            real_batch_id = self._get_real_batch_id(batch_id)
            
            if real_batch_id:
                logger.info(f"Found real OpenAI batch ID: {real_batch_id}")
                batch_id = real_batch_id
            else:
                logger.warning(f"Could not find real batch ID for folder {original_batch_id}. Will attempt to use as-is.")
        
        # Retrieve batch results using the batch processor
        return self.batch_processor.retrieve_batch_results(
            batch_id,
            batch_dir=self.batch_dir,
            output_dir=output_dir
        )
    
    def _check_parent_batch_status(self, parent_batch_id: str) -> Dict[str, Any]:
        """Check the status of a parent batch by examining all child batches.
        
        Args:
            parent_batch_id: The parent batch ID to check
            
        Returns:
            Dict containing status information
        """
        parent_dir = os.path.join(self.batch_dir, parent_batch_id)
        
        if not os.path.exists(parent_dir):
            return {"error": f"Parent batch directory not found: {parent_dir}", "status": "failed"}
        
        # Load parent batch metadata
        parent_info_path = os.path.join(parent_dir, "parent_batch_info.json")
        if not os.path.exists(parent_info_path):
            return {"error": f"Parent batch metadata not found: {parent_info_path}", "status": "failed"}
        
        try:
            with open(parent_info_path, 'r') as f:
                parent_info = json.load(f)
            
            batches = parent_info.get("batches", [])
            
            if not batches:
                return {
                    "parent_batch_id": parent_batch_id,
                    "status": "empty",
                    "total_batches": 0,
                    "completed_batches": 0,
                    "error": "No batches found in parent batch"
                }
            
            # Check status of each batch
            completed_batches = 0
            failed_batches = 0
            in_progress_batches = 0
            
            for batch in batches:
                batch_id = batch.get("batch_id")
                if not batch_id:
                    continue
                
                # Get the real OpenAI batch ID if this is a folder name
                if self._is_folder_name(batch_id):
                    real_batch_id = self._get_real_batch_id(batch_id)
                    if real_batch_id:
                        batch_id = real_batch_id
                
                status_info = self.batch_processor.check_batch_status(batch_id, self.batch_dir)
                batch_status = status_info.get("status", "unknown")
                
                if batch_status == "completed":
                    completed_batches += 1
                elif batch_status == "failed":
                    failed_batches += 1
                else:
                    in_progress_batches += 1
            
            # Determine overall status
            if completed_batches == len(batches):
                overall_status = "completed"
            elif failed_batches > 0:
                overall_status = "partially_failed"
            else:
                overall_status = "in_progress"
            
            return {
                "parent_batch_id": parent_batch_id,
                "status": overall_status,
                "total_batches": len(batches),
                "completed_batches": completed_batches,
                "failed_batches": failed_batches,
                "in_progress_batches": in_progress_batches,
                "processed_items": len(parent_info.get("processed_item_ids", []))
            }
            
        except Exception as e:
            return {"error": f"Error checking parent batch status: {str(e)}", "status": "failed"}
    
    def _retrieve_parent_batch_results(self, parent_batch_id: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve results from all batches in a parent batch.
        
        Args:
            parent_batch_id: The parent batch ID to retrieve results for
            output_dir: Directory to save results
            
        Returns:
            Dict containing results information
        """
        # First check parent batch status
        status_info = self._check_parent_batch_status(parent_batch_id)
        
        if "error" in status_info:
            return status_info
        
        parent_dir = os.path.join(self.batch_dir, parent_batch_id)
        
        # If output directory not specified, create one in the parent batch directory
        if not output_dir:
            output_dir = os.path.join(parent_dir, "combined_results")
        
        ensure_dir(output_dir)
        
        # Load parent batch metadata
        parent_info_path = os.path.join(parent_dir, "parent_batch_info.json")
        with open(parent_info_path, 'r') as f:
            parent_info = json.load(f)
        
        batches = parent_info.get("batches", [])
        
        # Retrieve results from each batch
        total_results = 0
        successful_batches = 0
        failed_batches: list[str] = []
        
        for batch in batches:
            batch_id = batch.get("batch_id")
            if not batch_id:
                continue
            
            # Get the real OpenAI batch ID if this is a folder name
            if self._is_folder_name(batch_id):
                real_batch_id = self._get_real_batch_id(batch_id)
                if real_batch_id:
                    batch_id = real_batch_id
            
            # Check if batch is completed before attempting to retrieve
            status_info = self.batch_processor.check_batch_status(batch_id, self.batch_dir)
            if status_info.get("status") != "completed":
                logger.info(f"Batch {batch_id} not yet completed. Skipping retrieval.")
                continue
            
            # Retrieve batch results
            results_info = self.batch_processor.retrieve_batch_results(
                batch_id,
                batch_dir=self.batch_dir,
                output_dir=output_dir
            )
            
            if "error" in results_info:
                logger.error(f"Error retrieving results for batch {batch_id}: {results_info['error']}")
                failed_batches.append(batch_id)
            else:
                total_results += results_info.get("n_results", 0)
                successful_batches += 1
        
        return {
            "parent_batch_id": parent_batch_id,
            "status": "completed" if not failed_batches else "partially_completed",
            "total_results": total_results,
            "total_batches": len(batches),
            "successful_batches": successful_batches,
            "failed_batches": failed_batches,
            "results_path": output_dir
        }

def main():
    """Retrieve results from OpenAI Batch API jobs."""
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Retrieve OpenAI Batch API results")
    parser.add_argument("batch_id", help="Batch ID or Parent Batch ID to retrieve results for")
    parser.add_argument("--parent", action="store_true", help="Treat the batch_id as a parent batch ID")
    parser.add_argument("--output_dir", help="Directory to save results (optional)")
    parser.add_argument("--batch_dir", default="data/batch_processing", 
                        help="Directory where batch metadata is stored (default: data/batch_processing)")
    parser.add_argument("--check_only", action="store_true", 
                        help="Only check batch status without retrieving results")
    parser.add_argument("--wait", action="store_true", 
                        help="Wait for batch to complete before retrieving results")
    parser.add_argument("--wait_interval", type=int, default=30, 
                        help="Seconds to wait between status checks when --wait is used")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Initialize batch retriever
    retriever = BatchRetriever(batch_dir=args.batch_dir)
    
    # Display batch type being checked
    batch_type = "parent batch" if args.parent else "batch"
    logger.info(f"Checking status for {batch_type} {args.batch_id}...")
    
    # Check status
    status_info = retriever.check_batch_status(args.batch_id, args.parent)
    
    if "error" in status_info:
        logger.error(f"Error checking {batch_type} status: {status_info['error']}")
        return 1
    
    # Log status info
    if args.parent:
        logger.info(f"Parent batch status: {status_info.get('status', 'unknown')}")
        logger.info(f"Completed batches: {status_info.get('completed_batches', 0)}/{status_info.get('total_batches', 0)}")
        if 'processed_items' in status_info:
            logger.info(f"Total processed items: {status_info.get('processed_items', 0)}")
    else:
        logger.info(f"Batch status: {status_info.get('status', 'unknown')}")
        logger.info(f"Completed: {status_info.get('completed', False)}")
        logger.info(f"Created at: {status_info.get('created_at', 'unknown')}")
        logger.info(f"Last checked: {status_info.get('last_checked', 'unknown')}")
        
        # Additional information if available
        if status_info.get("output_file_id"):
            logger.info(f"Output file ID: {status_info.get('output_file_id')}")
        if status_info.get("error_file_id"):
            logger.info(f"Error file ID: {status_info.get('error_file_id')}")
    
    # If only checking status, exit here
    if args.check_only:
        return 0
    
    # Handle the wait flag if batch is not yet completed
    batch_completed = (
        status_info.get("status") == "completed" if args.parent 
        else status_info.get("completed", False)
    )
    
    if not batch_completed and args.wait:
        logger.info(f"Waiting for {batch_type} to complete. Will check every {args.wait_interval} seconds...")
        
        while not batch_completed:
            # Wait before checking again
            time.sleep(args.wait_interval)
            
            # Check status again
            status_info = retriever.check_batch_status(args.batch_id, args.parent)
            
            if "error" in status_info:
                logger.error(f"Error checking {batch_type} status: {status_info['error']}")
                return 1
            
            # Update completion status
            batch_completed = (
                status_info.get("status") == "completed" if args.parent 
                else status_info.get("completed", False)
            )
            
            # Log updated status
            if args.parent:
                logger.info(
                    f"Parent batch status: {status_info.get('status', 'unknown')}. "
                    f"{status_info.get('completed_batches', 0)}/{status_info.get('total_batches', 0)} batches completed."
                )
            else:
                logger.info(f"Batch status: {status_info.get('status', 'unknown')}")
    
    # If batch is not completed and --wait flag is not set, exit
    if not batch_completed and not args.wait:
        logger.info(f"{batch_type.capitalize()} is not yet complete. Use --wait flag to wait for completion or try again later.")
        return 0
    
    # Retrieve results
    logger.info(f"Retrieving results for {batch_type} {args.batch_id}...")
    results_info = retriever.retrieve_batch_results(
        args.batch_id,
        is_parent=args.parent,
        output_dir=args.output_dir
    )
    
    if "error" in results_info:
        logger.error(f"Error retrieving results: {results_info['error']}")
        return 1
    
    # Log results information
    if args.parent:
        logger.info(f"Successfully retrieved {results_info.get('total_results', 0)} results from "
                   f"{results_info.get('successful_batches', 0)}/{results_info.get('total_batches', 0)} batches")
    else:
        logger.info(f"Successfully retrieved {results_info.get('n_results', 0)} results")
    
    logger.info(f"Results saved to: {results_info.get('results_path', 'unknown')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 