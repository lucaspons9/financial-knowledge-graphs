"""
Utilities for batch processing operations.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime

from src.utils.logging_utils import get_logger
from src.utils.file_utils import ensure_dir, save_json
from src.llm import BATCH_FOLDER_PATTERN, EXECUTION_PREFIX, DEFAULT_BATCH_DIR

# Initialize logger
logger = get_logger(__name__)

def is_batch_folder_name(batch_id: str) -> bool:
    """Check if the provided batch ID looks like a folder name.
    
    Args:
        batch_id: The batch ID to check
        
    Returns:
        bool: True if it looks like a folder name, False otherwise
    """
    # Use pattern from batch_config (support both old and new naming formats)
    if bool(re.match(BATCH_FOLDER_PATTERN, batch_id)):
        return True
    
    # For backwards compatibility, also check old format
    old_pattern = r'^batch_\d{8}_\d{6}_[a-f0-9]{8}$'
    return bool(re.match(old_pattern, batch_id))

def get_execution_path(execution_id: str, batch_dir: str = DEFAULT_BATCH_DIR) -> Optional[str]:
    """
    Get the full path to an execution directory. If no execution ID is provided,
    creates a new execution directory.
    
    Args:
        execution_id: The execution ID or name
        batch_dir: Base directory for batch processing
        
    Returns:
        Optional[str]: Path to the execution directory, or None if not found
    """
    # If no execution ID provided, create a new execution directory
    if not execution_id:
        return create_next_execution_dir(batch_dir)
        
    # Check if this is a full path
    if os.path.isdir(execution_id):
        return execution_id
        
    # Check if this is a direct execution ID in the batch_dir
    direct_path = os.path.join(batch_dir, execution_id)
    if os.path.isdir(direct_path):
        return direct_path
        
    # Check if we need to add the execution prefix
    if not execution_id.startswith(EXECUTION_PREFIX):
        prefixed_path = os.path.join(batch_dir, f"{EXECUTION_PREFIX}{execution_id}")
        if os.path.isdir(prefixed_path):
            return prefixed_path
    
    logger.error(f"Execution directory not found: {execution_id}")
    return None

def get_execution_info(execution_dir: str) -> Dict[str, Any]:
    """
    Get execution information from metadata file.
    
    Args:
        execution_dir: Path to execution directory
        
    Returns:
        Dict[str, Any]: Execution metadata
    """
    metadata_file = "execution_info.json"
    metadata_path = os.path.join(execution_dir, metadata_file)
    
    if not os.path.exists(metadata_path):
        logger.warning(f"Execution metadata file not found: {metadata_path}")
        return {}
    
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading execution metadata: {str(e)}")
        return {}

def find_latest_execution_dir(batch_dir: str = DEFAULT_BATCH_DIR) -> Optional[str]:
    """Find the latest execution directory.
    
    Args:
        batch_dir: Base directory for batch processing
        
    Returns:
        Optional[str]: Path to the latest execution directory, or None if not found
    """
    if not os.path.exists(batch_dir):
        logger.warning(f"Batch directory does not exist: {batch_dir}")
        return None
    
    # Find all execution directories
    execution_dirs = [d for d in os.listdir(batch_dir) 
                      if os.path.isdir(os.path.join(batch_dir, d)) 
                      and d.startswith(EXECUTION_PREFIX)]
    
    if not execution_dirs:
        logger.info(f"No execution directories found in {batch_dir}")
        return None
    
    # Get the most recently created directory
    latest_dir = max(execution_dirs, key=lambda d: os.path.getctime(os.path.join(batch_dir, d)))
    
    return os.path.join(batch_dir, latest_dir)

def create_next_execution_dir(batch_dir: str = DEFAULT_BATCH_DIR) -> str:
    """Create a new versioned execution directory.
    
    Args:
        batch_dir: Base directory for batch processing
        
    Returns:
        str: Path to the newly created execution directory
    """
    # Ensure the base directory exists
    ensure_dir(batch_dir)
    
    # Find execution directories and get their numbers
    execution_dirs = [d for d in os.listdir(batch_dir) 
                      if os.path.isdir(os.path.join(batch_dir, d)) 
                      and d.startswith(EXECUTION_PREFIX)]
    
    if not execution_dirs:
        next_num = 1
    else:
        # Extract numbers from directory names
        nums = []
        for d in execution_dirs:
            match = re.match(f"{EXECUTION_PREFIX}(\\d+)$", d)
            if match:
                nums.append(int(match.group(1)))
        
        next_num = max(nums) + 1 if nums else 1
    
    # Create and return the new directory
    exec_dir_name = f"{EXECUTION_PREFIX}{next_num}"
    exec_dir_path = os.path.join(batch_dir, exec_dir_name)
    
    ensure_dir(exec_dir_path)
    logger.info(f"Created new execution directory: {exec_dir_path}")
    
    # Initialize execution metadata
    exec_metadata = {
        "execution_id": exec_dir_name,
        "created_at": datetime.now().isoformat(),
        "batches": [],
        "processed_item_ids": []
    }
    
    metadata_path = os.path.join(exec_dir_path, "execution_info.json")
    save_json(exec_metadata, metadata_path)
    
    return exec_dir_path

def create_next_batch_dir(execution_dir: str) -> Tuple[str, str]:
    """Create a new batch directory with incremented number.
    
    Args:
        execution_dir: Path to execution directory
        
    Returns:
        Tuple[str, str]: (batch_id, batch_folder_path)
    """
    batch_dirs = [d for d in os.listdir(execution_dir) 
                 if os.path.isdir(os.path.join(execution_dir, d)) 
                 and d.startswith('batch_')]
    
    if not batch_dirs:
        batch_num = 1
    else:
        # Extract numbers from batch directory names
        nums = []
        for d in batch_dirs:
            match = re.match(r'batch_(\d+)$', d)
            if match:
                nums.append(int(match.group(1)))
        
        # If no valid numbered batches, start with 1
        batch_num = max(nums) + 1 if nums else 1
    
    # Create batch directory and return info
    batch_id = f"batch_{batch_num}"
    batch_folder = ensure_dir(os.path.join(execution_dir, batch_id))
    
    return batch_id, batch_folder

def get_real_batch_id(folder_name: str, batch_dir: str) -> Optional[str]:
    """Get the real OpenAI batch ID from a folder's metadata.
    
    Args:
        folder_name: The batch folder name
        batch_dir: The base directory for batch processing
        
    Returns:
        Optional[str]: The real OpenAI batch ID or None if not found
    """
    # Check in batch_dir
    folder_path = os.path.join(batch_dir, folder_name)
    metadata_path = os.path.join(folder_path, "metadata.json")
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get("batch_id")
        except Exception as e:
            logger.error(f"Error reading metadata from {metadata_path}: {str(e)}")
    
    return None

def find_batch_metadata(batch_id: str, batch_dir: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Find metadata file for a specific batch ID.
    
    Args:
        batch_id: Batch ID to find
        batch_dir: Directory to search
        
    Returns:
        tuple: (batch_metadata, batch_folder) or (None, None) if not found
    """
    # Check if there's a direct path match - the batch_id is actually the name of a batch folder
    direct_path = os.path.join(batch_dir, batch_id)
    if os.path.isdir(direct_path) and os.path.exists(os.path.join(direct_path, "metadata.json")):
        metadata_path = os.path.join(direct_path, "metadata.json")
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return metadata, direct_path
        except Exception as e:
            logger.error(f"Error reading metadata from direct path {direct_path}: {str(e)}")
    
    # If no direct folder match, fall back to looking for metadata file with matching batch_id
    logger.info(f"No direct folder match for batch ID {batch_id}, searching through metadata files...")
    for dir_path, _, files in os.walk(batch_dir):
        if "metadata.json" in files:
            metadata_path = os.path.join(dir_path, "metadata.json")
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if metadata.get("batch_id") == batch_id:
                        return metadata, dir_path
            except Exception:
                continue
                
    return None, None

def get_processed_item_ids(execution_dir: str) -> Set[str]:
    """
    Get all processed item IDs (newsIDs) from an execution directory.
    This function first checks execution_info.json for processed_item_ids.
    If not found, it scans through all batch subdirectories and retrieves the keys
    from the 'original_texts' dictionary in each batch's metadata.json file.
    
    Args:
        execution_dir: Path to the execution directory
            
    Returns:
        Set[str]: Set of all processed item IDs across all batches
    """
    if not os.path.exists(execution_dir):
        logger.warning(f"Execution directory not found: {execution_dir}")
        return set()
    
    processed_ids: Set[str] = set()
    
    # First try to get processed_item_ids from execution_info.json
    execution_info_path = os.path.join(execution_dir, "execution_info.json")
    if os.path.exists(execution_info_path):
        try:
            with open(execution_info_path, 'r') as f:
                execution_info = json.load(f)
            
            # Get processed_item_ids from execution_info
            if "processed_item_ids" in execution_info:
                processed_ids = set(execution_info["processed_item_ids"])
                logger.info(f"Found {len(processed_ids)} processed item IDs in execution_info.json")
                return processed_ids
        except Exception as e:
            logger.error(f"Error reading execution_info.json: {str(e)}")
    
    # If execution_info.json doesn't exist or doesn't have processed_item_ids,
    # fall back to scanning batch directories
    logger.info("No processed_item_ids found in execution_info.json, scanning batch directories...")
    
    # Find all batch directories in the execution directory
    batch_dirs = [d for d in os.listdir(execution_dir) 
                  if os.path.isdir(os.path.join(execution_dir, d)) 
                  and d.startswith('batch_')]
    
    for batch_dir in batch_dirs:
        batch_path = os.path.join(execution_dir, batch_dir)
        metadata_path = os.path.join(batch_path, "metadata.json")
        
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file not found for batch: {batch_dir}")
            continue
        
        try:
            # Load metadata and extract keys from original_texts
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Get keys from original_texts (newsIDs)
            original_texts = metadata.get("original_texts", {})
            processed_ids.update(original_texts.keys())
            
        except Exception as e:
            logger.error(f"Error reading metadata from batch {batch_dir}: {str(e)}")
    
    logger.info(f"Found {len(processed_ids)} processed item IDs across all batches in {execution_dir}")
    return processed_ids

def process_batch_results(output_file: str, output_dir: str, original_texts: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Process batch results from an output file.
    
    Args:
        output_file: Path to the batch output file
        output_dir: Directory to save individual results
        original_texts: Mapping from item IDs to original text
        
    Returns:
        List of result information dictionaries
    """
    ensure_dir(output_dir)
    results: List[Dict[str, str]] = []
    
    try:
        # Read the JSONL output file and process each entry
        with open(output_file, 'r') as f:
            for line in f:
                result = json.loads(line)
                item_id = result.get("custom_id")
                
                # Get the model's response content from the response
                if result.get("response") and result["response"].get("choices"):
                    content = result["response"]["choices"][0].get("message", {}).get("content", "")
                else:
                    logger.warning(f"No valid content found for item {item_id}")
                    content = ""
                
                # Skip if item_id not in original texts
                if item_id not in original_texts:
                    continue
                    
                # Get original index from item_id
                index = item_id.replace("item_", "")
                result_file = os.path.join(output_dir, f"result_{index}.json")
                
                # Try to parse JSON from content if available
                try:
                    if "```json" in content and "```" in content.split("```json")[1]:
                        json_str = content.split("```json")[1].split("```")[0].strip()
                        parsed_content = json.loads(json_str)
                    else:
                        parsed_content = json.loads(content.strip())
                except json.JSONDecodeError:
                    parsed_content = {"raw_output": content}
                
                # Save the result
                save_json(parsed_content, result_file)
                
                # Create batch result object and add to results list
                batch_result = {
                    "item_id": item_id,
                    "result_file": result_file
                }
                results.append(batch_result)
                
        return results
                
    except Exception as e:
        logger.error(f"Error processing batch results: {str(e)}")
        return []

def update_execution_metadata(execution_dir: str, batch_id: str, n_items: int, item_ids: Optional[List[str]] = None) -> None:
    """
    Update the execution metadata with new batch information.
    
    Args:
        execution_dir: The execution directory
        batch_id: The ID of the new batch
        n_items: Number of items in the batch
        item_ids: Optional list of processed item IDs
    """
    metadata_file = "execution_info.json"
    
    metadata_path = os.path.join(execution_dir, metadata_file)
    if not os.path.exists(metadata_path):
        logger.warning(f"Execution metadata file not found: {metadata_path}")
        return
        
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Add the new batch to the list
        metadata["batches"].append({
            "batch_id": batch_id,
            "created_at": datetime.now().isoformat(),
            "n_items": n_items
        })
        
        # Update the last updated timestamp
        metadata["last_updated"] = datetime.now().isoformat()
        
        # Update processed item IDs if provided
        if item_ids:
            if "processed_item_ids" not in metadata:
                metadata["processed_item_ids"] = []
            
            # Add unique item IDs to the list
            for item_id in item_ids:
                if item_id not in metadata["processed_item_ids"]:
                    metadata["processed_item_ids"].append(item_id)
        
        # Save the updated metadata
        save_json(metadata, metadata_path)
        logger.info(f"Updated execution metadata with new batch: {batch_id}")
        
    except Exception as e:
        logger.error(f"Error updating execution metadata: {str(e)}")

def resolve_batch_id(batch_id: str, batch_dir: str) -> str:
    """
    Resolve a batch ID from a folder name to the actual OpenAI batch ID if needed.
    
    Args:
        batch_id: The batch ID or folder name to resolve
        batch_dir: Directory where batch metadata is stored
        
    Returns:
        str: The resolved OpenAI batch ID
    """
    logger = get_logger(__name__)
    
    # If the batch ID looks like a folder name, get the real batch ID from metadata
    if is_batch_folder_name(batch_id):
        logger.info(f"Input appears to be a batch folder name: {batch_id}")
        real_batch_id = get_real_batch_id(batch_id, batch_dir)
        
        if real_batch_id:
            logger.info(f"Found real OpenAI batch ID: {real_batch_id}")
            return real_batch_id
        else:
            logger.warning(f"Could not find real batch ID for folder {batch_id}. Will use as-is.")
            
    return batch_id






