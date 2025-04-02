"""
Common file and directory utilities used across the application.
"""

import os
import re
import glob
from pathlib import Path
from typing import Optional, Dict, List, Any, TypeVar, Union
import json
from datetime import datetime

from src.utils.logging_utils import get_logger

# Initialize logger
logger = get_logger(__name__)

# Type variables for better typing
T = TypeVar('T')

def ensure_dir(directory: str) -> str:
    """
    Ensure a directory exists and return its path.
    
    Args:
        directory: Directory path to ensure exists
            
    Returns:
        str: Path to the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)

def find_next_versioned_dir(base_dir: str, prefix: str) -> str:
    """
    Create a versioned directory with incrementing index.
    Pattern: {prefix}_{i} where i is an incrementing number.
    
    Args:
        base_dir: Base directory to create the versioned directory in
        prefix: Prefix for the directory name
    
    Returns:
        str: Path to the newly created directory
    """
    # Ensure base directory exists
    ensure_dir(base_dir)
    
    # Look for existing folders with the pattern: prefix_i
    pattern = os.path.join(base_dir, f"{prefix}_*")
    existing_folders = glob.glob(pattern)
    
    if not existing_folders:
        # No existing folders, start with index 1
        output_dir = os.path.join(base_dir, f"{prefix}_1")
    else:
        # Extract indices from folder names
        indices: List[int] = []
        for folder in existing_folders:
            match = re.search(rf"{prefix}_(\d+)$", folder)
            if match:
                indices.append(int(match.group(1)))
        
        if not indices:
            # No valid indices found, start with index 1
            output_dir = os.path.join(base_dir, f"{prefix}_1")
        else:
            # Find the next index
            next_index = max(indices) + 1
            output_dir = os.path.join(base_dir, f"{prefix}_{next_index}")
    
    # Create the directory
    ensure_dir(output_dir)
    logger.info(f"Created versioned directory: {output_dir}")
    return output_dir

def find_latest_dir(base_dir: str, prefix: str) -> Optional[str]:
    """
    Find the latest (highest index) versioned directory.
    
    Args:
        base_dir: Base directory containing versioned directories
        prefix: Prefix of the versioned directories
    
    Returns:
        Optional[str]: Path to the latest directory, or None if none found
    """
    pattern = os.path.join(base_dir, f"{prefix}_*")
    dirs = glob.glob(pattern)
    
    if not dirs:
        logger.warning(f"No directories found matching pattern {pattern}")
        return None
    
    # Sort by creation time, newest first
    latest_dir = max(dirs, key=os.path.getctime)
    logger.info(f"Found latest directory: {latest_dir}")
    return latest_dir

def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the file to
        indent: JSON indentation level
    """
    # Ensure directory exists
    directory = os.path.dirname(file_path)
    ensure_dir(directory)
    
    # Save the file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)
    
    logger.info(f"Saved JSON file: {file_path}")

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        Dict[str, Any]: The loaded data
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded JSON file: {file_path}")
    return data

def create_run_summary(config: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a summary object for a run.
    
    Args:
        config: Configuration used for the run
        stats: Statistics about the run
    
    Returns:
        Dict[str, Any]: Summary object
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "stats": stats
    }

def extract_run_info(path: str) -> str:
    """
    Extract run information from a path.
    
    Args:
        path: Path to extract information from
    
    Returns:
        str: Extracted run information
    """
    # Extract the last directory name from the path
    run_name = os.path.basename(path)
    return run_name

def load_triplets_from_directory(directory: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Load all triplet JSON files from a directory.
    
    Args:
        directory: Directory containing JSON files with triplets
    
    Returns:
        Dict[str, List[Dict[str, str]]]: Dictionary mapping file names to lists of triplets
    """
    triplets_by_file: Dict[str, List[Dict[str, str]]] = {}
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    for json_file in json_files:
        if os.path.basename(json_file) == "summary.json":
            continue  # Skip summary files
            
        file_id = os.path.splitext(os.path.basename(json_file))[0]
        triplets = load_json(json_file)
        triplets_by_file[file_id] = triplets
    
    logger.info(f"Loaded triplets from {len(triplets_by_file)} files in {directory}")
    return triplets_by_file

def load_evaluation_files(directory: str) -> Dict[str, Dict[str, Any]]:
    """
    Load evaluation files from a directory.
    
    Args:
        directory: Directory containing evaluation JSON files
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping file IDs to their contents
    """
    results: Dict[str, Dict[str, Any]] = {}
    
    if not os.path.exists(directory):
        logger.error(f"Directory {directory} does not exist")
        return results
    
    json_files = [f for f in os.listdir(directory) if f.endswith('.json') and f != 'summary.json']
    logger.info(f"Found {len(json_files)} non-empty JSON files in {directory}")
    
    for file_name in json_files:
        file_path = os.path.join(directory, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data:  # Only add non-empty files
                    results[file_name.replace('.json', '')] = data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            continue
    
    return results

def save_evaluation_results(results: Dict[str, Any], llm_run_path: str, gt_path: str, 
                          config: Dict[str, Any], output_dir: str = "runs/evaluations") -> str:
    """
    Save evaluation results to a file.
    
    Args:
        results: Evaluation results to save
        llm_run_path: Path to the LLM run directory
        gt_path: Path to the ground truth directory
        config: Configuration used for evaluation
        output_dir: Directory to save results in
    
    Returns:
        str: Path to the saved results file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    llm_run_info = os.path.basename(llm_run_path)
    gt_info = os.path.basename(gt_path)
    
    filename = f"eval_{llm_run_info}_vs_{gt_info}_{timestamp}.json"
    file_path = os.path.join(output_dir, filename)
    
    ensure_dir(output_dir)
    save_json(results, file_path)
    logger.info(f"Evaluation results saved to: {file_path}")
    
    return file_path 