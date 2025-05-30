"""
Common file and directory utilities used across the application.
"""
import yaml
import pandas as pd
import os
import re
import glob
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Tuple, Set
import json
from datetime import datetime

from src.utils.logging_utils import get_logger
from src.utils.text_processing import extract_json_from_output

# Initialize logger
logger = get_logger(__name__)

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

def find_or_create_versioned_dir(base_dir: str, prefix: str, create_new: bool = True, version_format: str = "{prefix}_{index}") -> str:
    """
    Find the latest versioned directory or create a new one.
    
    Args:
        base_dir: Base directory containing versioned directories
        prefix: Prefix of the versioned directories
        create_new: If True, creates a new directory; if False, returns the latest existing one
        version_format: Format string for directory naming (default: "{prefix}_{index}")
        
    Returns:
        str: Path to directory (either newly created or latest existing)
    """
    ensure_dir(base_dir)
    
    # Look for existing folders with the pattern
    pattern = os.path.join(base_dir, f"{prefix}_*")
    existing_folders = glob.glob(pattern)
    
    if not existing_folders:
        if not create_new:
            logger.warning(f"No directories found matching pattern {pattern}")
            return None
        # No existing folders, start with index 1
        next_index = 1
    else:
        # Extract indices from folder names
        indices = []
        for folder in existing_folders:
            match = re.search(rf"{prefix}_(\d+)$", folder)
            if match:
                indices.append(int(match.group(1)))
        
        if not indices:
            next_index = 1
        else:
            next_index = max(indices) + 1
            
        # If we just want the latest directory, return it
        if not create_new:
            latest_dir = max(existing_folders, key=os.path.getctime)
            logger.info(f"Found latest directory: {latest_dir}")
            return latest_dir
    
    # Create the directory using the format string
    dir_name = version_format.format(prefix=prefix, index=next_index)
    dir_path = os.path.join(base_dir, dir_name)
    
    ensure_dir(dir_path)
    logger.info(f"Created versioned directory: {dir_path}")
    return dir_path

# Backwards compatibility functions
def find_next_versioned_dir(base_dir: str, prefix: str) -> str:
    """Backward compatibility wrapper for find_or_create_versioned_dir"""
    return find_or_create_versioned_dir(base_dir, prefix, create_new=True)

def find_latest_dir(base_dir: str, prefix: str) -> Optional[str]:
    """Backward compatibility wrapper for find_or_create_versioned_dir"""
    return find_or_create_versioned_dir(base_dir, prefix, create_new=False)

def save_json(data: Union[Dict[str, Any], List[Any]], file_path: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save (dict or list)
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

def load_json(file_path: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Union[Dict[str, Any], List[Any]]: Loaded JSON data
    """
    if not os.path.exists(file_path):
        logger.warning(f"JSON file not found: {file_path}")
        return {}
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {str(e)}")
        return {}

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
            data = load_json(file_path)
            if data:  # Only add non-empty files
                results[file_name.replace('.json', '')] = data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            continue
    
    return results

def save_evaluation_results(results: Dict[str, Any], llm_run_path: str, 
                            output_dir: str = "runs/evaluations") -> str:
    """
    Save evaluation results to a file.
    
    Args:
        results: Evaluation results to save
        llm_run_path: Path to the LLM run directory
        output_dir: Directory to save results in
    
    Returns:
        str: Path to the saved results file
    """
    llm_run_info = os.path.basename(llm_run_path)
    filename = f"prompt_{llm_run_info}.json"
    file_path = os.path.join(output_dir, filename)
    
    ensure_dir(output_dir)
    save_json(results, file_path)
    logger.info(f"Evaluation results saved to: {file_path}")
    
    return file_path

def save_results(results: Union[List[Dict[str, Any]], str, Dict[str, Any]], test_dir: str, sentence_id: str) -> str:
    """
    Save results to a file. Can handle different input types including raw strings with JSON.
    
    Args:
        results: Results to save (can be a list of triplets, raw string or dictionary)
        test_dir: Directory to save results in
        sentence_id: ID of the sentence/document to use in the filename
        
    Returns:
        str: Path to the saved results file
    """
    file_path = os.path.join(test_dir, f"{sentence_id}.json")
    
    # If results is a string, try to extract JSON if it's in the format
    # that LLMs typically output (with JSON inside markdown code blocks)
    if isinstance(results, str):
        try:
            # Try to extract JSON from a code block in the string
            json_content = extract_json_from_output(results)
            if json_content:
                results = json_content
        except Exception:
            # If extraction fails, keep as string
            results = {"raw_output": results}
    
    # Save the results
    save_json(results, file_path)
    
    return file_path

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

def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dict[str, Any]: YAML file contents
    """
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def load_tabular_data(file_path: str, id_column: str = "newsID", text_column: str = "story") -> Dict[str, str]:
    """
    Load data from tabular files (CSV/Excel).
    
    Args:
        file_path: Path to the tabular file
        id_column: Column name containing IDs
        text_column: Column name containing text content
        
    Returns:
        Dict[str, str]: Dictionary mapping IDs to text content
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        # Read file based on extension
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        # Check if required columns exist
        if id_column not in df.columns:
            raise ValueError(f"File does not contain column '{id_column}'")
        if text_column not in df.columns:
            raise ValueError(f"File does not contain column '{text_column}'")
        
        # Convert to dictionary format {ID: text}
        return {str(row[id_column]): str(row[text_column]) for _, row in df.iterrows()}
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        return {}

# Backwards compatibility functions
def load_csv_news(file_path: str, id_column: str = "newsID", text_column: str = "story") -> Dict[str, str]:
    """Backwards compatibility wrapper for load_tabular_data."""
    return load_tabular_data(file_path, id_column, text_column)

def load_excel_news(file_path: str, id_column: str = "newsID", text_column: str = "story") -> Dict[str, str]:
    """Backwards compatibility wrapper for load_tabular_data."""
    return load_tabular_data(file_path, id_column, text_column)

def load_data_by_extension(data_path: str, id_column: str = "newsID", text_column: str = "story") -> Dict[str, Any]:
    """
    Load data based on file extension.
    
    Args:
        data_path: Path to the data file
        id_column: Column name containing IDs (for CSV/Excel)
        text_column: Column name containing text content (for CSV/Excel)
        
    Returns:
        Dictionary with loaded data
        
    Raises:
        ValueError: If file extension is not supported
    """
    file_extension = os.path.splitext(data_path)[1].lower()
    
    if file_extension in ['.yaml', '.yml']:
        return load_yaml(data_path)
    elif file_extension in ['.csv', '.xlsx', '.xls']:
        return load_tabular_data(data_path, id_column, text_column)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}") 
    
def setup_results_directory(config: Dict[str, Any]) -> Optional[str]:
    """
    Set up results directory if storing results.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Optional[str]: Path to results directory or None if not storing results
    """
    store_results = config.get("store_results", False)
    
    if not store_results:
        return None
        
    results_dir = config.get("results_dir", "runs")
    test_name = config.get("test_name", "test_llm")
    test_dir = find_next_versioned_dir(results_dir, test_name)
    logger.info(f"Results will be stored in: {test_dir}")
    
    return test_dir





