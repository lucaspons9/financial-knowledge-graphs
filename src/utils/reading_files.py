import yaml
import pandas as pd
import os
from typing import Dict, Any
from pathlib import Path

def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load a YAML file and return its contents."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def load_csv_news(file_path: str, id_column: str = "newsID", text_column: str = "story") -> Dict[str, str]:
    """
    Load news texts from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        id_column: Column name containing news article IDs
        text_column: Column name containing news article texts
        
    Returns:
        Dictionary mapping news IDs to their text content
    """
    df = pd.read_csv(file_path)
    
    # Check if required columns exist
    if id_column not in df.columns:
        raise ValueError(f"CSV file does not contain column '{id_column}'")
    if text_column not in df.columns:
        raise ValueError(f"CSV file does not contain column '{text_column}'")
    
    # Convert to dictionary format {newsID: story}
    news_dict = {str(row[id_column]): str(row[text_column]) for _, row in df.iterrows()}
    
    return news_dict

def load_excel_news(file_path: str, id_column: str = "newsID", text_column: str = "story") -> Dict[str, str]:
    """
    Load news texts from an Excel file.
    
    Args:
        file_path: Path to the Excel file
        id_column: Column name containing news article IDs
        text_column: Column name containing news article texts
        
    Returns:
        Dictionary mapping news IDs to their text content
    """
    # Read Excel file
    df = pd.read_excel(file_path)
    
    # Check if required columns exist
    if id_column not in df.columns:
        raise ValueError(f"Excel file does not contain column '{id_column}'")
    if text_column not in df.columns:
        raise ValueError(f"Excel file does not contain column '{text_column}'")
    
    # Convert to dictionary format {newsID: story}
    news_dict = {str(row[id_column]): str(row[text_column]) for _, row in df.iterrows()}
    
    return news_dict

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
    elif file_extension == '.csv':
        return load_csv_news(data_path, id_column, text_column)
    elif file_extension in ['.xlsx', '.xls']:
        return load_excel_news(data_path, id_column, text_column)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}") 