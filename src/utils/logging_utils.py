"""
Utility functions for logging setup and configuration.
"""

import logging
import colorlog
from pathlib import Path
from typing import Optional

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Set up logging configuration with colored output for console.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to save logs to
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure color format for console
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # Configure plain format for file
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Setup console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if log_file specified
    if log_file:
        # Ensure directory exists
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    logging.info(f"Logging initialized at {log_level} level")

def get_logger(name: str):
    """
    Get a logger with the specified name.
    
    Args:
        name: Name for the logger
    
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name) 