"""
Utility functions for text processing and content extraction.
"""

import json
from typing import List, Dict
from src.utils.logging_utils import get_logger

# Initialize logger
logger = get_logger(__name__)

def extract_json_from_output(output_content: str) -> List[Dict[str, str]]:
    """
    Extract JSON data from the model output.
    
    Args:
        output_content: Raw text output from model that might contain JSON
            
    Returns:
        List[Dict[str, str]]: Extracted JSON data, or empty list if parsing fails
    """
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