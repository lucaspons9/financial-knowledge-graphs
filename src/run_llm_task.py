import os
import re
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
from src.utils.reading_files import load_yaml
from src.llm.batch_processor import BatchProcessor


def ensure_dir(directory: str) -> str:
    """Ensure a directory exists and return its path."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def find_next_test_folder(base_dir: str, test_name: str) -> str:
    """
    Determine the next test folder name by finding the latest one and incrementing the index.
    Pattern: test_name_i where i is a number
    """
    # Ensure base directory exists
    ensure_dir(base_dir)
    
    # Look for existing test folders with the pattern: test_name_i
    pattern = os.path.join(base_dir, f"{test_name}_*")
    existing_folders = glob.glob(pattern)
    
    if not existing_folders:
        # No existing folders, start with index 1
        return os.path.join(base_dir, f"{test_name}_1")
    
    # Extract indices from folder names
    indices = []
    for folder in existing_folders:
        match = re.search(rf"{test_name}_(\d+)$", folder)
        if match:
            indices.append(int(match.group(1)))
    
    if not indices:
        # No valid indices found, start with index 1
        return os.path.join(base_dir, f"{test_name}_1")
    
    # Find the next index
    next_index = max(indices) + 1
    return os.path.join(base_dir, f"{test_name}_{next_index}")


def extract_json_from_output(output_content: str) -> List[Dict[str, str]]:
    """Extract JSON data from the model output."""
    try:
        # Look for content between ```json and ``` markers if present
        if "```json" in output_content and "```" in output_content.split("```json")[1]:
            json_str = output_content.split("```json")[1].split("```")[0].strip()
        # Otherwise, try to parse the whole content as JSON
        else:
            json_str = output_content.strip()
        
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from output: {output_content}")
        # Try to handle other formats here if needed
        return []


def save_results(results: Union[List[Dict[str, str]], str], test_dir: str, sentence_id: str) -> str:
    """Save results to a JSON file and return the file path."""
    results_file = os.path.join(test_dir, f"{sentence_id}.json")
    
    # If results is a string, try to parse it as JSON
    if isinstance(results, str):
        try:
            results = extract_json_from_output(results)
        except Exception:
            # If parsing fails, store as is in a dict
            results = {"raw_output": results}
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results_file


def main():
    # Load config
    try:
        config = load_yaml("configs/config_llm_execution.yaml")
        
        # Check if results should be stored
        store_results = config.get("store_results", False)
        
        # Setup test folder if storing results
        test_dir = None
        if store_results:
            results_dir = config.get("results_dir", "runs")
            test_name = config.get("test_name", "test_llm_prompt")
            test_dir = find_next_test_folder(results_dir, test_name)
            ensure_dir(test_dir)
            print(f"Results will be stored in: {test_dir}")
        
        # Load sample texts
        sample_data = load_yaml(config["data_path"])
        
        # Extract text from sample data
        texts = list(sample_data.values())[:1]
        sentence_ids = list(sample_data.keys())[:1]
        
        print(f"Loaded {len(texts)} sample texts")
        
        # Initialize BatchProcessor
        batch_processor = BatchProcessor()
        
        # Run batch processing
        print(f"Running batch {config['task']}...")
        
        # Process each text individually to handle different sentence IDs
        results_files = {}
        for i, (sentence_id, text) in enumerate(zip(sentence_ids, texts), start=1):
            print(f"Processing {sentence_id}...")
            
            # Run the LLM task on the text
            response = batch_processor.run_task(config["task"], text)
            
            # Print the result
            print(f"Results for {sentence_id}: {response.content}")
            
            # Save results if configured
            if store_results and test_dir:
                if config["task"] == "triplet_extraction":
                    # For triplet extraction, parse the JSON output
                    triplets = extract_json_from_output(response.content)
                    results_file = save_results(triplets, test_dir, sentence_id)
                else:
                    # For other tasks, save the raw output
                    results_file = save_results(response.content, test_dir, sentence_id)
                
                results_files[sentence_id] = results_file
                print(f"Saved results to {results_file}")
        
        # Save a summary file if storing results
        if store_results and test_dir:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "config": {k: v for k, v in config.items() if k != "api_key"},
                "results_files": results_files
            }
            summary_file = os.path.join(test_dir, "summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Summary saved to {summary_file}")
            
        print("\nBatch processing completed successfully!")
        
    except Exception as e:
        print(f"Error during batch processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 