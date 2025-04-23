"""
Script to evaluate information extraction results against ground truth.

Usage:
    python -m src.run_evaluation
"""

import os
from typing import Optional
from src.utils.file_utils import load_yaml
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.file_utils import find_latest_dir
from src.utils.evaluation import Evaluator

# Initialize logger
logger = get_logger(__name__)

def main():
    """Main function to run information extraction evaluation."""
    # Setup logging
    setup_logging(log_level="INFO")
    logger.info("Starting information extraction evaluation")
    
    try:
        # Load evaluation configuration
        config = load_yaml("configs/config_evaluation.yaml")
        
        # Find the latest run directory or use specified one
        run_dir = config.get('specific_run_dir')
        if not run_dir:
            run_dir = find_latest_dir(
                'runs',
                config.get('test_name', 'test_llm_prompt')
            )
            
        if not run_dir:
            raise FileNotFoundError(f"No test runs found in 'runs'")
            
        logger.info(f"Evaluating run: {run_dir}")
        
        # Get ground truth directory
        gt_dir: str = config.get('ground_truth_dir', "data/ground_truth")
        if not gt_dir:
            raise ValueError("Ground truth directory not specified in config")
        
        # Check for ground truth subdirectory
        gt_subdir: Optional[str] = config.get('ground_truth_subdir')
        if gt_subdir:
            gt_dir = os.path.join(gt_dir, gt_subdir)
            logger.info(f"Using ground truth subdirectory: {gt_subdir}")
        
        logger.info(f"Using ground truth from: {gt_dir}")
        
        # Initialize evaluator with thresholds
        entity_threshold = config.get('entity_similarity_threshold', 80)
        relationship_threshold = config.get('relationship_similarity_threshold', 80)
        logger.info(f"Using thresholds - Entity: {entity_threshold}, Relationship: {relationship_threshold}")
        
        evaluator = Evaluator(entity_threshold, relationship_threshold)
        
        # Run evaluation
        results = evaluator.evaluate_directory(run_dir, gt_dir)
        
        # Print summary
        evaluator.print_summary(results, run_dir, gt_dir)
        
        # Save evaluation results if configured
        if config.get('save_evaluation', True):
            output_dir = config.get('output_dir', 'runs/evaluations')
            os.makedirs(output_dir, exist_ok=True)
            
            results_file = evaluator.save_results(results, run_dir, config, output_dir)
            logger.info(f"Evaluation results saved to: {results_file}")
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 