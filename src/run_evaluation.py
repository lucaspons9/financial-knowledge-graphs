"""
Script to evaluate triplet extraction results against ground truth.

Usage:
    python -m src.main evaluate
"""

from src.utils.reading_files import load_yaml
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.file_utils import find_latest_dir
from src.utils.evaluation import TripletEvaluator

# Initialize logger
logger = get_logger(__name__)

def main():
    """Main function to run triplet extraction evaluation."""
    # Setup logging first
    setup_logging(log_level="INFO")
    logger.info("Starting triplet extraction evaluation")
    
    try:
        # Load evaluation configuration
        config = load_yaml("configs/config_evaluation.yaml")
        
        # Get similarity threshold from config
        similarity_threshold = config.get('similarity_threshold', 80)
        logger.info(f"Using similarity threshold: {similarity_threshold}")
        
        # Find the latest run directory
        latest_run = find_latest_dir(
            config.get('results_dir', 'runs'),
            config.get('test_name', 'test_llm_prompt')
        )
        
        if not latest_run:
            raise FileNotFoundError(f"No test runs found in {config.get('results_dir', 'runs')}")
            
        logger.info(f"Evaluating latest run: {latest_run}")
        
        # Get ground truth directory
        gt_dir = config.get('ground_truth_dir')
        logger.info(f"Using ground truth from: {gt_dir}")
        
        # Create evaluator instance
        evaluator = TripletEvaluator(similarity_threshold)
        
        # Evaluate the run with configured threshold
        results = evaluator.evaluate_directory(latest_run, gt_dir)
        
        # Print summary
        evaluator.print_summary(results, latest_run, gt_dir)
        
        # Save evaluation results if configured
        if config.get('save_evaluation', True):
            results_file = evaluator.save_results(results, latest_run, gt_dir, config)
            logger.info(f"Evaluation results saved to: {results_file}")
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 