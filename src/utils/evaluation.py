"""
Utility functions for evaluating triplet extraction results against ground truth.
"""

from rapidfuzz import fuzz
from typing import List, Dict, Tuple, Any
import os
from datetime import datetime

from src.utils.logging_utils import get_logger
from src.utils.file_utils import load_triplets_from_directory, save_json

# Initialize logger
logger = get_logger(__name__)

class TripletEvaluator:
    """Class for evaluating triplet extraction against ground truth."""
    
    def __init__(self, similarity_threshold: int = 80):
        """
        Initialize the evaluator with a similarity threshold.
        
        Args:
            similarity_threshold: Threshold for fuzzy matching (0-100)
        """
        self.similarity_threshold = similarity_threshold
        logger.info(f"Initialized evaluator with similarity threshold: {similarity_threshold}")
    
    def normalize(self, text: str) -> str:
        """Normalize text for consistent comparison (lowercase, strip spaces)."""
        return text.lower().strip()
    
    def triplet_similarity(self, triplet1: Dict[str, str], triplet2: Dict[str, str]) -> bool:
        """
        Compares two triplets and determines if they are sufficiently similar.
        
        Args:
            triplet1: First triplet as a dictionary with 'subject', 'relation', 'object' keys
            triplet2: Second triplet as a dictionary with 'subject', 'relation', 'object' keys
        
        Returns:
            bool: True if the triplets are considered a match
        """
        h1 = self.normalize(triplet1['subject'])
        r1 = self.normalize(triplet1['relation'])
        t1 = self.normalize(triplet1['object'])
        
        h2 = self.normalize(triplet2['subject'])
        r2 = self.normalize(triplet2['relation'])
        t2 = self.normalize(triplet2['object'])
        
        # Compute similarity scores
        head_score = fuzz.token_sort_ratio(h1, h2)
        relation_score = fuzz.token_sort_ratio(r1, r2)
        tail_score = fuzz.token_sort_ratio(t1, t2)
        
        # Accept the match if the average similarity exceeds the threshold
        avg_score = (head_score + relation_score + tail_score) / 3
        return avg_score >= self.similarity_threshold
    
    def evaluate_predictions(self, predicted: List[Dict[str, str]], 
                           ground_truth: List[Dict[str, str]]) -> Tuple[int, int, int]:
        """
        Evaluates precision and recall with partial matching.
        
        Args:
            predicted: List of predicted triplets (each a dict with 'subject', 'predicate', 'object')
            ground_truth: List of ground truth triplets (same structure as predicted)
        
        Returns:
            Tuple[int, int, int]: True Positives (TP), False Positives (FP), False Negatives (FN)
        """
        matched_truth = set()
        true_positives = 0
        false_positives = 0

        for pred in predicted:
            matched = False
            for i, gt in enumerate(ground_truth):
                if self.triplet_similarity(pred, gt):
                    matched = True
                    matched_truth.add(i)  # Track the index of matched ground truth
                    break
            
            if matched:
                true_positives += 1
            else:
                false_positives += 1

        false_negatives = len(ground_truth) - len(matched_truth)  # Ground truth triplets that were not matched

        return true_positives, false_positives, false_negatives

    def calculate_metrics(self, tp: int, fp: int, fn: int) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score from true positives, false positives, and false negatives.
        
        Args:
            tp: True positives
            fp: False positives
            fn: False negatives
        
        Returns:
            Dict[str, float]: Dictionary containing precision, recall, and F1 score
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }

    def evaluate_directory(self, pred_dir: str, gt_dir: str) -> Dict[str, Any]:
        """
        Evaluate all files in a test run against ground truth.
        
        Args:
            pred_dir: Directory containing prediction files
            gt_dir: Directory containing ground truth files
        
        Returns:
            Dict[str, Any]: Evaluation results including metrics for each file and overall
        """
        # Load predictions and ground truth
        predictions = load_triplets_from_directory(pred_dir)
        ground_truth = load_triplets_from_directory(gt_dir)
        
        # Track metrics for each file and overall
        results = {
            'files': {},
            'overall': {
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0
            }
        }
        
        # Evaluate each file
        for file_id in predictions:
            file_preds = predictions[file_id]
            file_gt = ground_truth.get(file_id, [])
            
            # Evaluate predictions for this file
            tp, fp, fn = self.evaluate_predictions(file_preds, file_gt)
            
            # Update overall metrics
            results['overall']['true_positives'] += tp
            results['overall']['false_positives'] += fp
            results['overall']['false_negatives'] += fn
            
            # Calculate and store metrics for this file
            file_metrics = self.calculate_metrics(tp, fp, fn)
            results['files'][file_id] = file_metrics
        
        # Check for false negatives from files that exist in ground truth but not in predictions
        for file_id in ground_truth:
            if file_id not in predictions:
                fn = len(ground_truth[file_id])
                results['overall']['false_negatives'] += fn
                results['files'][file_id] = self.calculate_metrics(0, 0, fn)
        
        # Calculate overall metrics
        tp = results['overall']['true_positives']
        fp = results['overall']['false_positives']
        fn = results['overall']['false_negatives']
        overall_metrics = self.calculate_metrics(tp, fp, fn)
        
        # Update overall results with calculated metrics
        results['overall'].update(overall_metrics)
        
        return results

    def save_results(self, results: Dict[str, Any], llm_run_path: str, gt_path: str, 
                    config: Dict[str, Any], output_dir: str = "runs/evaluations") -> str:
        """
        Save evaluation results to a file.
        
        Args:
            results: Evaluation results
            llm_run_path: Path to the LLM run being evaluated
            gt_path: Path to the ground truth being compared against
            config: Evaluation configuration
            output_dir: Directory to save results in
        
        Returns:
            str: Path to the saved results file
        """
        # Create timestamp and extract run information
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        llm_run_info = os.path.basename(llm_run_path)
        gt_info = os.path.basename(gt_path)
        
        # Create filename with run information and timestamp
        filename = f"eval_{llm_run_info}_vs_{gt_info}_{timestamp}.json"
        file_path = os.path.join(output_dir, filename)
        
        # Create a clean, compact summary of the results
        compact_results = {
            'timestamp': datetime.now().isoformat(),
            'llm_run': llm_run_info,
            'ground_truth': gt_info,
            'similarity_threshold': config.get('similarity_threshold', self.similarity_threshold),
            'overall': results['overall'],
            'per_file': {
                file_id: {
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'tp': metrics['true_positives'],
                    'fp': metrics['false_positives'],
                    'fn': metrics['false_negatives']
                }
                for file_id, metrics in results['files'].items()
            }
        }
        
        # Save results using file_utils
        save_json(compact_results, file_path)
        logger.info(f"Evaluation results saved to: {file_path}")
        
        return file_path

    def print_summary(self, results: Dict[str, Any], llm_run_path: str, gt_path: str) -> None:
        """
        Print a summary of evaluation results.
        
        Args:
            results: Evaluation results
            llm_run_path: Path to the LLM run being evaluated
            gt_path: Path to the ground truth being compared against
        """
        overall = results['overall']
        llm_run_info = os.path.basename(llm_run_path)
        gt_info = os.path.basename(gt_path)
        
        print("\n===== EVALUATION SUMMARY =====")
        print(f"LLM Run: {llm_run_info}")
        print(f"Ground Truth: {gt_info}")
        print(f"Total files evaluated: {len(results['files'])}")
        print(f"Overall Precision: {overall['precision']:.4f}")
        print(f"Overall Recall: {overall['recall']:.4f}")
        print(f"Overall F1 Score: {overall['f1_score']:.4f}")
        print(f"True Positives: {overall['true_positives']}")
        print(f"False Positives: {overall['false_positives']}")
        print(f"False Negatives: {overall['false_negatives']}")
        print("=============================")
        
        # Print per-file results
        print("\nPer-file results:")
        print("----------------")
        for file_id, metrics in results['files'].items():
            print(f"{file_id}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
