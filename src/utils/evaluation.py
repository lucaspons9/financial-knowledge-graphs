"""
Utility functions for evaluating information extraction results against ground truth.
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Set, TypeVar, cast, Union

from rapidfuzz import fuzz

from src.utils.logging_utils import get_logger
from src.utils.file_utils import load_evaluation_files, save_evaluation_results

# Initialize logger
logger = get_logger(__name__)

# Type variables for better typing
T = TypeVar('T')

# Type for metrics dictionary values
MetricValue = Union[int, float]

class Evaluator:
    """Class for evaluating structured information extraction."""
    
    def __init__(self, entity_similarity_threshold: int = 80, relationship_similarity_threshold: int = 80):
        """Initialize the evaluator with similarity thresholds."""
        self.entity_similarity_threshold = entity_similarity_threshold
        self.relationship_similarity_threshold = relationship_similarity_threshold
        logger.info(f"Initialized evaluator with entity threshold: {entity_similarity_threshold}, "
                   f"relationship threshold: {relationship_similarity_threshold}")
    
    def normalize(self, text: str) -> str:
        """Normalize text for consistent comparison."""
        return str(text).lower().strip() if text else ""
    
    def calculate_metrics(self, tp: int, fp: int, fn: int) -> Dict[str, Any]:
        """Calculate precision, recall, and F1 score from counts."""
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
    
    def save_results(self, results: Dict[str, Any], llm_run_path: str, gt_path: str, 
                     config: Dict[str, Any], output_dir: str = "runs/evaluations") -> str:
        """Save evaluation results to a file."""
        return save_evaluation_results(results, llm_run_path, gt_path, config, output_dir)
    
    def entity_similarity(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> float:
        """Calculate similarity between two entities."""
        name1 = self.normalize(entity1.get('name', ''))
        name2 = self.normalize(entity2.get('name', ''))
        
        name_similarity = fuzz.token_sort_ratio(name1, name2)
        if name_similarity < 50:
            return name_similarity
        
        attrs1 = entity1.get('attributes', {})
        attrs2 = entity2.get('attributes', {})
        attribute_scores: List[float] = []
        
        for attr in ['companyName', 'ticker', 'industry', 'country']:
            if attr in attrs1 and attr in attrs2:
                val1 = self.normalize(attrs1[attr])
                val2 = self.normalize(attrs2[attr])
                if val1 and val2:
                    attribute_scores.append(fuzz.token_sort_ratio(val1, val2))
        
        if attribute_scores:
            return 0.7 * name_similarity + 0.3 * (sum(attribute_scores) / len(attribute_scores))
        return name_similarity
    
    def _evaluate_entity_attributes(self, pred_attrs: Dict[str, Any], 
                                   gt_attrs: Dict[str, Any],
                                   metrics: Dict[str, Dict[str, MetricValue]]) -> None:
        """Evaluate entity attribute accuracy."""
        for attr in ['companyName', 'ticker', 'industry', 'country']:
            if attr in gt_attrs and gt_attrs[attr]:
                metrics[attr]['total'] += 1
                if attr in pred_attrs and pred_attrs[attr]:
                    pred_val = self.normalize(pred_attrs[attr])
                    gt_val = self.normalize(gt_attrs[attr])
                    if fuzz.token_sort_ratio(pred_val, gt_val) >= self.entity_similarity_threshold:
                        metrics[attr]['correct'] += 1
    
    def evaluate_entities(self, predicted_entities: List[Dict[str, Any]], 
                         ground_truth_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate entity extraction against ground truth."""
        matched_truth_indices: Set[int] = set()
        entity_mappings: Dict[str, str] = {}
        true_positives = 0
        false_positives = 0
        
        attribute_metrics: Dict[str, Dict[str, MetricValue]] = {
            'companyName': {'correct': 0, 'total': 0},
            'ticker': {'correct': 0, 'total': 0},
            'industry': {'correct': 0, 'total': 0},
            'country': {'correct': 0, 'total': 0},
        }
        
        for pred in predicted_entities:
            matched = False
            best_match_idx = None
            best_match_score = 0.0
            
            for i, gt in enumerate(ground_truth_entities):
                if i in matched_truth_indices:
                    continue
                
                similarity = self.entity_similarity(pred, gt)
                if similarity > best_match_score:
                    best_match_score = similarity
                    best_match_idx = i
            
            if best_match_score >= self.entity_similarity_threshold and best_match_idx is not None:
                matched = True
                matched_truth_indices.add(best_match_idx)
                entity_mappings[pred.get('id', '')] = ground_truth_entities[best_match_idx].get('id', '')
                self._evaluate_entity_attributes(
                    pred.get('attributes', {}),
                    ground_truth_entities[best_match_idx].get('attributes', {}),
                    attribute_metrics
                )
            
            if matched:
                true_positives += 1
            else:
                false_positives += 1
        
        false_negatives = len(ground_truth_entities) - len(matched_truth_indices)
        metrics = self.calculate_metrics(true_positives, false_positives, false_negatives)
        
        for attr, attr_metrics in attribute_metrics.items():
            attr_metrics['accuracy'] = attr_metrics['correct'] / attr_metrics['total'] if attr_metrics['total'] > 0 else 0
        
        result_metrics = cast(Dict[str, Any], metrics)
        result_metrics['entity_mappings'] = entity_mappings
        result_metrics['attribute_metrics'] = attribute_metrics
        
        return result_metrics
    
    def relationship_match(self, pred_rel: Dict[str, Any], gt_rel: Dict[str, Any], 
                          entity_mappings: Dict[str, str]) -> bool:
        """Determine if a predicted relationship matches a ground truth relationship."""
        pred_type = self.normalize(pred_rel.get('type', ''))
        gt_type = self.normalize(gt_rel.get('type', ''))
        
        if fuzz.ratio(pred_type, gt_type) < self.relationship_similarity_threshold:
            return False
        
        pred_source = entity_mappings.get(pred_rel.get('source', ''), '')
        pred_target = entity_mappings.get(pred_rel.get('target', ''), '')
        gt_source = gt_rel.get('source', '')
        gt_target = gt_rel.get('target', '')
        
        if pred_type in ['mergedwith', 'partnerswith']:  # Symmetric relationships
            return ((pred_source == gt_source and pred_target == gt_target) or
                    (pred_source == gt_target and pred_target == gt_source))
        return pred_source == gt_source and pred_target == gt_target
    
    def _evaluate_relationship_attributes(self, pred_attrs: Dict[str, Any], 
                                        gt_attrs: Dict[str, Any],
                                        metrics: Dict[str, Dict[str, MetricValue]]) -> None:
        """Evaluate relationship attribute accuracy."""
        for attr in ['valueAmount', 'percentage']:
            if attr in gt_attrs and gt_attrs[attr] is not None:
                metrics[attr]['total'] += 1
                if attr in pred_attrs and pred_attrs[attr] is not None:
                    try:
                        pred_val = float(pred_attrs[attr])
                        gt_val = float(gt_attrs[attr])
                        if abs(pred_val - gt_val) / max(1, abs(gt_val)) <= 0.1:
                            metrics[attr]['correct'] += 1
                    except (ValueError, TypeError):
                        pass
        
        if 'transactionDate' in gt_attrs and gt_attrs['transactionDate']:
            metrics['transactionDate']['total'] += 1
            if 'transactionDate' in pred_attrs and pred_attrs['transactionDate']:
                pred_date = self.normalize(pred_attrs['transactionDate'])
                gt_date = self.normalize(gt_attrs['transactionDate'])
                if fuzz.ratio(pred_date, gt_date) >= self.relationship_similarity_threshold:
                    metrics['transactionDate']['correct'] += 1
    
    def evaluate_relationships(self, predicted_rels: List[Dict[str, Any]], 
                             ground_truth_rels: List[Dict[str, Any]],
                             entity_mappings: Dict[str, str]) -> Dict[str, Any]:
        """Evaluate relationship extraction against ground truth."""
        matched_truth_indices: Set[int] = set()
        true_positives = 0
        false_positives = 0
        
        attribute_metrics: Dict[str, Dict[str, MetricValue]] = {
            'valueAmount': {'correct': 0, 'total': 0},
            'percentage': {'correct': 0, 'total': 0},
            'transactionDate': {'correct': 0, 'total': 0},
        }
        
        for pred in predicted_rels:
            matched = False
            best_match_idx = None
            
            for i, gt in enumerate(ground_truth_rels):
                if i in matched_truth_indices:
                    continue
                
                if self.relationship_match(pred, gt, entity_mappings):
                    matched = True
                    best_match_idx = i
                    matched_truth_indices.add(i)
                    break
            
            if matched and best_match_idx is not None:
                true_positives += 1
                self._evaluate_relationship_attributes(
                    pred.get('attributes', {}),
                    ground_truth_rels[best_match_idx].get('attributes', {}),
                    attribute_metrics
                )
            else:
                false_positives += 1
        
        false_negatives = len(ground_truth_rels) - len(matched_truth_indices)
        metrics = self.calculate_metrics(true_positives, false_positives, false_negatives)
        
        for attr, attr_metrics in attribute_metrics.items():
            attr_metrics['accuracy'] = attr_metrics['correct'] / attr_metrics['total'] if attr_metrics['total'] > 0 else 0
        
        result_metrics = cast(Dict[str, Any], metrics)
        result_metrics['attribute_metrics'] = attribute_metrics
        
        return result_metrics
    
    def evaluate(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a prediction against ground truth."""
        entity_results = self.evaluate_entities(
            prediction.get('entities', []),
            ground_truth.get('entities', [])
        )
        
        relationship_results = self.evaluate_relationships(
            prediction.get('relationships', []),
            ground_truth.get('relationships', []),
            entity_results['entity_mappings']
        )
        
        overall_f1 = (entity_results['f1_score'] + relationship_results['f1_score']) / 2
        
        return {
            'entity_evaluation': entity_results,
            'relationship_evaluation': relationship_results,
            'overall': {
                'f1_score': overall_f1,
                'entity_f1': entity_results['f1_score'],
                'relationship_f1': relationship_results['f1_score']
            }
        }
    
    def evaluate_directory(self, pred_dir: str, gt_dir: str) -> Dict[str, Any]:
        """Evaluate all files in a test run against ground truth."""
        predictions = load_evaluation_files(pred_dir)
        ground_truth = load_evaluation_files(gt_dir)
        
        if not predictions:
            logger.warning(f"No valid prediction files found in {pred_dir}")
        if not ground_truth:
            logger.warning(f"No valid ground truth files found in {gt_dir}")
        
        results: Dict[str, Any] = {
            'files': {},
            'overall': {
                'entity': {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0},
                'relationship': {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}
            }
        }
        
        for file_id in predictions:
            file_pred = predictions[file_id]
            file_gt = ground_truth.get(file_id)
            
            if file_gt:
                file_results = self.evaluate(file_pred, file_gt)
                results['files'][file_id] = file_results
                
                for level in ['entity', 'relationship']:
                    results['overall'][level]['true_positives'] += file_results[f'{level}_evaluation']['true_positives']
                    results['overall'][level]['false_positives'] += file_results[f'{level}_evaluation']['false_positives']
                    results['overall'][level]['false_negatives'] += file_results[f'{level}_evaluation']['false_negatives']
            else:
                logger.warning(f"No ground truth found for file {file_id}")
        
        for file_id in ground_truth:
            if file_id not in predictions:
                gt_data = ground_truth[file_id]
                results['overall']['entity']['false_negatives'] += len(gt_data.get('entities', []))
                results['overall']['relationship']['false_negatives'] += len(gt_data.get('relationships', []))
                logger.warning(f"Missing prediction for file {file_id}")
        
        for level in ['entity', 'relationship']:
            tp = results['overall'][level]['true_positives']
            fp = results['overall'][level]['false_positives']
            fn = results['overall'][level]['false_negatives']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results['overall'][level]['precision'] = precision
            results['overall'][level]['recall'] = recall
            results['overall'][level]['f1_score'] = f1
        
        results['overall']['f1_score'] = (
            results['overall']['entity']['f1_score'] + 
            results['overall']['relationship']['f1_score']
        ) / 2
        
        return results
    
    def print_summary(self, results: Dict[str, Any], llm_run_path: str, gt_path: str) -> None:
        """Print a summary of evaluation results."""
        overall = results['overall']
        llm_run_info = os.path.basename(llm_run_path)
        gt_info = os.path.basename(gt_path)
        
        print("\n===== EVALUATION SUMMARY =====")
        print(f"LLM Run: {llm_run_info}")
        print(f"Ground Truth: {gt_info}")
        print(f"Total files evaluated: {len(results.get('files', {}))}")
        print("\nEntity Extraction:")
        print(f"  Precision: {overall['entity']['precision']:.4f}")
        print(f"  Recall: {overall['entity']['recall']:.4f}")
        print(f"  F1 Score: {overall['entity']['f1_score']:.4f}")
        
        print("\nRelationship Extraction:")
        print(f"  Precision: {overall['relationship']['precision']:.4f}")
        print(f"  Recall: {overall['relationship']['recall']:.4f}")
        print(f"  F1 Score: {overall['relationship']['f1_score']:.4f}")
        
        print(f"\nOverall F1 Score: {overall['f1_score']:.4f}")
        print("============================")
