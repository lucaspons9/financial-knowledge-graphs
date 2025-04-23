"""
Stanford OpenIE Ground Truth Extractor

This module provides a class for extracting ground truth triples from text using Stanford OpenIE.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.utils.file_utils import load_yaml
from src.utils.file_utils import find_next_versioned_dir, save_json, create_run_summary
from src.utils.logging_utils import get_logger
from openie import StanfordOpenIE

# Initialize logger
logger = get_logger(__name__)

class StanfordOpenIEExtractor:
    """
    A class for extracting ground truth triples from text using Stanford OpenIE.
    
    This class provides methods to extract triples from text and save them to files.
    It also supports generating visualization graphs of the extracted triples.
    """
    
    def __init__(self, config_path: str = "configs/config_stanford_openie.yaml"):
        """
        Initialize the StanfordOpenIEExtractor with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        # Load configuration
        self.config = load_yaml(config_path)

        # Setup output configuration
        output_config = self.config.get("output", {})
        self.base_dir = output_config.get("base_dir", "data/ground_truth")
        self.test_name = output_config.get("test_name", "openie_test")
        self.store_results = output_config.get("store_results", True)
        
        # Setup OpenIE properties
        self.properties = self.config.get("openie_properties", {})
        self.generate_graphs = self.config.get("generate_graphs", False)
        
        # Create output directory with versioning first
        if self.store_results:
            self.output_dir = find_next_versioned_dir(self.base_dir, self.test_name)
            logger.info(f"Ground truth results will be stored in: {self.output_dir}")
            
            # Create graphs directory after output_dir is created
            if self.generate_graphs:
                self.graphs_dir = os.path.join(self.output_dir, "graphs")
                os.makedirs(self.graphs_dir, exist_ok=True)
    
    def extract_triples(self, text: str) -> List[Dict[str, str]]:
        """
        Extract triples from text using Stanford OpenIE.
        
        Args:
            text: Text to extract triples from
            
        Returns:
            List of dictionaries containing subject, relation, and object.
        """

        with StanfordOpenIE(properties=self.properties) as client:
            return client.annotate(text)
    
    def extract_and_save(self, text: str, file_name: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Extract triples from text and save them to a file.
        
        Args:
            text: The text to extract triples from.
            file_name: Name of the file to save the triples to. If None, a timestamp will be used.
            
        Returns:
            List of dictionaries containing subject, relation, and object.
        """
        triples = self.extract_triples(text)
        
        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"triples_{timestamp}.json"
        
        file_path = os.path.join(self.output_dir, file_name)
        
        # Save triples to file
        with open(file_path, 'w') as file:
            json.dump(triples, file, indent=2)
        
        # Generate graph if enabled
        if self.generate_graphs and triples:
            graph_file_name = os.path.splitext(file_name)[0] + ".png"
            graph_path = os.path.join(self.graphs_dir, graph_file_name)
            
            with StanfordOpenIE(properties=self.properties) as client:
                client.generate_graphviz_graph(text, graph_path)
        
        return triples
    
    def batch_extract(self, texts: List[str], sentence_ids: Optional[List[str]] = None) -> List[List[Dict[str, str]]]:
        """
        Extract triples from a batch of texts.
        
        Args:
            texts: List of texts to extract triples from
            sentence_ids: List of sentence IDs (used for file naming)
            
        Returns:
            List[List[Dict[str, str]]]: List of lists of extracted triples
        """
        all_triples = []
        results_files = {}        
        
        for i, (sentence_id, text) in enumerate(zip(sentence_ids, texts)):
            # Use sentence_id for file naming
            file_name = f"{sentence_id}.json"
            logger.info(f"Processing {file_name} ({i+1}/{len(texts)})")
            
            try:
                # Extract triples from text
                triples = self.extract_triples(text)
                all_triples.append(triples)
                
                # Save results if configured
                if self.store_results:
                    file_path = os.path.join(self.output_dir, file_name)
                    save_json(triples, file_path)
                    results_files[sentence_id] = file_path
                    
                    logger.info(f"Saved {len(triples)} triples to {file_path}")
                
            except Exception as e:
                logger.error(f"Error processing text {file_name}: {str(e)}")
                all_triples.append([])  # Add empty list for failed extraction
        
        # Save summary if configured
        if self.store_results:
            total_triples = sum(len(triples) for triples in all_triples)
            
            summary = create_run_summary(
                config=self.config,
                stats={
                    "num_texts": len(texts),
                    "num_triples": total_triples,
                    "results_files": results_files
                }
            )
            
            summary_path = os.path.join(self.output_dir, "summary.json")
            save_json(summary, summary_path)
            
            logger.info(f"Saved summary to {summary_path}")
        
        return all_triples 