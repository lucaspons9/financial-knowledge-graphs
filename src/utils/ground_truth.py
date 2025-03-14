"""
Stanford OpenIE Ground Truth Extractor

This module provides a class for extracting ground truth triples from text using Stanford OpenIE.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.utils.reading_files import load_yaml
from openie import StanfordOpenIE


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
        self.config = self._load_config(config_path)
        self.properties = self.config.get("openie_properties", {})
        self.output_dir = self.config.get("output_directory", "data/ground_truth")
        self.generate_graphs = self.config.get("generate_graphs", False)
        self.graphs_dir = os.path.join(self.output_dir, "graphs")
        
        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        if self.generate_graphs:
            os.makedirs(self.graphs_dir, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            Dictionary containing configuration parameters.
        """
        try:
            return load_yaml(config_path)
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            # Return default configuration
            return {
                "openie_properties": {
                    "openie.affinity_probability_cap": 2/3,
                },
                "output_directory": "data/ground_truth",
                "generate_graphs": False
            }
    
    def extract_triples(self, text: str) -> List[Dict[str, str]]:
        """
        Extract triples from text using Stanford OpenIE.
        
        Args:
            text: The text to extract triples from.
            
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
    
    def batch_extract(self, texts: List[str], base_file_name: str = "batch") -> List[List[Dict[str, str]]]:
        """
        Extract triples from a batch of texts.
        
        Args:
            texts: List of texts to extract triples from.
            base_file_name: Base name for the output files.
            
        Returns:
            List of lists of dictionaries containing subject, relation, and object.
        """
        results = []
        
        for i, text in enumerate(texts):
            file_name = f"{base_file_name}_{i+1}.json"
            triples = self.extract_and_save(text, file_name)
            results.append(triples)
        
        # Save batch summary
        summary = {
            "batch_size": len(texts),
            "timestamp": datetime.now().isoformat(),
            "total_triples": sum(len(triples) for triples in results)
        }
        
        summary_path = os.path.join(self.output_dir, f"{base_file_name}_summary.json")
        with open(summary_path, 'w') as file:
            json.dump(summary, file, indent=2)
        
        return results 