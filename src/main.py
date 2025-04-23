"""
Main entry point for the financial knowledge graphs project.

This script can run either the LLM-based entity extraction or the Stanford OpenIE extraction
based on the command-line argument provided.

Usage:
    python -m src.main llm        # Run the LLM-based entity extraction
    python -m src.main openie     # Run the Stanford OpenIE extraction
    python -m src.main evaluate   # Evaluate the latest triplet extraction results
    python -m src.main neo4j      # Run Neo4j database operations
    python -m src.main retrieve execution_id   # Retrieve batch results (pass --help for options)
"""

import sys
from src.utils.logging_utils import setup_logging, get_logger
# Set up logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)

def main():
    # Check command-line arguments
    if len(sys.argv) < 2:
        logger.info("Please specify a task to run: llm, openie, evaluate, neo4j, or batch")
        logger.info("Usage: python -m src.main [llm|openie|evaluate|neo4j|batch]")
        return
    
    task = sys.argv[1].lower()
    
    if task == "llm":
        # Run LLM-based entity extraction
        logger.info("Running LLM-based entity extraction...")
        from src.runners.run_llm_task import main as run_llm
        run_llm()
    
    elif task == "openie":
        # Run Stanford OpenIE extraction
        logger.info("Running Stanford OpenIE extraction...")
        from src.runners.run_stanford_openie import main as run_openie
        run_openie()
    
    elif task == "evaluate":
        # Run evaluation
        logger.info("Evaluating triplet extraction results...")
        from src.runners.run_evaluation import main as run_evaluation
        run_evaluation()
    
    elif task == "neo4j":
        # Run Neo4j database operations
        logger.info("Running Neo4j database operations...")
        from src.runners.run_neo4j_task import main as run_neo4j
        run_neo4j()
    
    elif task == "retrieve":
        # Run batch retrieval with the remaining arguments
        from src.runners.run_retrieve_batch import main as run_retrieve_batch
        
        # Create a new argv list without 'batch' for the retrieve_batch script
        new_argv = [sys.argv[0]] + sys.argv[2:]
        sys.argv = new_argv
        if len(sys.argv) < 2:
            # If no execution ID provided, will use the latest execution
            logger.info("No execution ID provided, using the latest execution directory")
        
        # Run the batch retrieval
        run_retrieve_batch()
    
    else:
        logger.info(f"Unknown task: {task}")
        logger.info("Please specify a valid task: llm, openie, evaluate, neo4j, or batch")
        logger.info("Usage: python -m src.main [llm|openie|evaluate|neo4j|batch]")


if __name__ == "__main__":
    main()