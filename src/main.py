"""
Main entry point for the financial knowledge graphs project.

This script can run either the LLM-based entity extraction or the Stanford OpenIE extraction
based on the command-line argument provided.

Usage:
    python -m src.main llm        # Run the LLM-based entity extraction
    python -m src.main openie     # Run the Stanford OpenIE extraction
    python -m src.main evaluate   # Evaluate the latest triplet extraction results
    python -m src.main neo4j      # Run Neo4j database operations
    python -m src.main batch ID   # Retrieve batch results (pass --help for options)
"""

import sys


def main():
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Please specify a task to run: llm, openie, evaluate, neo4j, or batch")
        print("Usage: python -m src.main [llm|openie|evaluate|neo4j|batch]")
        return
    
    task = sys.argv[1].lower()
    
    if task == "llm":
        # Run LLM-based entity extraction
        print("Running LLM-based entity extraction...")
        from src.runners.run_llm_task import main as run_llm
        run_llm()
    
    elif task == "openie":
        # Run Stanford OpenIE extraction
        print("Running Stanford OpenIE extraction...")
        from src.runners.run_stanford_openie import main as run_openie
        run_openie()
    
    elif task == "evaluate":
        # Run evaluation
        print("Evaluating triplet extraction results...")
        from src.runners.run_evaluation import main as run_evaluation
        run_evaluation()
    
    elif task == "neo4j":
        # Run Neo4j database operations
        print("Running Neo4j database operations...")
        from src.runners.run_neo4j_task import main as run_neo4j
        run_neo4j()
    
    elif task == "batch":
        # Run batch retrieval with the remaining arguments
        from src.llm.retrieve_batch import main as run_batch
        
        # Create a new argv list without 'batch' for the retrieve_batch script
        new_argv = [sys.argv[0]] + sys.argv[2:]
        sys.argv = new_argv
        
        if len(sys.argv) < 2:
            # If no batch ID provided, show help message
            print("Please specify a batch ID")
            print("Usage: python -m src.main batch <batch_id> [options]")
            print("Options:")
            print("  --parent             Treat the batch_id as a parent batch ID")
            print("  --output_dir DIR     Directory to save results")
            print("  --check_only         Only check batch status without retrieving results")
            print("  --wait               Wait for batch to complete before retrieving results")
            return
        
        # Run the batch retrieval
        run_batch()
    
    else:
        print(f"Unknown task: {task}")
        print("Please specify a valid task: llm, openie, evaluate, neo4j, or batch")
        print("Usage: python -m src.main [llm|openie|evaluate|neo4j|batch]")


if __name__ == "__main__":
    main()