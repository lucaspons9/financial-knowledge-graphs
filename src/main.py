"""
Main entry point for the financial knowledge graphs project.

This script can run either the LLM-based entity extraction or the Stanford OpenIE extraction
based on the command-line argument provided.

Usage:
    python -m src.main llm        # Run the LLM-based entity extraction
    python -m src.main openie     # Run the Stanford OpenIE extraction
    python -m src.main evaluate   # Evaluate the latest triplet extraction results
"""

import sys


def main():
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Please specify a task to run: llm, openie, or evaluate")
        print("Usage: python -m src.main [llm|openie|evaluate]")
        return
    
    task = sys.argv[1].lower()
    
    if task == "llm":
        # Run LLM-based entity extraction
        print("Running LLM-based entity extraction...")
        from src.run_llm_task import main as run_llm
        run_llm()
    
    elif task == "openie":
        # Run Stanford OpenIE extraction
        print("Running Stanford OpenIE extraction...")
        from src.run_stanford_openie import main as run_openie
        run_openie()
    
    elif task == "evaluate":
        # Run evaluation
        print("Evaluating triplet extraction results...")
        from src.run_evaluation import main as run_evaluation
        run_evaluation()
    
    else:
        print(f"Unknown task: {task}")
        print("Please specify a valid task: llm, openie or evaluate")
        print("Usage: python -m src.main [llm|openie|evaluate]")


if __name__ == "__main__":
    main()