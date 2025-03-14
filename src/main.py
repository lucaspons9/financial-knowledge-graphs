"""
Main entry point for the financial knowledge graphs project.

This script can run either the LLM-based entity extraction or the Stanford OpenIE extraction
based on the command-line argument provided.

Usage:
    python -m src.main llm        # Run the LLM-based entity extraction
    python -m src.main openie     # Run the Stanford OpenIE extraction
    python -m src.main both       # Run both tasks sequentially
"""

import sys


def main():
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Please specify a task to run: llm, openie, or both")
        print("Usage: python -m src.main [llm|openie|both]")
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
    
    elif task == "both":
        # Run both tasks sequentially
        print("Running both LLM-based entity extraction and Stanford OpenIE extraction...")
        
        # Run LLM-based entity extraction
        print("\n=== LLM-based Entity Extraction ===")
        from src.run_llm_task import main as run_llm
        run_llm()
        
        # Run Stanford OpenIE extraction
        print("\n=== Stanford OpenIE Extraction ===")
        from src.run_stanford_openie import main as run_openie
        run_openie()
    
    else:
        print(f"Unknown task: {task}")
        print("Please specify a valid task: llm, openie, or both")
        print("Usage: python -m src.main [llm|openie|both]")


if __name__ == "__main__":
    main()