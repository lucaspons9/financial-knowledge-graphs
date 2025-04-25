"""
Script for running Neo4j database tasks.

This script sets up a Neo4j database with Docker, creates necessary schema 
and loads entity-relationship data from processed execution batches.
"""

import os
import glob
import sys
import time
import subprocess
import json
from typing import Dict, Any, List
from datetime import datetime

from src.utils.file_utils import load_yaml
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.batch_utils import get_execution_path
from src.db.neo4j_handler import Neo4jHandler
from src.llm import DEFAULT_BATCH_DIR

# Initialize logger
logger = get_logger(__name__)

def check_docker_running() -> bool:
    """Check if Docker daemon is running"""
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True)
        return True
    except subprocess.SubprocessError:
        return False

def start_neo4j_docker(config: Dict[str, Any]) -> bool:
    """
    Start a Neo4j Docker container if it's not already running.
    
    Args:
        config: Configuration dictionary with Neo4j settings
        
    Returns:
        bool: True if container is running, False otherwise
    """
    # Check if Docker is running
    if not check_docker_running():
        logger.error("Docker daemon is not running. Please start Docker and try again.")
        return False
    
    # Extract config values with defaults
    container_name = config.get("container_name", "neo4j-fkg")
    neo4j_password = config.get("password", "password")
    neo4j_port = config.get("port", 7687)
    neo4j_browser_port = config.get("browser_port", 7474)
    
    try:
        # Check if container already exists
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=True
        )

        existing_container = container_name in result.stdout
        
        # Check if a different Neo4j container is running (if we can't find our container)
        if not existing_container:
            neo4j_result = subprocess.run(
                ["docker", "ps", "--filter", "ancestor=neo4j", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if neo4j_result.stdout.strip():
                container_name = neo4j_result.stdout.strip().split('\n')[0]
                logger.info(f"Found existing Neo4j container: {container_name}")
                return True
        
        if existing_container:
            # Check if it's running
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if container_name in result.stdout:
                logger.info(f"Neo4j container {container_name} is already running")
                return True
            else:
                # Container exists but not running, start it
                logger.info(f"Starting existing Neo4j container {container_name}")
                subprocess.run(["docker", "start", container_name], check=True)
                time.sleep(10)  # Give it time to start
                return True
        else:
            # Container doesn't exist, create and run it
            logger.info(f"Creating new Neo4j container {container_name}")
            subprocess.run([
                "docker", "run", 
                "--name", container_name,
                "-p", f"{neo4j_port}:7687", 
                "-p", f"{neo4j_browser_port}:7474",
                "-e", f"NEO4J_AUTH=neo4j/{neo4j_password}",
                "-d", "neo4j:latest"
            ], check=True)
            
            # Wait for Neo4j to start
            logger.info("Waiting for Neo4j to start...")
            time.sleep(20)
            return True
            
    except subprocess.SubprocessError as e:
        logger.error(f"Error starting Neo4j Docker container: {str(e)}")
        return False

def process_execution_batches(execution_id: str, config: Dict[str, Any], neo4j_handler: Neo4jHandler, batch_dir: str = DEFAULT_BATCH_DIR) -> Dict[str, Any]:
    """
    Process all batches in an execution directory and load their results into Neo4j.
    
    Args:
        execution_id: The execution ID to process
        config: Neo4j configuration dictionary
        neo4j_handler: Initialized Neo4jHandler
        batch_dir: Base directory for batch processing
        
    Returns:
        Dict containing summary of processing results
    """
    # Get the execution directory path
    execution_dir = get_execution_path(execution_id, batch_dir)
    if not execution_dir:
        logger.error(f"Execution directory not found for ID: {execution_id}")
        return {"error": "Execution directory not found", "status": "failed"}
    
    logger.info(f"Processing batches in execution directory: {execution_dir}")
    
    # Track results
    batches_list: List[Dict[str, Any]] = []
    results: Dict[str, Any] = {
        "execution_id": execution_id,
        "processed_batches": 0,
        "loaded_to_neo4j": 0,
        "already_loaded": 0,
        "skipped": 0,
        "failed": 0,
        "entities_loaded": 0,
        "relationships_loaded": 0,
        "batches": batches_list
    }
    
    # Find all batch directories in the execution directory
    batch_dirs = [d for d in os.listdir(execution_dir) 
                  if os.path.isdir(os.path.join(execution_dir, d)) 
                  and d.startswith('batch_')]
    
    # Process each batch directory
    total_entities = 0
    total_relationships = 0
    
    for batch_dir in batch_dirs:
        batch_path = os.path.join(execution_dir, batch_dir)
        results_path = os.path.join(batch_path, "results")
        metadata_path = os.path.join(batch_path, "metadata.json")

        # Get batch details from metadata if available
        batch_id = batch_dir  # Default to directory name
        metadata = {}
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                batch_id = metadata.get("batch_id", batch_dir)
                
                # Check if this batch has already been processed
                if metadata.get("saved_to_neo4j", False):
                    logger.info(f"Batch {batch_dir} (ID: {batch_id}) already loaded to Neo4j")
                    results["already_loaded"] += 1
                    batches_list.append({
                        "batch_id": batch_id,
                        "status": "already_loaded"
                    })
                    continue
            except Exception as e:
                logger.warning(f"Could not read metadata for batch {batch_dir}: {str(e)}")

        # Check if the batch has been retrieved (results directory exists)
        if not os.path.isdir(results_path):
            logger.warning(f"Results directory not found for batch {batch_dir}. Skipping.")
            results["skipped"] += 1
            batches_list.append({
                "batch_id": batch_id,
                "status": "skipped", 
                "reason": "No results directory"
            })
            continue
            
        try:
            # Find all JSON files in the results directory
            json_files = glob.glob(os.path.join(results_path, "*.json"))
            
            if not json_files:
                logger.warning(f"No JSON files found in results directory for batch {batch_dir}")
                results["skipped"] += 1
                batches_list.append({
                    "batch_id": batch_id,
                    "status": "skipped", 
                    "reason": "No JSON files in results"
                })
                continue
                
            # Process each JSON file
            batch_entities = 0
            batch_relationships = 0
            batch_success = True
            
            for json_file in json_files:
                success, message = neo4j_handler.process_json_file(json_file)
                
                if success:
                    # Extract metrics from the success message
                    import re
                    entities_match = re.search(r'Processed (\d+) entities', message)
                    relationships_match = re.search(r'and (\d+) relationships', message)
                    
                    if entities_match:
                        batch_entities += int(entities_match.group(1))
                    if relationships_match:
                        batch_relationships += int(relationships_match.group(1))
                else:
                    logger.error(f"Failed to process {json_file}: {message}")
                    batch_success = False
            
            # Update counts
            total_entities += batch_entities
            total_relationships += batch_relationships
            
            # Add batch result and update metadata
            if batch_success:
                results["loaded_to_neo4j"] += 1
                batches_list.append({
                    "batch_id": batch_id,
                    "status": "loaded",
                    "entities": batch_entities,
                    "relationships": batch_relationships
                })
                
                # Update metadata to mark as saved to Neo4j
                if os.path.exists(metadata_path):
                    metadata["saved_to_neo4j"] = True
                    metadata["neo4j_saved_at"] = datetime.now().isoformat()
                    metadata["entities_loaded"] = batch_entities
                    metadata["relationships_loaded"] = batch_relationships
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                        
                    logger.info(f"Updated metadata for batch {batch_id} with Neo4j loading status")
            else:
                results["failed"] += 1
                batches_list.append({
                    "batch_id": batch_id,
                    "status": "partially_failed",
                    "entities": batch_entities,
                    "relationships": batch_relationships
                })
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_dir}: {str(e)}")
            results["failed"] += 1
            batches_list.append({
                "batch_id": batch_dir,
                "status": "failed",
                "error": str(e)
            })
    
    # Update total counts
    results["processed_batches"] = len(batch_dirs)
    results["entities_loaded"] = total_entities
    results["relationships_loaded"] = total_relationships
    
    # Print summary
    logger.info(f"Execution {execution_id} processing summary:")
    logger.info(f"Total batches: {results['processed_batches']}")
    logger.info(f"Already loaded to Neo4j: {results['already_loaded']}")
    logger.info(f"Successfully loaded to Neo4j: {results['loaded_to_neo4j']}")
    logger.info(f"Skipped: {results['skipped']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Total entities loaded: {results['entities_loaded']}")
    logger.info(f"Total relationships loaded: {results['relationships_loaded']}")
    
    return results

def main():
    """Main function to run Neo4j database tasks."""
    # Setup logging
    setup_logging(log_level="INFO")
    logger.info("Starting Neo4j database task")
    
    # Check for execution ID argument
    if len(sys.argv) < 2:
        print("Usage: python run_neo4j_task.py <execution_id>")
        print("  execution_id: The ID or name of the execution directory to process")
        sys.exit(1)
    
    # Get execution ID from command line arguments
    execution_id = sys.argv[2]
    batch_dir = DEFAULT_BATCH_DIR
    
    start_time = time.time()
    
    try:
        # Load configuration
        config = load_yaml("configs/config_neo4j.yaml")
        
        # Start Neo4j Docker container
        if not start_neo4j_docker(config):
            logger.error("Failed to start Neo4j Docker container. Exiting...")
            return
            
        # Initialize Neo4j handler
        neo4j_handler = Neo4jHandler(
            uri=f"bolt://localhost:{config.get('port', 7687)}",
            user=config.get('user', 'neo4j'),
            password=config.get('password', 'password')
        )
        
        # Check database actions
        if config.get("clear_database", False):
            logger.info("Clearing database...")
            neo4j_handler.clear_database()
        
        # Create schema constraints
        if config.get("create_schema", True):
            logger.info("Creating schema constraints...")
            neo4j_handler.create_schema_constraints()
        
        # Process batches from the execution
        logger.info(f"Processing batches for execution ID: {execution_id}")
        results = process_execution_batches(execution_id, config, neo4j_handler, batch_dir)
        
        # Check for errors
        if "error" in results:
            logger.error(f"Batch processing failed: {results['error']}")
            sys.exit(1)
        
        # Get database statistics
        try:
            stats = neo4j_handler.get_database_stats()
            logger.info(f"Database Statistics: {stats}")
        except Exception as e:
            logger.warning(f"Could not get database statistics: {str(e)}")
            logger.info("This might be because the APOC plugin is not installed in Neo4j.")
        
        # Close connection
        neo4j_handler.close()
        logger.info("Neo4j database task completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        execution_time = time.time() - start_time
        logger.info(f"Total execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main() 