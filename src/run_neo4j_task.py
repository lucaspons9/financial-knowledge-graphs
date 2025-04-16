"""
Script for running Neo4j database tasks.

This script sets up a Neo4j database with Docker, creates necessary schema 
and loads entity-relationship data from processed JSON files.
"""

import os
import glob
import time
import subprocess
from typing import Dict, Any

from src.utils.reading_files import load_yaml
from src.utils.logging_utils import setup_logging, get_logger
from src.db.neo4j_handler import Neo4jHandler

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

def main():
    """Main function to run Neo4j database tasks."""
    # Setup logging
    setup_logging(log_level="INFO")
    logger.info("Starting Neo4j database task")
    
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
        
        # Process data files
        if config.get("load_data", True):
            data_path = config.get("data_path", "data/processed/json")
            logger.info(f"Processing data files from {data_path}...")
            
            # Find all JSON files in the data directory
            pattern = os.path.join(data_path, "*.json")
            json_files = glob.glob(pattern)
            
            if not json_files:
                logger.warning(f"No JSON files found in {data_path}")
            else:
                logger.info(f"Found {len(json_files)} JSON files to process")
                
                for json_file in json_files:
                    logger.info(f"Processing {json_file}...")
                    success, message = neo4j_handler.process_json_file(json_file)
                    
                    if success:
                        logger.info(message)
                    else:
                        logger.error(message)
        
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