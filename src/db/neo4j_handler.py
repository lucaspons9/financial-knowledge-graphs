from neo4j import GraphDatabase
import json
from typing import Dict, Any, Tuple
import os

class Neo4jHandler:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        """Connect to Neo4j database"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        """Close the Neo4j connection"""
        self.driver.close()

    def create_schema_constraints(self) -> None:
        """Create schema constraints for the database"""
        with self.driver.session() as session:
            # Create constraint on Entity.id
            session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            # Create constraint on Company.id
            session.run("CREATE CONSTRAINT company_id IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE")

    def insert_entity(self, entity_data: Dict[str, Any]) -> None:
        """Insert an entity with all its attributes"""
        with self.driver.session() as session:
            # Extract entity attributes
            entity_id = entity_data.get("id", "")
            entity_type = entity_data.get("type", "Entity")
            entity_name = entity_data.get("name", "")
            attributes = entity_data.get("attributes", {})
            
            # Convert all attributes to string representation for Neo4j
            attr_params = {k: str(v) for k, v in attributes.items()}
            attr_params["id"] = entity_id
            attr_params["name"] = entity_name
            attr_params["type"] = entity_type  # Store entity type as a property instead of a label
            
            # Create the entity with only Entity label and all attributes
            query1 = f"""
            MERGE (e:Entity {{id: $id}})
            RETURN e
            """
            
            session.run(query1, {"id": entity_id})
            
            # Add or update all other attributes
            for attr_key, attr_value in attr_params.items():
                if attr_key != "id":  # Skip the id as it's already set in the MERGE
                    query2 = f"""
                    MATCH (e:Entity {{id: $id}})
                    SET e.{attr_key} = $value
                    RETURN e
                    """
                    session.run(query2, {"id": entity_id, "value": attr_value})

    def insert_relationship(self, relationship_data: Dict[str, Any]) -> None:
        """Insert a relationship with all its attributes"""
        with self.driver.session() as session:
            # Extract relationship data
            rel_id = relationship_data.get("id", "")
            rel_type = relationship_data.get("type", "").upper()
            source_id = relationship_data.get("source", "")
            target_id = relationship_data.get("target", "")
            attributes = relationship_data.get("attributes", {})
            
            # Create the relationship first
            query1 = f"""
            MATCH (source:Entity {{id: $source_id}}), (target:Entity {{id: $target_id}})
            MERGE (source)-[r:{rel_type}]->(target)
            RETURN r
            """
            
            session.run(query1, {"source_id": source_id, "target_id": target_id})
            
            # Then set all attributes individually
            for key, value in attributes.items():
                query2 = f"""
                MATCH (source:Entity {{id: $source_id}})-[r:{rel_type}]->(target:Entity {{id: $target_id}})
                SET r.{key} = $value
                """
                session.run(query2, {"source_id": source_id, "target_id": target_id, "value": str(value)})
            
            # Set the id attribute separately
            if rel_id:
                query3 = f"""
                MATCH (source:Entity {{id: $source_id}})-[r:{rel_type}]->(target:Entity {{id: $target_id}})
                SET r.id = $rel_id
                """
                session.run(query3, {"source_id": source_id, "target_id": target_id, "rel_id": rel_id})

    def process_json_file(self, json_file_path: str) -> Tuple[bool, str]:
        """Process a JSON file with entities and relationships"""
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
                
            # Extract filename without path and extension to use as prefix
            filename = os.path.basename(json_file_path)
            filename = os.path.splitext(filename)[0]
            
            # Process entities
            entities = data.get("entities", [])
            entity_id_mapping = {}  # Map original IDs to prefixed IDs
            
            for entity in entities:
                # Create a globally unique ID by prefixing with filename
                original_id = entity.get("id", "")
                entity["id"] = f"{filename}_{original_id}"
                
                # Store mapping for relationship processing
                entity_id_mapping[original_id] = entity["id"]
                
                # Remove original_id if it exists to avoid duplication
                if "original_id" in entity:
                    del entity["original_id"]
                    
                self.insert_entity(entity)
                
            # Process relationships
            relationships = data.get("relationships", [])
            for relationship in relationships:
                # Create a globally unique ID by prefixing with filename
                original_id = relationship.get("id", "")
                relationship["id"] = f"{filename}_{original_id}"
                
                # Update source and target references using the mapping
                original_source = relationship.get("source", "")
                original_target = relationship.get("target", "")
                
                relationship["source"] = entity_id_mapping.get(original_source, f"{filename}_{original_source}")
                relationship["target"] = entity_id_mapping.get(original_target, f"{filename}_{original_target}")
                
                self.insert_relationship(relationship)
                
            return True, f"Successfully processed {len(entities)} entities and {len(relationships)} relationships from {json_file_path}"
        except Exception as e:
            return False, f"Error processing {json_file_path}: {str(e)}"

    def clear_database(self) -> str:
        """Clear all data in the database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            return "Database cleared"

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database"""
        with self.driver.session() as session:
            # Count nodes by label
            node_query = """
            MATCH (n)
            RETURN labels(n) as label, count(n) as count
            """
            node_result = session.run(node_query)
            
            # Count relationships by type
            rel_query = """
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
            """
            rel_result = session.run(rel_query)
            
            stats = {
                "nodes": {},
                "relationships": {}
            }
            
            # Process node counts
            for record in node_result:
                label_list = record["label"]
                count = record["count"]
                # Use the last label in the list (most specific)
                if label_list:
                    for label in label_list:
                        if label not in stats["nodes"]:
                            stats["nodes"][label] = 0
                        stats["nodes"][label] += count
            
            # Process relationship counts
            for record in rel_result:
                rel_type = record["type"]
                count = record["count"]
                stats["relationships"][rel_type] = count
                
            return stats

# Example Usage
if __name__ == "__main__":
    neo4j_handler = Neo4jHandler()
    
    # Create schema constraints
    neo4j_handler.create_schema_constraints()
    
    # Process a sample JSON file
    success, message = neo4j_handler.process_json_file("data/ground_truth/sample.json")
    print(message)
    
    # Get database statistics
    stats = neo4j_handler.get_database_stats()
    print("Database Statistics:", stats)
    
    neo4j_handler.close()