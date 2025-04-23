from neo4j import GraphDatabase, Query
import json
from typing import Dict, Any, Tuple, Optional
import os
import re
import logging

# Get logger
logger = logging.getLogger(__name__)

logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)  # or logging.ERROR

class Neo4jHandler:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        """Connect to Neo4j database"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Track disambiguation statistics
        self.disambiguation_count = 0

    def close(self) -> None:
        """Close the Neo4j connection"""
        self.driver.close()
        if self.disambiguation_count > 0:
            logger.warning(f"🔍 DISAMBIGUATION SUMMARY: {self.disambiguation_count} entities were disambiguated")

    def create_schema_constraints(self) -> None:
        """Create schema constraints for the database"""
        with self.driver.session() as session:
            # Create constraint on Entity.id
            session.run(Query(r"CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE"))
            # Create constraint on Company.id
            session.run(Query(r"CREATE CONSTRAINT company_id IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE"))
            # Create index on Entity.name for faster lookups during disambiguation
            session.run(Query(r"CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)"))

    def _normalize_entity_name(self, name: str) -> str:
        """
        Normalize entity name for better matching by removing special characters,
        converting to lowercase, and removing common words.
        
        Args:
            name: Entity name to normalize
            
        Returns:
            Normalized entity name
        """
        if not name:
            return ""
            
        # Convert to lowercase
        name = name.lower()
        
        # Remove common legal entity suffixes
        common_suffixes = [
            "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
            "company", "co", "group", "holdings", "plc", "ag", "gmbh", "sa", "nv", "bv"
        ]
        
        # First remove suffixes with dots and commas
        name = re.sub(r'[\s,]+(inc|corp|co|ltd|llc)\.?$', '', name)
        
        # Replace special characters with spaces
        name = re.sub(r'[^\w\s]', ' ', name)
        
        # Split into words and filter out common suffixes and short words
        words = [word for word in name.split() if word not in common_suffixes and len(word) > 1]
        
        return " ".join(words)

    def find_matching_entity(self, entity_data: Dict[str, Any]) -> Optional[str]:
        """
        Find if an entity already exists in the database based on name and other attributes.
        
        Args:
            entity_data: Dictionary containing entity attributes
            
        Returns:
            The ID of the matching entity if found, None otherwise
        """
        with self.driver.session() as session:
            entity_name = entity_data.get("name", "")
            entity_type = entity_data.get("type", "Entity")
            
            if not entity_name:
                return None
                
            # First try exact name match with same type
            query = Query(r"""
            MATCH (e:Entity)
            WHERE e.name = $name AND e.type = $type
            RETURN e.id AS id
            """)
            
            result = session.run(query, {"name": entity_name, "type": entity_type})
            records = list(result)
            
            if records:
                return records[0]["id"]
                
            # If no exact match found, try normalized name matching
            normalized_name = self._normalize_entity_name(entity_name)
            
            if len(normalized_name) > 2:  # Only attempt fuzzy matching for names with sufficient content
                # Find entities with similar normalized names
                query = Query(r"""
                MATCH (e:Entity)
                WHERE e.type = $type
                RETURN e.id AS id, e.name AS name
                """)
                
                result = session.run(query, {"type": entity_type})
                candidates = list(result)
                
                # Find the best match among candidates
                best_match_id = None
                best_similarity = 0
                
                for candidate in candidates:
                    candidate_name = candidate["name"]
                    candidate_normalized = self._normalize_entity_name(candidate_name)
                    
                    # Simple containment check for now - either name contains the other
                    if (normalized_name in candidate_normalized and len(normalized_name) > 3) or \
                       (candidate_normalized in normalized_name and len(candidate_normalized) > 3):
                        # For multiple matches, prefer the most similar one
                        similarity = self._calculate_similarity(normalized_name, candidate_normalized)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_id = candidate["id"]
                
                if best_match_id and best_similarity > 0.5:  # Only return if similarity is above threshold
                    return best_match_id
                    
            return None
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate string similarity between normalized entity names.
        Simple implementation - can be replaced with more sophisticated algorithms.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Convert to sets of words for comparison
        set1 = set(str1.split())
        set2 = set(str2.split())
        
        # Calculate Jaccard similarity
        if not set1 or not set2:
            return 0
            
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0

    def insert_entity(self, entity_data: Dict[str, Any]) -> str:
        """
        Insert an entity with all its attributes or update if it already exists.
        
        Args:
            entity_data: Dictionary containing entity data
            
        Returns:
            The ID of the entity (either existing or newly created)
        """
        # First check if a matching entity already exists
        existing_id = self.find_matching_entity(entity_data)
            
        # If found, use the existing entity ID
        if existing_id:
            entity_id = existing_id
            self._update_entity_attributes(entity_id, entity_data)
            
            # Log disambiguation event with special marker
            self.disambiguation_count += 1
            logger.warning(f"🔍 DISAMBIGUATED: '{entity_data.get('name', '')}' matched to existing entity (ID: {entity_id})")
        else:
            # Create a new entity
            entity_id = self._create_new_entity(entity_data)
        
        return entity_id
        
    def _update_entity_attributes(self, entity_id: str, entity_data: Dict[str, Any]) -> None:
        """
        Update attributes of an existing entity.
        
        Args:
            entity_id: ID of the entity to update
            entity_data: Dictionary containing entity data with new attributes
        """
        with self.driver.session() as session:
            # Update attributes of the existing entity
            attributes = entity_data.get("attributes", {})
            attr_params = {k: str(v) for k, v in attributes.items()}
            
            # Update basic entity attributes using parametrized queries
            for attr_key, attr_value in attr_params.items():
                # Use raw string for Query to satisfy LiteralString requirement
                session.run(
                    Query(r"MATCH (e:Entity {id: $id}) SET e." + attr_key + r" = $value RETURN e"),
                    {"id": entity_id, "value": attr_value}
                )
            
            # Ensure the name is updated if provided
            if "name" in entity_data and entity_data["name"]:
                query = Query(r"""
                MATCH (e:Entity {id: $id})
                SET e.name = $name
                RETURN e
                """)
                session.run(query, {"id": entity_id, "name": entity_data["name"]})
    
    def _create_new_entity(self, entity_data: Dict[str, Any]) -> str:
        """
        Create a new entity with all its attributes.
        
        Args:
            entity_data: Dictionary containing entity data
            
        Returns:
            The ID of the newly created entity
        """
        entity_id = entity_data.get("id", "")
        entity_type = entity_data.get("type", "Entity")
        entity_name = entity_data.get("name", "")
        attributes = entity_data.get("attributes", {})
        
        # Convert all attributes to string representation for Neo4j
        attr_params = {k: str(v) for k, v in attributes.items()}
        attr_params["id"] = entity_id
        attr_params["name"] = entity_name
        attr_params["type"] = entity_type
        
        with self.driver.session() as session:
            # Create the entity
            query = Query(r"""
            MERGE (e:Entity {id: $id})
            RETURN e
            """)
            
            session.run(query, {"id": entity_id})
            
            # Add or update all other attributes
            for attr_key, attr_value in attr_params.items():
                if attr_key != "id":
                    # Use raw string for Query to satisfy LiteralString requirement
                    session.run(
                        Query(r"MATCH (e:Entity {id: $id}) SET e." + attr_key + r" = $value RETURN e"),
                        {"id": entity_id, "value": attr_value}
                    )
        
        return entity_id

    def insert_relationship(self, relationship_data: Dict[str, Any]) -> None:
        """Insert a relationship with all its attributes"""
        with self.driver.session() as session:
            # Extract relationship data
            rel_id = relationship_data.get("id", "")
            rel_type = relationship_data.get("type", "").upper()
            source_id = relationship_data.get("source", "")
            target_id = relationship_data.get("target", "")
            attributes = relationship_data.get("attributes", {})
            
            # Create the relationship first - use raw string parts to satisfy linter
            session.run(
                Query(r"MATCH (source:Entity {id: $source_id}), (target:Entity {id: $target_id}) " + 
                      r"MERGE (source)-[r:" + rel_type + r"]->(target) RETURN r"),
                {"source_id": source_id, "target_id": target_id}
            )
            
            # Then set all attributes individually
            for key, value in attributes.items():
                session.run(
                    Query(r"MATCH (source:Entity {id: $source_id})-[r:" + rel_type + 
                          r"]->(target:Entity {id: $target_id}) SET r." + key + r" = $value"),
                    {"source_id": source_id, "target_id": target_id, "value": str(value)}
                )
            
            # Set the id attribute separately
            if rel_id:
                session.run(
                    Query(r"MATCH (source:Entity {id: $source_id})-[r:" + rel_type + 
                          r"]->(target:Entity {id: $target_id}) SET r.id = $rel_id"),
                    {"source_id": source_id, "target_id": target_id, "rel_id": rel_id}
                )

    def process_json_file(self, json_file_path: str) -> Tuple[bool, str]:
        """
        Process a JSON file with entities and relationships
        
        Args:
            json_file_path: Path to the JSON file
            
        Returns:
            Tuple of (success, message)
        """
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
                
            # Extract filename without path and extension to use as prefix
            filename = os.path.basename(json_file_path)
            filename = os.path.splitext(filename)[0]
            
            # Process entities
            entities = data.get("entities", [])
            entity_id_mapping: Dict[str, str] = {}  # Map original IDs to final IDs (either existing or new)
            
            for entity in entities:
                # Create an ID for the entity
                original_id = entity.get("id", "")
                temp_id = f"{filename}_{original_id}"  # Temporary ID before disambiguation
                
                # Store the original ID for reference
                entity["original_id"] = original_id
                entity["id"] = temp_id
                
                # Insert or update the entity and get its final ID
                final_id = self.insert_entity(entity)
                
                # Store mapping for relationship processing
                entity_id_mapping[original_id] = final_id
                
            # Process relationships
            relationships = data.get("relationships", [])
            processed_rels = 0
            
            for relationship in relationships:
                # Create a relationship ID
                original_id = relationship.get("id", "")
                relationship["id"] = f"{filename}_{original_id}"
                
                # Update source and target references using the mapping
                original_source = relationship.get("source", "")
                original_target = relationship.get("target", "")
                
                # Use the mapped entity IDs (which could be existing entities or new ones)
                source_id = entity_id_mapping.get(original_source)
                target_id = entity_id_mapping.get(original_target)
                
                if source_id and target_id:
                    relationship["source"] = source_id
                    relationship["target"] = target_id
                    self.insert_relationship(relationship)
                    processed_rels += 1
                
            return True, f"Processed {len(entities)} entities ({self.disambiguation_count} disambiguated) and {processed_rels} relationships"
        except Exception as e:
            return False, f"Error processing {json_file_path}: {str(e)}"

    def clear_database(self) -> str:
        """Clear all data in the database"""
        with self.driver.session() as session:
            session.run(Query(r"MATCH (n) DETACH DELETE n"))
            return "Database cleared"

    def get_database_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about the database"""
        with self.driver.session() as session:
            # Count nodes by label
            node_query = Query(r"""
            MATCH (n)
            RETURN labels(n) as label, count(n) as count
            """)
            node_result = session.run(node_query)
            
            # Count relationships by type
            rel_query = Query(r"""
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
            """)
            rel_result = session.run(rel_query)
            
            stats: Dict[str, Dict[str, int]] = {
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