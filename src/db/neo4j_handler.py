from neo4j import GraphDatabase
from owlready2 import get_ontology
import csv

class Neo4jHandler:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        """Connect to Neo4j database"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()

    def insert_triplet(self, subject, relationship, object_):
        """Insert a relationship triplet (Entity A)-[:RELATION]->(Entity B)"""
        with self.driver.session() as session:
            query = """
            MERGE (a:Entity {name: $subject})
            MERGE (b:Entity {name: $object_})
            MERGE (a)-[r:{rel}]->(b)
            """
            session.run(query, subject=subject, rel=relationship.upper(), object_=object_)

    def insert_ontology_from_owl(self, owl_file):
        """Convert OWL ontology to Neo4j format and insert it"""
        onto = get_ontology(owl_file).load()
        with self.driver.session() as session:
            for cls in onto.classes():
                query = "MERGE (:Entity {name: $name})"
                session.run(query, name=cls.name)

            for prop in onto.object_properties():
                for s, p, o in prop.get_relations():
                    query = """
                    MATCH (a:Entity {name: $subject}), (b:Entity {name: $object_})
                    MERGE (a)-[:{rel}]->(b)
                    """
                    session.run(query, subject=s.name, rel=prop.name.upper(), object_=o.name)

    def insert_ontology_from_csv(self, node_csv, relationship_csv):
        """Load ontology from CSV files and insert into Neo4j"""
        with self.driver.session() as session:
            # Insert nodes
            with open(node_csv, "r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    query = "MERGE (:Entity {name: $name})"
                    session.run(query, name=row["name"])

            # Insert relationships
            with open(relationship_csv, "r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    query = """
                    MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
                    MERGE (a)-[:{rel}]->(b)
                    """
                    session.run(query, source=row["source"], rel=row["relationship"].upper(), target=row["target"])

    def query_all_relationships(self):
        """Retrieve all relationships"""
        with self.driver.session() as session:
            query = "MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name"
            result = session.run(query)
            return [(record["a.name"], record["type(r)"], record["b.name"]) for record in result]

    def query_entity_relationships(self, entity_name):
        """Find relationships for a given entity"""
        with self.driver.session() as session:
            query = """
            MATCH (e:Entity {name: $entity_name})-[r]->(related)
            RETURN e.name, type(r), related.name
            """
            result = session.run(query, entity_name=entity_name)
            return [(record["e.name"], record["type(r)"], record["related.name"]) for record in result]

# Example Usage
if __name__ == "__main__":
    neo4j_handler = Neo4jHandler()

    # Load Ontology from OWL
    neo4j_handler.insert_ontology_from_owl("ontology/market_ontology.owl")

    # Insert from CSV (optional alternative)
    # neo4j_handler.insert_ontology_from_csv("ontology/ontology_nodes.csv", "ontology/ontology_relationships.csv")

    # Insert LLM-extracted triplets
    neo4j_handler.insert_triplet("Microsoft", "ACQUIRED", "Activision Blizzard")
    neo4j_handler.insert_triplet("Tesla", "PARTNERS_WITH", "Panasonic")
    neo4j_handler.insert_triplet("Apple", "INVESTED_IN", "OpenAI")

    # Query the database
    print("All Relationships:", neo4j_handler.query_all_relationships())
    print("Microsoft Relationships:", neo4j_handler.query_entity_relationships("Microsoft"))

    neo4j_handler.close()