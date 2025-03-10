Ontology structure (nodes and edges)

# TODO when testing:
🛠 How to Run

1️⃣ Start Neo4j (Docker)

docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j

2️⃣ Run Python Script

python src/neo4j/neo4j_handler.py

3️⃣ Connect via Neo4j Desktop
	•	Open Neo4j Desktop
	•	Connect using bolt://localhost:7687
	•	Run query:

MATCH (n) RETURN n LIMIT 10;






---
# Idea of the folder structure
├── 📂 ontology/                # OWL Ontology and its converted formats
│   ├── 📄 market_ontology.owl  # Main OWL file from Protégé
│   ├── 📄 ontology_nodes.csv   # Converted ontology nodes
│   ├── 📄 ontology_relationships.csv  # Converted ontology relationships
│   ├── 📄 ontology.cypher      # Cypher script for loading into Neo4j
│   ├── 📄 convert_owl_to_cypher.py  # Python script to convert OWL to Neo4j



1️⃣ Creating the Ontology in Protégé

You will use Protégé to define:
	•	Classes (Node Labels) → e.g., Company, Person, FinancialConcept
	•	Object Properties (Relationships) → e.g., ACQUIRED, PARTNERS_WITH, INVESTED_IN
	•	Individuals (Instances of Classes) → e.g., Microsoft, Apple, Google

After defining the ontology, save it as an OWL file:
📄 ontology/market_ontology.owl

2️⃣ Convert OWL to Neo4j-Compatible Format

Neo4j does not directly support OWL, so we need to convert it into a format that Neo4j can use.

Option 1: Convert OWL to CSV (Recommended)

We extract:
	•	Nodes (Entities like Companies & People)
	•	Edges (Relationships) (e.g., (Microsoft)-[:ACQUIRED]->(Activision))


📄 Example ontology/ontology_nodes.csv
 id,label,name
1,Company,Microsoft
2,Company,Activision Blizzard
3,Person,Elon Musk

📄 Example ontology/ontology_relationships.csv
source,target,relationship
1,2,ACQUIRED
3,1,CEO_OF