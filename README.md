# Financial Knowledge Graphs

A powerful tool for extracting financial entities and relationships from financial news and building knowledge graphs using LLMs and Neo4j.

## Overview

Financial Knowledge Graphs is a Python-based application that leverages Large Language Models (LLMs) to extract financial entities and relationships from news articles. The extracted information is then stored in a Neo4j graph database, creating a queryable knowledge graph of financial information. The project also includes Stanford OpenIE integration for ground truth extraction.

## Features

- **Entity Extraction**: Automatically identify companies, organizations, and financial entities in text
- **Relationship Extraction**: Discover connections between entities (acquisitions, partnerships, investments)
- **Ground Truth Extraction**: Use Stanford OpenIE to extract triples as ground truth
- **Flexible LLM Integration**: Support for multiple LLM providers (OpenAI, Anthropic, Cohere, Mistral, local models)
- **Neo4j Integration**: Store and query extracted information in a graph database
- **Batch Processing**: Process multiple texts efficiently
- **Configurable**: Easy configuration through YAML files

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/financial-knowledge-graphs.git
   cd financial-knowledge-graphs
   ```

2. Create a virtual environment and sync dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv sync
   ```

3. Set up your environment variables by creating a `.env` file:

   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   COHERE_API_KEY=your_cohere_api_key
   MISTRAL_API_KEY=your_mistral_api_key
   ```

4. Start Neo4j (using Docker):
   ```bash
   docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
   ```

## Configuration

The application is configured through YAML files in the `configs` directory:

- `config_llm_execution.yaml`: Configuration for LLM-based entity extraction
- `config_stanford_openie.yaml`: Configuration for Stanford OpenIE extraction
- `models.yaml`: Configuration for LLM models
- `prompts/`: Directory containing prompt templates

### LLM Configuration Options

```yaml
llm_provider: "openai" # Options: "openai", "anthropic", "cohere", "mistral", "local"
mode: "light" # Options: "full" for high-quality, "light" for efficiency
task: "entity_extraction" # Default task to run
```

### Stanford OpenIE Configuration Options

```yaml
# Properties for Stanford OpenIE
openie_properties:
  openie.affinity_probability_cap: 0.6667 # 2/3
  openie.max_clauses_per_sentence: 10
  openie.resolve_coref: true

# Output directory for extracted triples
output_directory: "data/ground_truth"

# Whether to generate visualization graphs
generate_graphs: true
```

## Usage

The application can run in different modes:

### LLM-based Entity Extraction

```bash
python -m src.main llm
```

This will:

1. Load sample news data from the configured data path
2. Extract entities from each news article using the configured LLM
3. Print the extracted entities to the console

### Stanford OpenIE Ground Truth Extraction

```bash
python -m src.main openie
```

This will:

1. Load sample news data from the configured data path
2. Extract triples from each news article using Stanford OpenIE
3. Save the extracted triples to JSON files in the configured output directory
4. Generate visualization graphs if enabled

### Running Both Tasks

```bash
python -m src.main both
```

This will run both the LLM-based entity extraction and the Stanford OpenIE ground truth extraction sequentially.

<!--

## Advanced Usage

### Entity Extraction

### Relationship Extraction

### Storing in Neo4j
-->

## Project Structure

```
├── configs/                   # Configuration directory
│   ├── config_llm_execution.yaml  # LLM configuration
│   ├── config_stanford_openie.yaml  # Stanford OpenIE configuration
│   ├── models.yaml            # LLM models configuration
│   └── prompts/               # LLM prompt templates
├── data/
│   ├── ground_truth/          # Ground truth data from Stanford OpenIE
│   └── raw/                   # Raw data files
│       └── sample_news.yaml   # Sample news articles
├── docs/                      # Documentation
├── src/
│   ├── db/                    # Database handlers
│   │   └── neo4j_handler.py   # Neo4j integration
│   ├── llm/                   # LLM integration
│   │   ├── batch_processor.py # Process multiple texts
│   │   └── model_handler.py   # LLM provider management
│   ├── utils/                 # Utility functions
│   │   └── ground_truth.py    # Stanford OpenIE ground truth extractor
│   ├── main.py                # Main application entry point
│   ├── run_llm_task.py        # LLM task runner
│   └── run_stanford_openie.py # Stanford OpenIE runner
```
