# Financial Knowledge Graphs

A powerful tool for extracting financial entities and relationships from financial news and building knowledge graphs using LLMs and Neo4j.

## Overview

Financial Knowledge Graphs is a Python-based application that leverages Large Language Models (LLMs) to extract financial entities and relationships from news articles. The extracted information is then stored in a Neo4j graph database, creating a queryable knowledge graph of financial information.

## Features

- **Entity Extraction**: Automatically identify companies, organizations, and financial entities in text
- **Relationship Extraction**: Discover connections between entities (acquisitions, partnerships, investments)
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

The application is configured through `config.yaml` in the root directory. This file controls which LLM provider to use, model selection, and other settings. Model definitions are located in `models.yaml`.

### Configuration Options

```yaml
llm_provider: "openai" # Options: "openai", "anthropic", "cohere", "mistral", "local"
mode: "light" # Options: "full" for high-quality, "light" for efficiency
task: "entity_extraction" # Default task to run
```


## Usage

To run the application with the default configuration:

```bash
python src/main.py
```

This will:

1. Load sample news data from `data/raw/sample_news.yaml`
2. Extract entities from each news article using the configured LLM
3. Print the extracted entities to the console

<!-- 

## Advanced Usage

### Entity Extraction

### Relationship Extraction

### Storing in Neo4j
-->

## Project Structure

```
├── config.yaml                # Main configuration file
├── data/
│   └── raw/                   # Raw data files
│       └── sample_news.yaml   # Sample news articles
├── docs/                      # Documentation
├── prompts/
│   └── prompts.yaml           # LLM prompt templates
├── src/
│   ├── db/                    # Database handlers
│   │   └── neo4j_handler.py   # Neo4j integration
│   ├── llm/                   # LLM integration
│   │   ├── batch_processor.py # Process multiple texts
│   │   └── model_handler.py   # LLM provider management
│   ├── utils/                 # Utility functions
│   └── main.py                # Main application entry point
```

