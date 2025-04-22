# Financial Knowledge Graphs: Execution Guide

### Commands

#### Entity Extraction (LLM)

```bash
python -m src.main llm
```

#### Neo4j Database Operations

```bash
python -m src.main neo4j
```

#### Evaluation

```bash
python -m src.main evaluate
```

#### Retrieve Batch Results

```bash
python -m src.main batch <batch_id> [options]
```

**Options:**

- `--parent`: Treat as parent batch
- `--check_only`: Check status only
- `--wait`: Wait for completion
- `--output_dir DIR`: Custom output location

### Configuration Files

#### LLM Processing

**File:** `configs/config_llm_execution.yaml`

- `llm_provider`: "openai", /_ "anthropic", "cohere", "mistral", _/ "local"
- `mode`: "full" (high-quality) or "light" (efficient)
- `task`: "entity_extraction"
- `data_path`: Path to input data
- `use_batch`: Enable batch processing
- `wait_for_completion`: Wait for batch jobs
- `batch_size`: Number of samples per batch

> **Local LLM Execution**: For running extraction with local Llama3 models via Ollama, see [Ollama Setup Guide](ollama_setup.md)

#### Neo4j Database

**File:** `configs/config_neo4j.yaml`

- `container_name`: Docker container name
- `port`: Bolt protocol port
- `browser_port`: Web interface port
- `user`: Neo4j username
- `password`: Neo4j password
- `clear_database`: Clear database before loading
- `create_schema`: Create schema constraints
- `load_data`: Load data from JSON files
- `data_path`: Path to JSON files

#### Evaluation

**File:** `configs/config_evaluation.yaml`

- `ground_truth_dir`: Ground truth data path
- `results_dir`: Test results directory
- `test_name`: Name of test to evaluate

#### Model Configuration

**File:** `configs/models.yaml`

- Model settings for different providers
- Configuration for "full" and "light" modes

#### Authentication

API keys should be stored in a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key
```

---

### Legacy Features

#### Stanford OpenIE Extraction

```bash
python -m src.main openie
```

**File:** `configs/config_stanford_openie.yaml`

- `data_path`: Input data path
- `batch_name`: Base name for output files
- `output_directory`: Output directory
- `generate_graphs`: Generate visualizations
- `openie_properties`: Configuration parameters
  - `openie.affinity_probability_cap`: Probability cap
  - `openie.max_clauses_per_sentence`: Max clauses
  - `openie.resolve_coref`: Resolve coreferences
- `batch_processing`: Batch settings
  - `max_batch_size`: Maximum texts per batch
  - `parallel_processing`: Process in parallel
  - `num_workers`: Number of workers

> **Note:** Stanford OpenIE is included for research comparison but produces less accurate results than LLM-based extraction. It is not recommended for production use.
