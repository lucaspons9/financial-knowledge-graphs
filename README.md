# Financial Knowledge Graphs

A powerful tool for extracting financial entities and relationships from financial news and building knowledge graphs using LLMs and Neo4j.

## Overview

Financial Knowledge Graphs is a Python-based application that leverages Large Language Models (LLMs) to extract financial entities and relationships from news articles. The extracted information is then stored in a Neo4j graph database, creating a queryable knowledge graph of financial information. The project also includes Stanford OpenIE integration for ground truth extraction.

## Features

- **Entity Extraction**: Automatically identify companies, organizations, and financial entities in text
- **Relationship Extraction**: Discover connections between entities (acquisitions, partnerships, investments)
- **Triplet Extraction**: Extract knowledge triplets (Subject, Predicate, Object) from financial text
- **Ground Truth Extraction**: Use Stanford OpenIE to extract triples as ground truth
- **Flexible LLM Integration**: Support for multiple LLM providers (OpenAI, Anthropic, Cohere, Mistral, local models)
- **Neo4j Integration**: Store and query extracted information in a graph database
- **Batch Processing**: Process multiple texts efficiently, with special support for OpenAI's Batch API
- **Cost-Efficient Processing**: Save up to 50% on API costs using OpenAI's Batch API
- **Test Result Storage**: Automatically store test results in versioned directories
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
- `prompts.yaml`: Prompt templates for different extraction tasks

### LLM Configuration Options

```yaml
llm_provider: "openai" # Options: "openai", "anthropic", "cohere", "mistral", "local"
mode: "light" # Options: "full" for high-quality, "light" for efficiency
task: "entity_extraction" # Default task to run
use_batch: false # Enable to use batch processing (OpenAI Batch API for cost savings)
wait_for_completion: false # Wait for batch processing to complete
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

The system now stores ground truth data from Stanford OpenIE in versioned directories:

```bash
python -m src.main openie
```

This will:

1. Load sample data from the configured data path
2. Extract triples using Stanford OpenIE
3. Store results in a versioned directory structure:
   - Base directory: `data/ground_truth/`
   - Run directories: `openie_test_1`, `openie_test_2`, etc.
   - Each run contains:
     - Extracted triples for each sentence in JSON format
     - A summary.json file with metadata about the run

You can configure the test name and other extraction parameters in `configs/config_stanford_openie.yaml`.

### Running Triplet Extraction Tests

To run triplet extraction on sample sentences and store the results:

1. Configure the test in `configs/config_llm_execution.yaml`:

   ```yaml
   task: "triplet_extraction"
   data_path: "data/raw/your_sample_file.yaml"
   store_results: true
   results_dir: "runs"
   test_name: "test_llm_prompt"
   ```

2. Run the LLM-based triplet extraction:

   ```bash
   python -m src.main llm
   ```

3. Results will be stored in sequentially numbered directories (`test_llm_prompt_1`, `test_llm_prompt_2`, etc.) in the `runs` directory, with each test result in a JSON file named after the sentence ID.

### Cost-Efficient Batch Processing with OpenAI

For processing large datasets (thousands of articles), you can use OpenAI's Batch API for significant cost savings:

1. Enable batch processing in `configs/config_llm_execution.yaml`:

   ```yaml
   llm_provider: "openai"
   task: "triplet_extraction"
   data_path: "data/raw/large_dataset.csv"
   store_results: true
   results_dir: "runs"
   test_name: "batch_processing"
   use_batch: true # Enable batch processing
   wait_for_completion: false # Set to true to wait for results
   ```

2. Run the batch processing:

   ```bash
   python -m src.main llm
   ```

3. If `wait_for_completion` is set to `false`, the system will:

   - Submit your batch to OpenAI's Batch API
   - Report the batch ID for later reference
   - Return immediately without waiting for results

4. To retrieve results later:

   ```bash
   python -m src.retrieve_batch your_batch_id
   ```

This approach can save approximately 50% on API costs compared to synchronous API calls and works well for processing datasets of 20,000+ articles.

### Handling Large Datasets (20,000+ Samples)

The system automatically handles datasets of any size through intelligent batch management:

#### Key Concepts

- **Single Batch**: For smaller datasets (under 2,000 samples by default). Each batch is processed as a single unit.
- **Parent Batch**: For larger datasets (20,000+ samples). The system automatically splits the data into multiple sub-batches (e.g., 10 batches of 2,000 samples each) and tracks them under a parent batch ID.
- **Batch Size**: Configurable parameter (default: 2,000) determining when data gets split into multiple batches.

#### Using Parent Batches

When processing large datasets:

1. Configure batch size in `configs/config_llm_execution.yaml`:

   ```yaml
   batch_size: 2000 # Determines when data gets split into multiple batches
   ```

2. The system automatically:

   - Detects when dataset exceeds batch size
   - Creates a parent batch with multiple sub-batches
   - Returns a parent batch ID (different format from regular batch IDs)

3. To check status or retrieve results using the unified batch retrieval interface:

   ```bash
   # Check status of an individual batch
   python -m src.retrieve_batch your_batch_id --check_only

   # Check status of a parent batch
   python -m src.retrieve_batch your_parent_batch_id --parent --check_only

   # Wait for an individual batch to complete and retrieve results
   python -m src.retrieve_batch your_batch_id --wait

   # Wait for a parent batch to complete and retrieve results
   python -m src.retrieve_batch your_parent_batch_id --parent --wait
   ```

   The `--parent` flag indicates you're working with a parent batch rather than an individual batch.

Parent batch processing provides significant advantages:

- Efficiently handles datasets of any size
- Maintains OpenAI's batch API cost savings
- Combines results from all sub-batches in a single operation
- Provides progress tracking across all sub-batches

### Unified Batch Retrieval

The system now provides a unified approach for retrieving both individual batches and parent batches using a single command:

```bash
# Check status of an individual batch
python -m src.retrieve_batch your_batch_id --check_only

# Check status of a parent batch
python -m src.retrieve_batch your_parent_batch_id --parent --check_only

# Wait for an individual batch to complete and retrieve results
python -m src.retrieve_batch your_batch_id --wait

# Wait for a parent batch to complete and retrieve results
python -m src.retrieve_batch your_parent_batch_id --parent --wait

# Specify a custom output directory for results
python -m src.retrieve_batch your_batch_id --output_dir /path/to/output

# Specify wait interval when using the wait flag (default is 30 seconds)
python -m src.retrieve_batch your_batch_id --wait --wait_interval 60
```

The `--parent` flag indicates you're working with a parent batch rather than an individual batch.

### Avoiding Duplicate Processing

The system intelligently tracks which newsIDs have already been processed to avoid duplicate processing:

1. When submitting a batch with a specified parent batch directory:

   - The system reads the parent batch metadata to identify already processed newsIDs
   - Any newsIDs that have already been processed are automatically filtered out
   - Only new, unprocessed newsIDs are included in the new batch

2. To continue processing with an existing parent batch:

   ```yaml
   # In configs/config_llm_execution.yaml
   parent_batch_dir: "data/batch_processing/parent_batch_20250414_184157_03d77ca6"
   ```

   This ensures that:

   - Each article is processed exactly once, even across multiple execution runs
   - You can incrementally process large datasets in multiple sessions
   - Processing can be resumed after interruptions without duplicating work

3. The parent batch metadata tracks:
   - All processed newsIDs
   - All child batches
   - Timestamps and processing status

This approach is ideal for processing large datasets over time, ensuring each unique article is processed only once while maintaining a complete record of all processing operations.

### Test Results Storage

All LLM test runs are stored in the `runs` directory by default. The system automatically:

- Creates a new folder for each test run with an incremented index (e.g., `test_llm_prompt_1`, `test_llm_prompt_2`)
- Stores individual test results as JSON files named after the sentence/document ID
- Creates a summary.json file with metadata about the test configuration and timestamp
- Maintains a clear, versioned history of all test runs for easy comparison and analysis

You can customize the storage location by modifying the `results_dir` parameter in the configuration file.

### Running Both Tasks

```bash
python -m src.main both
```

This will run both the LLM-based entity extraction and the Stanford OpenIE ground truth extraction sequentially.

### Evaluating Triplet Extraction Results

To evaluate the latest triplet extraction results against ground truth:

1. Configure the evaluation in `configs/config_evaluation.yaml`:

   ```yaml
   # Ground truth directory
   ground_truth_dir: "data/ground_truth/triplets"

   # Results settings
   results_dir: "runs"
   test_name: "test_llm_prompt"
   ```

2. Run the evaluation:

   ```bash
   python -m src.main evaluate
   ```

3. The evaluation results will be:
   - Displayed in the console
   - Saved to a timestamped JSON file in the `evaluations` directory (if configured)

The evaluation compares triplets using fuzzy matching to handle slight variations in wording. It calculates precision, recall, and F1 score for each file and overall.

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
│   └── prompts.yaml           # LLM prompt templates
├── data/
│   ├── batch_processing/      # OpenAI Batch API processing files
│   ├── ground_truth/          # Ground truth data from Stanford OpenIE
│   ├── processed/             # Processed data files
│   └── raw/                   # Raw data files
│       └── sample_news.yaml   # Sample news articles
├── docs/                      # Documentation
├── src/
│   ├── db/                    # Database handlers
│   │   └── neo4j_handler.py   # Neo4j integration
│   ├── llm/                   # LLM integration
│   │   ├── model_handler.py   # LLM provider management
│   │   └── openai_batch_processor.py # OpenAI Batch API integration
│   ├── utils/                 # Utility functions
│   │   ├── data_processing.py # CSV data processing utilities
│   │   ├── evaluation.py      # Evaluation metrics and comparison tools
│   │   ├── file_utils.py      # File I/O and path management
│   │   └── ground_truth.py    # Stanford OpenIE ground truth extractor
│   ├── main.py                # Main application entry point
│   ├── run_llm_task.py        # LLM task runner
│   ├── retrieve_batch.py      # OpenAI Batch result retriever
│   └── run_stanford_openie.py # Stanford OpenIE runner
```
