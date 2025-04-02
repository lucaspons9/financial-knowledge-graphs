# Configuration Files

This directory contains configuration files for the financial knowledge graphs project.

## Configuration Files

- `config_llm_execution.yaml`: Configuration for LLM-based entity extraction
- `config_stanford_openie.yaml`: Configuration for Stanford OpenIE extraction
- `models.yaml`: Configuration for LLM models

## Prompts Directory

The `prompts` directory contains prompt templates used by the LLM-based entity extraction.

## Usage

### LLM-based Entity Extraction

To run the LLM-based entity extraction:

```bash
python -m src.main llm
```

This will use the configuration in `config_llm_execution.yaml`.

### Stanford OpenIE Extraction

To run the Stanford OpenIE extraction:

```bash
python -m src.main openie
```

This will use the configuration in `config_stanford_openie.yaml`.

### Running Both Tasks

To run both tasks sequentially:

```bash
python -m src.main both
```

This will run both tasks using their respective configuration files.

## Configuration Parameters

### LLM Execution Configuration

- `llm_provider`: The LLM provider to use (e.g., "openai", "anthropic", "cohere", "mistral", "local")
- `mode`: The execution mode ("full" for high-quality, "light" for efficiency)
- `task`: The task to perform (e.g., "entity_extraction")
- `data_path`: Path to the data file containing texts to process
- `models_path`: Path to the models configuration file
- `prompt_path`: Path to the prompt YAML file

### Stanford OpenIE Configuration

- `data_path`: Path to the data file containing texts to process
- `batch_name`: Base name for the batch processing output files
- `openie_properties`: Properties for Stanford OpenIE
  - `openie.affinity_probability_cap`: Probability cap for affinity (default: 1/3)
  - `openie.max_clauses_per_sentence`: Maximum number of clauses per sentence
  - `openie.resolve_coref`: Whether to resolve coreferences
- `output_directory`: Directory to save extracted triples
- `generate_graphs`: Whether to generate visualization graphs
- `batch_processing`: Batch processing settings
  - `max_batch_size`: Maximum number of texts to process in a single batch
  - `parallel_processing`: Whether to process texts in parallel
  - `num_workers`: Number of parallel workers

### Models Configuration

The `models.yaml` file contains configuration for different LLM models:

- Model names and versions for each provider
- Configuration for both "full" (high-quality) and "light" (efficient) modes
- Parameters specific to each model and provider
