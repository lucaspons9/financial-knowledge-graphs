# Using Ollama with Llama3 Models

This guide will help you set up and use Llama3 models locally via Ollama.

## Installation

1. **Install Ollama**:

   - Download and install Ollama from [ollama.ai](https://ollama.ai/)
   - Follow the installation instructions for your operating system

2. **Pull the Llama3 model**:

   ```bash
   # Pull the 8B parameter base model
   ollama pull llama3:8b

   # Or pull the instruction-tuned version (recommended for chat applications)
   ollama pull llama3:8b-instruct
   ```

3. **Verify Ollama is running**:
   Ollama should start running as a service automatically after installation. You can verify by running:

   ```bash
   ollama list
   ```

   If Ollama is not running, you can start it with:

   ```bash
   ollama serve
   ```

## Configuration and Usage

**Update configuration files**:

- The `config_llm_execution.yaml` file has been updated to use Ollama as the provider
- The `models.yaml` file includes configuration for Llama3 models

Run your application as usual. The system will now connect to your local Ollama instance and use the Llama3 model specified in the configuration.

```bash
python -m src.main llm
```

## Troubleshooting

1. **Model not found**:

   - Ensure you've pulled the correct model: `ollama pull llama3:8b`
   - Check if the model name in `models.yaml` matches the name used in Ollama

2. **Ollama not running**:
   - Ensure the Ollama service is running: `ollama serve`
3. **Slow responses**:
   - Llama3 models run locally and may be slower than cloud-based LLMs, especially on machines without a powerful GPU
   - Consider using a smaller model if performance is an issue

## Performance Considerations

- The 8B parameter model is recommended for most consumer hardware
- For better performance, make sure your machine has a decent GPU with at least 8GB VRAM
- CPU inference is possible but will be significantly slower
