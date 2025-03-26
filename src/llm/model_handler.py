import os
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from src.utils.reading_files import load_yaml

# Load environment variables from .env
load_dotenv()

class LLMHandler:
    def __init__(self, config_path: str = "configs/config_llm_execution.yaml"):
        # Load main configuration from config_llm_execution.yaml (or specified path)
        self.config = load_yaml(config_path)
        # Load models configuration from separate file (defaults to configs/models.yaml)
        self.models = load_yaml("configs/models.yaml")
        
        self.provider = self.config["llm_provider"]
        self.mode = self.config["mode"]  
        self.prompts = load_yaml("configs/prompts.yaml")
        
    def get_model(self) -> Any:
        """Dynamically selects the LLM model based on config_llm_execution.yaml mode setting"""
        mode_key = "full_model" if self.mode == "full" else "light_model"

        # Load API key for selected provider
        api_key = os.getenv(f"{self.provider.upper()}_API_KEY")

        if self.provider == "openai":
            return ChatOpenAI(
                openai_api_key=api_key,
                temperature=self.models["openai"]["temperature"],
                model_name=self.models["openai"][mode_key],
            )
        elif self.provider == "llama3":
            # Ollama doesn't need an API key as it's locally hosted
            return OllamaLLM(
                model=self.models["llama3"][mode_key],
                temperature=self.models["llama3"]["temperature"],
            )
        else:
            raise ValueError("Invalid LLM provider in config_llm_execution.yaml")