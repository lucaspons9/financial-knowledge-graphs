import os
import yaml
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv
from langchain_community.llms import OpenAI, Anthropic, Cohere
from langchain_openai import ChatOpenAI
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from src.utils.reading_files import load_yaml

# Load environment variables from .env
load_dotenv()

class LLMHandler:
    def __init__(self, config_path: str = "config.yaml"):
        # Load Config
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.provider = self.config["llm_provider"]
        self.models = self.config["models"]
        self.mode = self.config["mode"]  
        self.prompts = load_yaml(self.config.get("prompt_path", "prompts/prompts.yaml"))
        
    def get_model(self, role: Optional[str] = None, content: Optional[str] = None) -> Any:
        """Dynamically selects the LLM model based on config.yaml mode setting"""
        mode_key = "full_model" if self.mode == "full" else "light_model"

        # Load API key for selected provider
        api_key = os.getenv(f"{self.provider.upper()}_API_KEY")

        if self.provider == "openai":
            return ChatOpenAI(
                openai_api_key=api_key,
                temperature=self.models["openai"]["temperature"],
                model_name=self.models["openai"][mode_key],
            )
        
        elif self.provider == "anthropic":
            return Anthropic(
                model=self.models["anthropic"][mode_key],
                anthropic_api_key=api_key,
                temperature=self.models["anthropic"]["temperature"],
            )

        elif self.provider == "cohere":
            return Cohere(
                model=self.models["cohere"][mode_key],
                cohere_api_key=api_key
            )

        elif self.provider == "mistral":
            return Cohere(  # Mistral API works similarly to Cohere
                model=self.models["mistral"][mode_key],
                cohere_api_key=api_key,
            )

        elif self.provider == "local":
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            model_key = "full_model_path" if self.mode == "full" else "light_model_path"
            model_name = self.models["local"][model_key]
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
            return HuggingFacePipeline(pipeline=hf_pipeline)

        else:
            raise ValueError("Invalid LLM provider in config.yaml")