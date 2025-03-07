import os
import yaml
from dotenv import load_dotenv
from langchain.llms import OpenAI, Anthropic, Cohere
from langchain.chat_models import ChatOpenAI
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

# Load environment variables from .env
load_dotenv()

class LLMHandler:
    def __init__(self, config_path="config.yaml"):
        # Load Config
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.provider = self.config["llm_provider"]
        self.models = self.config["models"]
        self.mode = self.config["mode"]  # Choose between "full" and "light"

    def get_model(self):
        """Dynamically selects the LLM model based on config.yaml mode setting"""
        mode_key = "full_model" if self.mode == "full" else "light_model"

        # Load API keys from environment variables
        api_key_env_mapping = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "cohere": os.getenv("COHERE_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY"),
        }
        api_key = api_key_env_mapping.get(self.provider)

        if self.provider == "openai":
            return ChatOpenAI(
                model_name=self.models["openai"][mode_key],
                openai_api_key=api_key,
                temperature=self.models["openai"]["temperature"],
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
                cohere_api_key=api_key
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