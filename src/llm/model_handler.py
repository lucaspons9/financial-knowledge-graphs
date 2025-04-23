import os
from typing import Any, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from src.utils.file_utils import load_yaml
from langchain_core.prompts import ChatPromptTemplate
import json

# Load environment variables from .env
load_dotenv()

class LLMHandler:
    """
    Handler for interacting with various Large Language Models.
    
    Coordinates interactions with different LLM providers
    based on configuration settings.
    """

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
        elif self.provider == "t5":
            # Import T5 components only when needed
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            
            # Get model name from config
            model_name = self.models["t5"][mode_key]
            model_path = os.path.join("src/llm/models", model_name)
            
            # Check if model exists locally, otherwise download
            if os.path.exists(model_path):
                tokenizer = T5Tokenizer.from_pretrained(model_path)
                model = T5ForConditionalGeneration.from_pretrained(model_path)
            else:
                tokenizer = T5Tokenizer.from_pretrained(model_name)
                model = T5ForConditionalGeneration.from_pretrained(model_name)
                
                # Save model locally for future use
                os.makedirs(model_path, exist_ok=True)
                tokenizer.save_pretrained(model_path)
                model.save_pretrained(model_path)
            
            return model, tokenizer
        else:
            raise ValueError(f"Invalid LLM provider in config_llm_execution.yaml: {self.provider}")
            
    def run_task(self, task: str, text: str) -> Any:
        """
        Generate a prompt and pass it to the selected model.
        
        Args:
            task: The task name corresponding to a prompt in prompts.yaml
            text: Text input to process
            
        Returns:
            Model response in appropriate format
        """
        # Get the model
        model = self.get_model()
        
        # Check if using T5
        if self.provider == "t5":
            # Handle T5 model
            model_t5, tokenizer = model
            
            # Format the prompt for T5
            input_text = self.prompts[task].replace("{text}", text)
            
            # Tokenize and generate
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            outputs = model_t5.generate(
                inputs.input_ids,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode and format response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Format response as JSON for consistency
            try:
                # Try to parse as JSON if it's already in JSON format
                json.loads(response)
                return response
            except json.JSONDecodeError:
                # If not JSON, format it as a triplet
                formatted_response = f'[{{"subject": "{response}"}}]'
                return formatted_response
        else:
            # Handle LangChain models (OpenAI, Llama, etc.)
            # Create a chat prompt template
            chat_prompt = ChatPromptTemplate.from_template(self.prompts[task])
            
            # Create a runnable sequence using the pipe operator
            chain = chat_prompt | model
            
            # Run the chain with the input
            return chain.invoke({"text": text})