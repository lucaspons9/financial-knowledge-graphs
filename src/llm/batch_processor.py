from typing import List, Dict, Any
import json
from src.llm.model_handler import LLMHandler
from langchain_core.prompts import ChatPromptTemplate


class BatchProcessor:
    def __init__(self):
        self.llm_handler = LLMHandler()
        self.model = self.llm_handler.get_model()
        self.is_t5 = self.llm_handler.provider == "t5"

    def run_batch(self, task: str, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of texts using the modern LangChain approach
        :param task: The task type (e.g., 'entity_extraction', 'relationship_extraction')
        :param texts: A list of text inputs to process
        :return: List of LLM-generated outputs
        """
        # Create a chat prompt template from the string template
        chat_prompt = ChatPromptTemplate.from_template(self.llm_handler.prompts[task])
        
        # Create a runnable sequence using the pipe operator
        chain = chat_prompt | self.model
        
        # Run in batch mode
        responses = chain.batch([{"text": t} for t in texts])

        return responses

    def run_task(self, task: str, text: str) -> Any:
        """Generate a prompt and pass it to the selected model"""
        if self.is_t5:
            # Handle T5 model
            model, tokenizer = self.model
            
            # Format the prompt for T5
            input_text = self.llm_handler.prompts[task].replace("\n{text}\n", text)
            
            # Tokenize and generate
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(
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
            # Handle LangChain models
            # Create a chat prompt template
            chat_prompt = ChatPromptTemplate.from_template(self.llm_handler.prompts[task])
            
            # Create a runnable sequence using the pipe operator
            chain = chat_prompt | self.model
            
            # Run the chain with the input
            return chain.invoke({"text": text})

# Example Usage
if __name__ == "__main__":
    processor = BatchProcessor()
    texts = [
        "Google and Samsung announced a strategic partnership for AI development.",
        "Microsoft acquired Activision Blizzard."
    ]
    responses = processor.run_task("v3", texts[0])
    for i, response in enumerate(responses):
        print(f"Response {i+1}:", response)