from typing import List, Dict, Any
from src.llm.model_handler import LLMHandler
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage


class BatchProcessor:
    def __init__(self):
        self.llm_handler = LLMHandler()
        self.model = self.llm_handler.get_model()

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
        """Generate a prompt and pass it to the selected LLM"""
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
    responses = processor.run_task("entity_extraction", texts[0])
    for i, response in enumerate(responses):
        print(f"Response {i+1}:", response)