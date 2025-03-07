from src.llm.model_handler import LLMHandler
from src.llm.prompt_manager import PromptManager
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class BatchProcessor:
    def __init__(self):
        self.llm_handler = LLMHandler()
        self.prompt_manager = PromptManager()
        self.model = self.llm_handler.get_model()

    def run_batch(self, task, texts):
        """
        Process a batch of texts using LangChain's `chain.batch()`
        :param task: The task type (e.g., 'entity_extraction', 'relationship_extraction')
        :param texts: A list of text inputs to process
        :return: List of LLM-generated outputs
        """
        # Load the appropriate prompt template
        prompt_template = PromptTemplate(
            input_variables=["text"],
            template=self.prompt_manager.get_prompt(task, text="{text}")
        )

        # Initialize LangChain with the model and prompt
        chain = LLMChain(llm=self.model, prompt=prompt_template)

        # Run in batch mode
        responses = chain.batch([{"text": t} for t in texts], return_only_outputs=True)
        return responses
    
    def run_task(self, task, text):
        """Generate a prompt and pass it to the selected LLM"""
        prompt = self.prompt_manager.get_prompt(task, text=text)
        return model(prompt)