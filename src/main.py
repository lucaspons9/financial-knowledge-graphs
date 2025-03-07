from src.llm.prompt_manager import PromptManager
from src.llm.model_handler import LLMHandler

class LLMProcessor:
    def __init__(self):
        self.llm_handler = LLMHandler()
        self.prompt_manager = PromptManager()

    def run_task(self, task, text):
        """Generate a prompt and pass it to the selected LLM"""
        prompt = self.prompt_manager.get_prompt(task, text=text)
        model = self.llm_handler.get_model()
        return model(prompt)

# Example Usage
processor = LLMProcessor()

text = "Google and Samsung announced a strategic partnership for AI development."

# Extract Entities
entities = processor.run_task("entity_extraction", text)
print("Entities:", entities)

# Extract Relationships
relationships = processor.run_task("relationship_extraction", text)
print("Relationships:", relationships)