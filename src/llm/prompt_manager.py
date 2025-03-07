import yaml
from langchain.prompts import PromptTemplate

class PromptManager:
    def __init__(self, prompt_path="prompts/prompts.yaml"):
        """Load all prompts from the YAML file."""
        with open(prompt_path, "r") as file:
            self.prompts = yaml.safe_load(file)

    def get_prompt(self, task, **kwargs):
        """
        Retrieve and format a prompt for a specific task.
        :param task: Name of the task (e.g., 'entity_extraction', 'relationship_extraction')
        :param kwargs: Variables to fill in the template
        :return: LangChain PromptTemplate instance
        """
        if task not in self.prompts:
            raise ValueError(f"Prompt for task '{task}' not found.")
        
        prompt_text = self.prompts[task]["template"]
        prompt_template = PromptTemplate(input_variables=list(kwargs.keys()), template=prompt_text)
        return prompt_template.format(**kwargs)