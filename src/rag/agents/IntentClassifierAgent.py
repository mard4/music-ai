from pathlib import Path
from typing import Dict, Any, List
from rag.agents.base import BaseAgent


class IntentClassifierAgent(BaseAgent):
    def __init__(self):

        descriptors_path = Path("data_exploration/extract_descriptors.py")
        descriptors_content = descriptors_path.read_text(encoding="utf-8")

        super().__init__(
            prompt_file="intent_classification.txt",
            context={
                "descriptors": descriptors_content,
                "agent_name": "IntentClassifier"
            }
        )

    def run(self, user_input: str) -> Dict[str, str]:
        self.logger.info(f"Classificando intento per: '{user_input}'")

        dynamic_content = self.render_prompt(user_query=user_input)

        messages = [
            {"role": "system", "content": dynamic_content},
            {"role": "user", "content": user_input}
        ]

        # La BaseAgent gestisce il JSON parsing
        result = self.call_llm(messages, temperature=0.0, json_mode=True)

        if result:
            return result
        return {"intent": "OTHER", "params": user_input}
