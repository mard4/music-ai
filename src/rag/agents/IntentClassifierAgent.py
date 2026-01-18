import sys
from pathlib import Path
from typing import Dict, Any, List
from rag.agents.AgentBase import AgentBase

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

class IntentClassifierAgent(AgentBase):
    """"
    Router  per smistare le richieste verso il componente specializzato corretto

    • Funzione: Riceve l'input in linguaggio naturale.
    • Logica: Se l'input contiene un file audio o parole come "analizza", attiva il ramo ANALYSIS. Se contiene richieste come "trova" o "dammi", attiva il ramo RETRIEVAL.
    """

    def __init__(self):

        current_file = Path(__file__).resolve()
        src_root = current_file.parent.parent.parent
        descriptors_path = src_root / "data_exploration" / "descriptors_list.txt"
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
