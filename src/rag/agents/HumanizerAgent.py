import json
from typing import Dict, Any, List
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from jinja2 import Template
from config.settings import settings
from rag.utils import read_prompt, logger

class HumanizerAgent:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.prompt_analysis = read_prompt("humanizer_analysis.txt")
        self.prompt_default = read_prompt("humanizer_default.txt")

    def generate_response(self, user_query: str, intent: str, data: Any) -> str:
        if intent == "ANALYSIS":
            system_prompt = self.prompt_analysis
        else:
            system_prompt = self.prompt_default

        user_message = f"Query Utente: {user_query}\nDATI SISTEMA:\n{json.dumps(data, indent=2)}\nRispondi in italiano:"
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            if response.choices:
                return response.choices[0].message.content or ""
            return "Nessuna risposta generata."
        except Exception as e:
            logger.error(f"Errore generazione risposta: {e}")
            return "Mi dispiace, ho riscontrato un errore nel generare la risposta."