import json
from typing import Dict, Any, List
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from jinja2 import Template
from config.settings import settings
from rag.utils import read_prompt, logger



class SoundDesignerAgent:
    def __init__(self):
        # FIX: Uso la nuova classe ParameterRetriever
        self.tool = ParameterRetriever()

    def run(self, query: str) -> Dict:
        logger.info(f"Sound Designer: Ricerca parametri per '{query}'...")
        return self.tool.retrieve(query)  # Nota: metodo rinominato da get_parameters a retrieve per coerenza