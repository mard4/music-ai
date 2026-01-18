import json
from typing import Dict, Any, List
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from jinja2 import Template
from config.settings import settings
from rag.utils import read_prompt, logger



class AudioFinderAgent:
    def __init__(self):
        # FIX: Uso la nuova classe AudioRetriever
        self.tool = AudioRetriever()

    def run(self, query: str) -> List[Dict]:
        logger.info(f"Finder Agent: Ricerca audio per '{query}'...")
        return self.tool.retrieve(query, k=3)