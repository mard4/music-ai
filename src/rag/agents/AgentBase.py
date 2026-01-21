import json
from typing import Optional, List, Dict, Any, Union

from jinja2 import Template
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from config.settings import settings
from rag.utils import read_prompt, logger


class AgentBase:
    """
    Classe base generica per agenti che utilizzano OpenAI.
    Gestisce l'inizializzazione del client, del modello e il caricamento del prompt.
    """

    def __init__(self,
                 prompt_file: Optional[str] = None,
                 tools: Optional[List[Any]] = None,
                 context: Optional[Dict[str, Any]] = None,
                 agent_name: Optional[str] = None):
        self.client = OpenAI(api_key=settings.MODEL_API_KEY)
        self.model = settings.MODEL_MODEL
        self.tools = tools
        self.logger = logger
        self.context = context or {}
        self.agent_name = agent_name
        self.prompt = prompt_file

        if prompt_file:
            try:
                self.prompt = read_prompt(prompt_file)
            except Exception as e:
                self.logger.error(f"Errore caricamento prompt '{prompt_file}': {e}")
                self.prompt = ""
        else:
            self.prompt = None

    def call_llm(
            self,
            messages: List[ChatCompletionMessageParam],
            temperature: float = 0.7,
            json_mode: bool = False,
            tool_choice: str = "auto"
    ) -> Union[str, Dict[str, Any], None]:
        """
        Wrapper centrale per le chiamate a OpenAI.
        Gestisce automaticamente tools e JSON mode.
        """
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if self.tools:
            params["tools"] = self.tools
            params["tool_choice"] = tool_choice

        if json_mode:
            params["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**params)
            message = response.choices[0].message

            # Caso 1: L'LLM ha deciso di chiamare un tool
            if message.tool_calls:
                return message.tool_calls  # Ritorna la lista delle chiamate ai tool

            # Caso 2: Risposta testuale (o JSON string)
            content = message.content or ""

            if json_mode:
                return self._safe_json_parse(content)

            return content

        except Exception as e:
            self.logger.error(f"Errore chiamata LLM ({self.__class__.__name__}): {e}", exc_info=True)
            return None

    def _safe_json_parse(self, content: str) -> Dict[str, Any]:
        """Helper per parsare JSON in sicurezza."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            self.logger.warning(f"Fallito parsing JSON. Raw content: {content}")
            return {}

    def get_system_message(self) -> Dict[str, str]:
        """Restituisce il messaggio di sistema formattato per OpenAI."""
        if not self.prompt:
            return {"role": "system", "content": "You are a helpful assistant."}
        return {"role": "system", "content": self.prompt}

    def render_prompt(self, **kwargs) -> str:
        """
        Unisce il context globale (self.context) con i parametri dinamici (kwargs).
        I parametri passati qui (es. user_query) vincono sul context globale in caso di conflitti.
        """
        if not self.prompt:
            return ""

        try:
            # 2. Merge dei dizionari: kwargs sovrascrive self.context se ci sono chiavi uguali
            full_context = {**self.context, **kwargs}

            return Template(self.prompt).render(**full_context)
        except Exception as e:
            self.logger.warning(f"Errore rendering template: {e}")
            return self.prompt