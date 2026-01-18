import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from jinja2 import Template
from rag.retrieval import AudioRetriever, ParameterRetriever
from rag.enrichment import LabelEnricher
from config.settings import settings

logger = logging.getLogger(__name__)
for lib in ["httpcore", "httpx", "clap", "transformers", "pymongo", "numba", "qdrant_client"]:
    logging.getLogger(lib).setLevel(logging.WARNING)


def read_prompt(filename: str) -> str:
    path = Path(__file__).parent / "prompts" / filename
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Impossibile leggere il prompt {filename}: {e}")
        return ""


class IntentClassifierAgent:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.raw_template = read_prompt("intent_classification.txt")

    def run(self, user_input: str) -> Dict[str, str]:
        logger.info(f"Classificando intento per: '{user_input}'")
        try:
            prompt_content = Template(self.raw_template).render(user_query=user_input)
            messages: List[ChatCompletionMessageParam] = [
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": prompt_content}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            return {
                "intent": data.get("intent", "RETRIEVAL_MIX"),
                "params": data.get("params", user_input)
            }
        except Exception as e:
            logger.error(f"Errore classificazione intento: {e}")
            return {"intent": "RETRIEVAL_MIX", "params": user_input}


class AudioFinderAgent:
    def __init__(self):
        # FIX: Uso la nuova classe AudioRetriever
        self.tool = AudioRetriever()

    def run(self, query: str) -> List[Dict]:
        logger.info(f"Finder Agent: Ricerca audio per '{query}'...")
        return self.tool.retrieve(query, k=3)


class SoundDesignerAgent:
    def __init__(self):
        # FIX: Uso la nuova classe ParameterRetriever
        self.tool = ParameterRetriever()

    def run(self, query: str) -> Dict:
        logger.info(f"Sound Designer: Ricerca parametri per '{query}'...")
        return self.tool.retrieve(query)  # Nota: metodo rinominato da get_parameters a retrieve per coerenza


class AudioAnalystAgent:
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info("Init Analyst Agent (caricamento CLAP)...")
        self.enricher = LabelEnricher(openai_client=self.openai_client)
        self.rag_engine = AudioRetriever()

    def run(self, file_path: str) -> Dict[str, Any]:
        clean_path = file_path.strip('"').strip("'")

        if not os.path.exists(clean_path):
            return {"error": f"File locale non trovato: {clean_path}"}

        filename = os.path.basename(clean_path)
        logger.info(f"Analizzando file locale: {filename}")

        try:
            name_no_ext = os.path.splitext(filename)[0]
            raw_tags = re.split(r'[-_\s]+', name_no_ext)

            caption_result = self.enricher.enrich_tags(
                raw_tags=raw_tags,
                categories=["imported_sample"],
                audio_path=clean_path
            )

            is_hallucination = "[FLAGGED]" in caption_result
            clean_caption = caption_result.replace("[FLAGGED]", "").strip()

            logger.info(f"Cerco file simili nel DB per: '{clean_caption}'")
            similar_files = self.rag_engine.retrieve(clean_caption, k=3)

            return {
                "source": "LOCAL_FILE",
                "filename": filename,
                "analysis": {
                    "generated_description": clean_caption,
                    "clap_status": "Low Confidence" if is_hallucination else "Verified",
                    "inferred_tags": raw_tags
                },
                "similar_db_samples": similar_files
            }

        except Exception as e:
            logger.error(f"Errore analisi: {e}", exc_info=True)
            return {"error": str(e)}


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