import os
import json
import logging
import re
from typing import Dict, Any, List
from datapizza.clients.openai import OpenAIClient
from datapizza.type import TextBlock
from rag.retrieval import AudioRAG
from rag.parameters_search import VectorParameterRetriever
from rag.enrichment import LabelEnricher
from config.settings import settings

logger = logging.getLogger(__name__)
for lib in ["httpcore", "httpx", "clap", "transformers", "pymongo","numba"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

class AudioFinderAgent:
    """Agente che cerca sample nel DB tramite testo"""
    def __init__(self):
        self.tool = AudioRAG()

    def run(self, query: str) -> List[Dict]:
        logger.info(f"Finder Agent: Ricerca audio per '{query}'...")
        return self.tool.retrieve(query, k=3)

class SoundDesignerAgent:
    """Agente che cerca parametri DSP"""
    def __init__(self):
        self.tool = VectorParameterRetriever()

    def run(self, query: str) -> Dict:
        logger.info(f"Sound Designer: Ricerca parametri per '{query}'...")
        descriptor = query.replace("Make it", "").replace("sound", "").replace("find", "").strip()
        return self.tool.get_parameters(descriptor)

class AudioAnalystAgent:
    """
    Agente di Analisi & Similarità (Hybrid).
    1. Analizza un file LOCALE (Enrichment + CLAP).
    2. Usa l'analisi per trovare file SIMILI nel DB (RAG).
    """
    def __init__(self):
        self.openai_client = OpenAIClient(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )
        logger.info("Init Analyst Agent (caricamento CLAP)...")
        self.enricher = LabelEnricher(self.openai_client)        
        self.rag_engine = AudioRAG()

    def run(self, file_path: str) -> Dict[str, Any]:
        clean_path = file_path.strip('"').strip("'")
        
        if not os.path.exists(clean_path):
            return {"error": f"File locale non trovato: {clean_path}"}
        
        filename = os.path.basename(clean_path)
        logger.info(f"Analizzando file locale: {filename}")

        try:
            name_no_ext = os.path.splitext(filename)[0]
            raw_tags = re.split(r'[-_\s]+', name_no_ext)
            
            # Generazione Caption & Verifica CLAP
            caption_result = self.enricher.enrich_tags(
                raw_tags=raw_tags,
                categories=["imported_sample"], 
                audio_path=clean_path
            )
            
            # Gestione flag allucinazioni
            is_hallucination = "[FLAGGED]" in caption_result
            clean_caption = caption_result.replace("[FLAGGED]", "").strip()
            
            # Similarity Search (RAG)
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
    """Traduce i dati JSON in linguaggio naturale."""
    def __init__(self):
        self.client = OpenAIClient(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )

    def generate_response(self, user_query: str, intent: str, data: Any) -> str:
        if intent == "ANALYSIS":
            system_prompt = (
                "Sei un Tecnico Audio AI. L'utente ti ha inviato un file audio locale per l'analisi.\n"
                "Il sistema ha generato una descrizione del file e ha trovato dei sample simili nel database.\n"
                "1. Descrivi il file analizzato (basandoti su 'generated_description').\n"
                "2. Presenta i file simili trovati nel DB ('similar_db_samples') come alternative utili.\n"
                "Usa un tono professionale."
            )
        else:
            system_prompt = (
                "Sei un AI Music Producer Assistant. Aiuta l'utente a trovare suoni e modificarli."
            )

        user_message = f"Query Utente: {user_query}\nDATI SISTEMA:\n{json.dumps(data, indent=2)}\nRispondi in italiano:"

        response = self.client.invoke(input=[
            TextBlock(content=system_prompt),
            TextBlock(content=user_message)
        ])
        
        return response.text