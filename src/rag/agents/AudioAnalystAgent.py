import json
from typing import Dict, Any, List
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from jinja2 import Template
from config.settings import settings
from rag.utils import read_prompt, logger


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