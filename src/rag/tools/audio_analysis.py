import logging
import re
import json  # <--- Serve per il parsing
from typing import List, Dict, Any, Optional
import torch
from torch.nn.functional import cosine_similarity
from openai import OpenAI

from config.settings import settings
from rag.clap.model_handler import CLAPModelHandler, create_clap_model
from rag.utils import read_prompt, logger


class LabelEnricherTool:
    """
    TOOL: Semantic Audio Enrichment.
    """

    def __init__(self, clap_handler: Optional[CLAPModelHandler] = None):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.clap = clap_handler or create_clap_model(pretrained=True)
        self.prompt = read_prompt("clean_label.txt")

    def enrich_and_verify(self, filename: str, audio_path: str, original_tags: List[str] = None) -> Dict[str, Any]:
        """
        Pipeline:
        1. Analisi LLM (Caption + Tag Filtering)
        2. Validazione CLAP
        """
        clean_name = filename.replace(".wav", "").replace(".mp3", "")

        # Se non passiamo tag dal DB, usiamo quelli estratti dal nome come fallback
        if not original_tags:
            original_tags = re.split(r'[-_\s]+', clean_name)

        # 1. Generazione (Caption + Smart Tags)
        llm_result = self._generate_metadata(clean_name, original_tags)

        caption = llm_result.get("caption", clean_name)
        smart_tags = llm_result.get("smart_tags", original_tags)

        # 2. Validazione CLAP (Hallucination Check)
        is_hallucination, score = self._check_hallucination(caption, audio_path)

        return {
            "caption": caption,
            "smart_tags": smart_tags,
            "original_tags": original_tags,
            "is_hallucination": is_hallucination,
            "clap_score": score,
            "status": "Verified" if not is_hallucination else "Low Confidence"
        }

    def _generate_metadata(self, label: str, tags: List[str]) -> Dict[str, Any]:
        """Chiede all'LLM di generare caption e filtrare i tag."""
        try:
            prompt = self.prompt.format(
                label=label,
                tags=", ".join(tags)
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert audio taxonomist. Output only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content.strip()
            return json.loads(content)

        except Exception as e:
            logger.error(f"Enrichment LLM Error: {e}")
            # Fallback in caso di errore
            return {"caption": label, "smart_tags": tags}

    def _check_hallucination(self, caption: str, audio_path: str, threshold: float = 0.25):
        try:
            if not audio_path: return False, 0.0

            audio_embed = self.clap.get_audio_embedding([audio_path])
            text_embed = self.clap.get_text_embedding([caption])

            if len(audio_embed) == 0 or len(text_embed) == 0: return False, 0.0

            sim = cosine_similarity(
                torch.from_numpy(audio_embed),
                torch.from_numpy(text_embed)
            ).item()

            return sim < threshold, sim

        except Exception as e:
            logger.warning(f"CLAP Check skipped: {e}")
            return False, 0.0