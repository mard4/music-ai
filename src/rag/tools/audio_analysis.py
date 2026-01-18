import logging
import re
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
    Responsabilità: Generare descrizioni da tag grezzi e validarle con CLAP.
    """

    def __init__(self, clap_handler: Optional[CLAPModelHandler] = None):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.clap = clap_handler or create_clap_model(pretrained=True)
        self.prompt = read_prompt("clean_label.txt")

    def enrich_and_verify(self, filename: str, audio_path: str) -> Dict[str, Any]:
        """Pipeline completa: Clean Name -> Generate Caption -> Validate."""

        clean_name = filename.replace(".wav", "").replace(".mp3", "")
        raw_tags = re.split(r'[-_\s]+', clean_name)

        # 2. Generazione Caption
        caption = self._generate_caption(raw_tags, ["imported_sample"])

        # 3. Validazione CLAP (Hallucination Check)
        is_hallucination = self._check_hallucination(caption, audio_path)

        return {
            "caption": caption,
            "raw_tags": raw_tags,
            "is_hallucination": is_hallucination,
            "status": "Low Confidence" if is_hallucination else "Verified"
        }

    def _generate_caption(self, raw_tags: List[str], categories: List[str]) -> str:
        try:
            prompt = self.prompt.format(
                label=", ".join(raw_tags),
                categories=", ".join(categories)
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert audio taxonomist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Enrichment Error: {e}")
            return ", ".join(raw_tags)

    def _check_hallucination(self, caption: str, audio_path: str, threshold: float = 0.25) -> bool:
        try:
            audio_embed = self.clap.get_audio_embedding([audio_path])
            text_embed = self.clap.get_text_embedding([caption])

            if len(audio_embed) == 0 or len(text_embed) == 0:
                return False

            sim = cosine_similarity(
                torch.from_numpy(audio_embed),
                torch.from_numpy(text_embed)
            ).item()

            return sim < threshold
        except Exception as e:
            logger.warning(f"CLAP Check skipped: {e}")
            return False