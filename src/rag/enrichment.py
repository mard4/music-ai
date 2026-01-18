import logging
from pathlib import Path
from typing import List, Optional
from openai import OpenAI
from torch.nn.functional import cosine_similarity
import torch
from rag.clap.model_handler import CLAPModelHandler, create_clap_model
from config.settings import settings

logger = logging.getLogger(__name__)


def read_prompt(filename: str) -> str:
    """Legge il prompt dalla cartella prompts usando pathlib."""
    path = Path(__file__).parent / "prompts" / filename
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Impossibile leggere il prompt {filename}: {e}")
        # Fallback minimale in caso di errore I/O
        return "Raw Tags: {label}. Categories: {categories}."


class LabelEnricher:
    """
    Layer 1: Semantic Enrichment
    Trasforma tag rumorosi in descrizioni descrittive usando AudSem logic.
    """

    def __init__(self, openai_client: OpenAI, clap_handler: Optional[CLAPModelHandler] = None):
        self.client = openai_client
        self.model = settings.OPENAI_MODEL

        # Inizializza CLAP
        self.clap = clap_handler or create_clap_model(pretrained=True)

        # Carica il template dal file di testo
        self.prompt_template = read_prompt("clean_label.txt")

    def enrich_tags(self, raw_tags: List[str], categories: List[str], audio_path: str = None) -> str:
        """
        Genera una caption pulita dai tag grezzi.
        """
        # Costruzione del messaggio utente formattando il template
        try:
            user_content = self.prompt_template.format(
                label=", ".join(raw_tags),
                categories=", ".join(categories)
            )
        except Exception as e:
            logger.warning(f"Errore formattazione prompt: {e}")
            user_content = f"Tags: {raw_tags}, Categories: {categories}"

        try:
            # Chiamata nativa OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert audio taxonomist."},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3
            )
            generated_caption = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Enrichment fallito: {e}")
            # Fallback sui tag originali
            return ", ".join(raw_tags)

        # Validation Step (CLAP Hallucination Check)
        if audio_path and self._is_hallucination(generated_caption, audio_path):
            logger.warning(f"Flagged as hallucination: {generated_caption}")
            return f"[FLAGGED] {generated_caption}"

        return generated_caption

    def _is_hallucination(self, caption: str, audio_path: str, threshold: float = 0.3) -> bool:
        """
        Calcola la similarità coseno tra gli embedding CLAP dell'audio e della caption generata.
        Ritorna True se la similarità è sotto la soglia (indica possibile allucinazione).
        """
        try:
            # Usa il model handler esistente (src/clap/model_handler.py)
            audio_embed = self.clap.get_audio_embedding([audio_path])
            text_embed = self.clap.get_text_embedding([caption])

            if len(audio_embed) == 0 or len(text_embed) == 0:
                logger.warning("Embedding CLAP vuoti, salto validazione.")
                return False

            t_audio = torch.from_numpy(audio_embed)
            t_text = torch.from_numpy(text_embed)

            similarity = cosine_similarity(t_audio, t_text).item()
            return similarity < threshold
        except Exception as e:
            logger.error(f"CLAP validation failed: {e}")
            return False
