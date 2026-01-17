from typing import List, Optional
import logging
import numpy as np
from torch.nn.functional import cosine_similarity
from datapizza.clients.openai import OpenAIClient
from datapizza.type import TextBlock
from clap.model_handler import CLAPModelHandler, create_clap_model
from pathlib import Path
logger = logging.getLogger(__name__)

def read_txt_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

CLEAN_LABELS_COLLECTION = "clean_audio_labels"
file = Path(__file__).parent / "prompts" / "clean_label.txt"
prompt = read_txt_file(str(file))


class LabelEnricher:
    """
    Layer 1: Semantic Enrichment
    Transforms noisy tags into descriptive captions using the AudSem reasoning pipeline.
    """

    def __init__(self, openai_client: OpenAIClient, clap_handler: CLAPModelHandler = None):
        self.client = openai_client
        self.clap = clap_handler or create_clap_model(pretrained=True)
        self.prompt_template = prompt

    def enrich_tags(self, raw_tags: List[str], categories: List[str], audio_path: str = None) -> str:
        """
        Applies AudSem logic (Source -> Attribute -> Action) to generate a caption.
        """
        prompt = self.prompt_template.format(
            label=", ".join(raw_tags),
            categories=", ".join(categories)
        )

        response = self.client.invoke(input=[TextBlock(content=prompt)])
        generated_caption = response.text.strip()

        # Validation Step
        if audio_path and self._is_hallucination(generated_caption, audio_path):
            logger.warning(f"Flagged as hallucination: {generated_caption}")
            return f"[FLAGGED] {generated_caption}"

        return generated_caption

    def _is_hallucination(self, caption: str, audio_path: str, threshold: float = 0.3) -> bool:
        """
        Calculates cosine similarity between CLAP embeddings of audio and text.
        Returns True if similarity < threshold.
        """
        try:
            # Placeholder: Get embeddings using the CLAP handler
            # Note: actual implementation depends on specific CLAP library signature
            audio_embed = self.clap.model.get_audio_embedding([audio_path])
            text_embed = self.clap.model.get_text_embedding([caption])

            # Convert to tensors if needed and calculate similarity
            similarity = cosine_similarity(audio_embed, text_embed).item()
            return similarity < threshold
        except Exception as e:
            logger.error(f"CLAP validation failed: {e}")
            return False