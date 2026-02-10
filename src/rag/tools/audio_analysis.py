import logging
import re
import json  # <--- Serve per il parsing
from typing import List, Dict, Any, Optional
import torch
from jinja2 import Template
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
        self.client = OpenAI(api_key=settings.MODEL_API_KEY)
        self.model = settings.MODEL_MODEL
        self.clap = clap_handler or create_clap_model(pretrained=True)
        self.clean_prompt = read_prompt("clean_label.txt")
        self.synthesis_prompt = read_prompt("analysis_synthesis.txt")

    def enrich_and_verify(self,
                          filename: str,
                          audio_path: str,
                          label: Optional[str] = None,
                          main_category: Optional[str] = None,
                          original_tags: List[str] = None) -> Dict[str, Any]:
        """
        Pipeline:
        1. Analisi LLM (Caption + Tag Filtering)
        2. Validazione CLAP
        """

        # Se non passiamo tag dal DB, usiamo quelli estratti dal nome come fallback
        if not original_tags:
            original_tags = re.split(r'[-_\s]+', label)

        # 1. Generazione (Caption + Smart Tags)
        llm_result = self._generate_metadata(label=label,
                                             tags=original_tags,
                                             main_category=main_category,
                                             )

        new_label = llm_result.get("caption", label)
        ai_tags = llm_result.get("ai_tags", original_tags)

        # 2. Validazione CLAP (Hallucination Check)
        is_hallucination, score = self._check_hallucination(new_label=new_label, original_label=label, audio_path=audio_path)

        return {
            "caption": new_label,
            "ai_tags": ai_tags,
            "original_tags": original_tags,
            "is_hallucination": is_hallucination,
            "clap_score": score,
            "status": "Verified" if not is_hallucination else "Low Confidence"
        }

    def _generate_metadata(self,
                           label: str,
                           main_category: Optional[str],
                           tags: List[str]) -> Dict[str, Any]:
        """Chiede all'LLM di generare caption e filtrare i tag."""
        try:
            prompt = Template(self.clean_prompt).render(
                label=label,
                tags=", ".join(tags),
                main_category=main_category
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert audio taxonomist. Output only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content.strip()
            return json.loads(content)

        except Exception as e:
            logger.error(f"Enrichment LLM Error: {e}")
            return {"caption": label, "ai_tags": tags}

    def _check_hallucination(self,
                             audio_path: str,
                             new_label: str,
                             original_label: str, threshold: float = 0.25):
        try:
            if not audio_path: return False, 0.0

            audio_embed = self.clap.get_audio_embedding([audio_path])
            text_embed = self.clap.get_text_embedding([new_label])

            if len(audio_embed) == 0 or len(text_embed) == 0: return False, 0.0

            sim = cosine_similarity(
                torch.from_numpy(audio_embed),
                torch.from_numpy(text_embed)
            ).item()

            return sim < threshold, sim

        except Exception as e:
            logger.warning(f"CLAP Check skipped: {e}")
            return False, 0.0

    def predict_label_from_neighbors(self,
                                     filename: str,
                                     original_label: str,
                                     audio_path: str,
                                     neighbors: List[Dict[str, Any]]) -> Dict[
        str, Any]:
        """
        PIPELINE REVERSE-RAG (Audio-First):
        1. Sintesi label basata sui 'vicini' trovati nel DB.
        2. Validazione CLAP (Hallucination Check) usando il metodo condiviso.
        """
        # 1. Costruzione Contesto per LLM
        if neighbors:
            logger.info(f"DEBUG STRUCTURE - First Neighbor: {neighbors[0]}")
        else:
            logger.warning("Neighbors list is empty!")

        context_text = ""
        for i, n in enumerate(neighbors):
            label = n.get('label') or n.get('description') or "N/A"
            tags = n.get('ai_tags') or n.get('tags') or []
            similarity_score = n.get('score', 0.0)
            quality_score = n.get('clap_score', 0.0)
            context_text += (
                f"{i + 1}. Label: '{label}' | Tags: {tags} | "
                f"Sim: {similarity_score:.4f} | LabelQuality: {quality_score:.2f}\n"
            )

        # 2. Chiamata LLM (Sintesi)
        try:
            full_prompt = Template(self.synthesis_prompt).render(similar_samples_context=context_text)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert audio taxonomist. Output valid JSON."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result_json = json.loads(content)

        except Exception as e:
            logger.error(f"Sintesi LLM fallita: {e}")
            return {
                "generated_label": "Analysis Failed",
                "confidence": "None",
                "reasoning": str(e)
            }

        # 3. Validazione CLAP (Hallucination Check)
        generated_label = result_json.get("generated_label", "")

        if generated_label and original_label:
            is_hallucination, label_clap_score = self._check_hallucination(new_label=generated_label, original_label=original_label, audio_path=audio_path)

            # Arricchiamo il JSON con i dati di verifica
            result_json["clap_score"] = round(label_clap_score, 4)
            result_json["is_hallucination"] = is_hallucination

            if is_hallucination:
                result_json["confidence"] = "Low (Audio Mismatch)"
                result_json["hallucination_warning"] = True
            else:
                result_json["hallucination_warning"] = False

        return result_json
