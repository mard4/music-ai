import logging
import numpy as np
from typing import Dict, Any, List, Optional
from qdrant_client import QdrantClient
from openai import OpenAI

from config.settings import settings
from core.domain.search import SocialFXSearchResult
from core.infrastructure.database.dependecies import get_vector_repository

logger = logging.getLogger(__name__)


class DSPSearchTool:
    """
    TOOL: DSP Parameter Retrieval (SocialFX).
    Responsabilità: Trovare i parametri tecnici e creare interpolazioni tra concetti.
    """

    def __init__(self, qdrant_client: Optional[QdrantClient] = None, openai_client: Optional[OpenAI] = None):
        self.qdrant = qdrant_client or QdrantClient(
            host=settings.QDRANT_CONNECTION_HOST,
            port=settings.QDRANT_PORT
        )
        self.vector_repo = get_vector_repository()
        self.openai = OpenAI(api_key=settings.MODEL_API_KEY)
        self.embedding_model = settings.MODEL_EMBEDDING_MODEL

    async def search_parameters(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Cerca i parametri semanticamente simili.
        Ritorna una LISTA di risultati per permettere la creazione di catene di effetti.
        """
        try:
            emb_res = self.openai.embeddings.create(
                input=query,
                model=self.embedding_model
            )
            vector = emb_res.data[0].embedding

            results: List[SocialFXSearchResult] = await self.vector_repo.search_social_fx(
                vector=vector,
            )

            if not results:
                logger.warning(f"Nessun parametro trovato per: '{query}'")
                return []

            mapped_results = []
            for res in results:
                mapped_results.append({
                    "descriptor": res.descriptor,
                    "score": res.score,
                    "effect_type": res.effect_type,
                    "param_values": res.param_values,
                    "param_keys": res.param_keys
                })

            return mapped_results

        except Exception as e:
            logger.error(f"Errore ricerca parametri DSP: {e}")
            return []

    async def blend_parameters(self, queries: List[str]) -> Dict[str, Any]:
        """
        Interpola (fa la media) dei parametri tra più descrittori.
        """
        results = []
        logger.info(f"Blending parametri per: {queries}")

        for q in queries:
            # Usiamo il nuovo metodo con limit=1 perché per il blending
            # vogliamo il match esatto di ogni termine
            res_list = await self.search_parameters(q, limit=1)
            if res_list:
                results.append(res_list[0])

        if not results:
            return {}

        if len(results) == 1:
            return results[0]

        # 2. Controllo Compatibilità
        reference = results[0]
        ref_effect = reference.get("effect_type")
        ref_keys = reference.get("param_keys")

        valid_params_matrix = []
        blended_descriptors = []

        for r in results:
            if r.get("effect_type") == ref_effect and r.get("param_keys") == ref_keys:
                valid_params_matrix.append(r.get("param_values"))
                blended_descriptors.append(r.get("descriptor"))
            else:
                logger.warning(
                    f"Escluso '{r.get('descriptor')}' dal blending: tipo effetto incompatibile")

        if not valid_params_matrix:
            return {}

        try:
            # 3. Calcolo della Media
            matrix = np.array(valid_params_matrix)
            blended_values = np.mean(matrix, axis=0).tolist()

            return {
                "descriptor": " + ".join(blended_descriptors),
                "score": 1.0,
                "effect_type": ref_effect,
                "param_values": blended_values,
                "param_keys": ref_keys,
                "is_blended": True
            }

        except Exception as e:
            logger.error(f"Errore calcolo blending numpy: {e}")
            return reference