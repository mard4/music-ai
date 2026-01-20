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

    async def find_parameters(self, query: str) -> Dict[str, Any]:
        """Cerca i parametri per un singolo concetto semantico."""
        try:
            emb_res = self.openai.embeddings.create(
                input=query,
                model=self.embedding_model
            )
            vector = emb_res.data[0].embedding

            results: List[SocialFXSearchResult] = await self.vector_repo.search_social_fx(
                vector=vector,
                limit=1
            )

            if not results:
                logger.warning(f"Nessun parametro trovato per: '{query}'")
                return {}

            # 3. Mapping (Domain Object -> Dict per l'Agent)
            best_match = results[0]

            return {
                "descriptor": best_match.descriptor,
                "score": best_match.score,
                "effect_type": best_match.effect_type,
                "params": best_match.param_values,
                "keys": best_match.param_keys
            }

        except Exception as e:
            logger.error(f"Errore ricerca parametri DSP: {e}")
            return {}

    async def blend_parameters(self, queries: List[str]) -> Dict[str, Any]:
        """
        Interpola (fa la media) dei parametri tra più descrittori.
        Esempio: blend_parameters(["warm", "bright"]) -> Parametri mediati.
        """
        results = []
        logger.info(f"Blending parametri per: {queries}")

        # 1. Recupero dei singoli preset
        for q in queries:
            res = await self.find_parameters(q)
            if res:
                results.append(res)

        if not results:
            return {}

        # Se abbiamo trovato un solo risultato valido, ritorniamo quello senza media
        if len(results) == 1:
            return results[0]

        # 2. Controllo Compatibilità (Devono essere lo stesso effetto, es. EQ)
        # Prendiamo il primo come riferimento
        reference = results[0]
        ref_effect = reference.get("effect_type")
        ref_keys = reference.get("keys")

        valid_params_matrix = []
        blended_descriptors = []

        for r in results:
            # Verifica che l'effetto e le chiavi dei parametri coincidano
            if r.get("effect_type") == ref_effect and r.get("keys") == ref_keys:
                valid_params_matrix.append(r.get("params"))
                blended_descriptors.append(r.get("descriptor"))
            else:
                logger.warning(
                    f"Escluso '{r.get('descriptor')}' dal blending: tipo effetto incompatibile ({r.get('effect_type')})")

        if not valid_params_matrix:
            return {}

        try:
            # 3. Calcolo della Media (Interpolazione)
            # Trasforma in matrice numpy per calcolare la media colonna per colonna
            matrix = np.array(valid_params_matrix)

            # axis=0 significa che facciamo la media "verticale" (parametro per parametro)
            # Esempio: [LowGain1, MidGain1] + [LowGain2, MidGain2] -> [AvgLowGain, AvgMidGain]
            blended_values = np.mean(matrix, axis=0).tolist()

            return {
                "descriptor": " + ".join(blended_descriptors),  # Es. "Warm + Bright"
                "score": 1.0,  # Score sintetico
                "effect_type": ref_effect,
                "params": blended_values,
                "keys": ref_keys,
                "is_blended": True
            }

        except Exception as e:
            logger.error(f"Errore calcolo blending numpy: {e}")
            return reference  # Fallback al primo risultato