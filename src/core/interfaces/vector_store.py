from abc import ABC, abstractmethod
from typing import List
from core.domain.search import AudioSearchResult, SocialFXSearchResult

class VectorStoreRepository(ABC):
    """
    Interfaccia astratta per il database vettoriale.
    Disaccoppia la logica di business (RAG) dall'implementazione (Qdrant).
    """

    @abstractmethod
    async def search_audio(self, vector: List[float], vector_name: str = "text_vector") -> List[AudioSearchResult]:
        """
        Cerca sample audio simili.
        :param vector: Il vettore di embedding (1536 per testo, 512 per audio).
        :param limit: Numero di risultati.
        :param vector_name: Nome del vettore in Qdrant ('text_vector' o 'audio_vector').
        """
        pass

    @abstractmethod
    async def search_social_fx(self, vector: List[float]) -> List[SocialFXSearchResult]:
        """Cerca parametri DSP simili al vettore dato."""
        pass