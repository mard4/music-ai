from abc import ABC, abstractmethod
from typing import List
from core.domain.search import AudioSearchResult, SocialFXSearchResult

class VectorStoreRepository(ABC):
    """
    Interfaccia astratta per il database vettoriale.
    Disaccoppia la logica di business (RAG) dall'implementazione (Qdrant).
    """

    @abstractmethod
    async def search_audio(self, vector: List[float], limit: int) -> List[AudioSearchResult]:
        """Cerca sample audio simili al vettore dato."""
        pass

    @abstractmethod
    async def search_social_fx(self, vector: List[float], limit: int) -> List[SocialFXSearchResult]:
        """Cerca parametri DSP simili al vettore dato."""
        pass