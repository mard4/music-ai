import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Any
from abc import ABC, abstractmethod

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI

from config.settings import settings
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BaseIngestor(ABC):
    """
    Classe base astratta per l'ingestion dei dati in Qdrant.
    Gestisce la configurazione dei client e la creazione delle collezioni.
    """

    def __init__(self, collection_name: str, vector_size: int = 1536):
        self.collection_name = collection_name
        self.vector_size = vector_size

        # Inizializzazione Client
        self.qdrant = QdrantClient(
            host=settings.QDRANT_CONNECTION_HOST,
            port=settings.QDRANT_PORT
        )
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def _prepare_collection(self):
        """Ricrea la collezione su Qdrant."""
        if self.qdrant.collection_exists(self.collection_name):
            self.qdrant.delete_collection(self.collection_name)
            logger.info(f"Collezione '{self.collection_name}' esistente eliminata.")

        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Collezione '{self.collection_name}' creata.")

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Genera embedding in batch usando OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                input=texts,
                model=settings.OPENAI_EMBEDDING_MODEL
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Errore generazione embedding: {e}")
            return []

    async def _upsert_batch(self, points: List[PointStruct]):
        """Carica un batch di punti su Qdrant."""
        if points:
            try:
                self.qdrant.upsert(collection_name=self.collection_name, points=points)
                logger.info(f"Upsert di {len(points)} punti completato.")
            except Exception as e:
                logger.error(f"Errore upsert Qdrant: {e}")

    @abstractmethod
    async def run(self):
        """Metodo principale da implementare nelle sottoclassi."""
        pass
