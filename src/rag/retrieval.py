# rag/retrieval.py
from typing import List, Dict
from datapizza.vectorstores.qdrant import QdrantVectorstore
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.pipeline import DagPipeline
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class AudioRAG:
    """
    Agente di Recupero: Catalogatore Audio
    Cerca solo nella collezione dei file audio (audio_vectors).
    """

    def __init__(self):
        # PUNTA ALLA NUOVA COLLEZIONE AUDIO
        self.collection_name = "audio_vectors"
        
        self.vectorstore = QdrantVectorstore(host="localhost", port=6333)
        self.embedder = OpenAIEmbedder(
            api_key=settings.OPENAI_API_KEY, 
            model_name="text-embedding-3-small"
        )

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Recupera i file audio più simili alla query."""
        pipeline = DagPipeline()
        pipeline.add_module("embedder", self.embedder)
        pipeline.add_module("retriever", self.vectorstore.as_retriever(
            collection_name=self.collection_name,
            k=k,
            vector_name=""
        ))
        pipeline.connect("embedder", "retriever", target_key="query_vector")

        try:
            result = pipeline.run({
                "embedder": {"text": query},
                "retriever": {
                    "collection_name": self.collection_name, 
                    "k": k, 
                    "vector_name": ""
                }
            })

            chunks = result["retriever"]
            results = []
            
            for chunk in chunks:
                meta = getattr(chunk, "metadata", {})
                results.append({
                    "type": "audio_file", # Tag utile per l'Orchestrator
                    "score": getattr(chunk, "score", 0.0),
                    "filename": meta.get("filename"),
                    "label": meta.get("label"),
                    "bpm": meta.get("bpm"),
                    "key": meta.get("key"),
                    "tags": meta.get("categories")
                })
            
            return results

        except Exception as e:
            logger.error(f"Errore Audio RAG: {e}")
            return []