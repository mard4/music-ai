import asyncio
import os
import sys
import logging
from pathlib import Path

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import datapizza components
from qdrant_client import QdrantClient
from openai import OpenAI

from config.settings import settings
from core.infrastructure.database.dependecies import get_mongo_client

# Configurazione Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def ingest_socialfx_vectors():
    # 1. Connessione a MongoDB (client asincrono)
    client_mongo = get_mongo_client()
    db = client_mongo[os.getenv("MONGODB_DATABASE_NAME", "rag_audio_db")]
    collection = db["socialfx_kb"]

    # Usa find() sincrono sul client asincrono
    docs = []
    async for doc in collection.find({}):
        docs.append(doc)
    
    logger.info(f"Trovati {len(docs)} descrittori in MongoDB")
    
    if not docs:
        logger.warning("Nessun documento trovato in MongoDB. Uscita.")
        client_mongo.close()
        return

    # 2. Configurazione Qdrant (usa client nativo)
    qdrant = QdrantClient(host="localhost", port=6333)
    collection_name = "socialfx_vectors"

    # Ricrea la collezione senza named vectors (compatibile con formato vector semplice)
    try:
        qdrant.delete_collection(collection_name)
        logger.info(f"Collezione '{collection_name}' eliminata")
    except Exception:
        pass

    from qdrant_client.models import VectorParams, Distance
    try:
        # Crea la collezione con vettore singolo (unnamed) usando vectors_config
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        logger.info(f"Collezione '{collection_name}' ricreata (unnamed vector)")
    except Exception as e:
        logger.info(f"Collezione '{collection_name}' già esiste: {e}")

    # Assume single unnamed vector format; upload points with key 'vector'
    use_named_vectors = False

    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # 4. Creazione Punti (con batch di embedding)
    points = []
    batch_size = 50
    
    logger.info(f"Creazione {len(docs)} embedding in batch di {batch_size}...")
    
    for batch_start in range(0, len(docs), batch_size):
        batch_end = min(batch_start + batch_size, len(docs))
        batch_docs = docs[batch_start:batch_end]
        
        # Batch di descrittori
        descriptors = [doc.get("descriptor", "") for doc in batch_docs]
        
        try:
            # Embedding batch
            response = openai_client.embeddings.create(
                input=descriptors,
                model="text-embedding-3-small"
            )
            
            for idx_in_batch, (doc, embedding_data) in enumerate(zip(batch_docs, response.data)):
                vector = embedding_data.embedding
                idx = batch_start + idx_in_batch
                
                payload = {
                    "descriptor": doc.get("descriptor", ""),
                    "effect_type": doc.get("effect_type", "eq"),
                    "param_values": doc.get("parameters", {}).get("param_values", []),
                    "param_keys": doc.get("parameters", {}).get("param_keys", [])
                }
                
                if use_named_vectors:
                    points.append({
                        "id": idx + 1,
                        "vectors": {"embedding": vector},
                        "payload": payload
                    })
                else:
                    points.append({
                        "id": idx + 1,
                        "vector": vector,
                        "payload": payload
                    })
            
            logger.info(f"[{batch_end}/{len(docs)}] Batch elaborato")
            
        except Exception as e:
            logger.error(f"Errore batch embedding [{batch_start}-{batch_end}]: {e}")

    # 5. Caricamento su Qdrant
    if points:
        logger.info(f"Caricamento {len(points)} punti su Qdrant...")
        try:
            qdrant.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"✓ Indicizzati {len(points)} parametri su Qdrant.")
        except Exception as e:
            logger.error(f"Errore upsert su Qdrant: {e}", exc_info=True)
    else:
        logger.warning("Nessun punto creato. Verifica i dati in MongoDB.")

    client_mongo.close()

if __name__ == "__main__":
    asyncio.run(ingest_socialfx_vectors())