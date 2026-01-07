# read the original label from audio file and extract correspolnding instrument and timbre

import asyncio
import logging
from urllib.parse import urlparse, parse_qs
from motor.motor_asyncio import AsyncIOMotorClient
from config.settings import settings

# Configurazione Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def fix_and_extract_timbres():
    """
    Script avanzato di manutenzione:
    1. Trova le categorie 'sporche' (es. con query params).
    2. Estrae i tag semantici (timbri) dai parametri URL.
    3. Sposta i timbri nel campo 'metadata.categories'.
    4. Pulisce il campo 'metadata.main_category'.
    """

    # 1. Connessione al Database
    try:
        connection_string = settings.database.mongodb_connection_string
        db_name = settings.database.mongodb_database_name
        collection_name = settings.database.mongodb_audio_collection

        client = AsyncIOMotorClient(connection_string)
        db = client[db_name]
        collection = db[collection_name]

        logger.info(f"Connesso a {db_name}.{collection_name}")

    except Exception as e:
        logger.error(f"Errore di connessione: {e}")
        return

    # 2. Query: Trova documenti con '?' in main_category
    query_filter = {"metadata.main_category": {"$regex": "\?"}}

    cursor = collection.find(query_filter)
    count = await collection.count_documents(query_filter)

    logger.info(f"Trovati {count} documenti da processare. Inizio estrazione timbri...")

    updated_count = 0

    async for doc in cursor:
        try:
            metadata = doc.get('metadata', {})
            dirty_string = metadata.get('main_category', "")

            if '?' in dirty_string:
                # --- A. Parsing dell'URL ---
                # Aggiungiamo un fake domain per far funzionare urlparse se manca
                if not dirty_string.startswith("http"):
                    fake_url = f"http://fake.com/{dirty_string}"
                else:
                    fake_url = dirty_string

                parsed = urlparse(fake_url)
                query_params = parse_qs(parsed.query)

                # --- B. Estrazione Timbri (Tags) ---
                # SampleFocus usa 'tags[]' come chiave per i timbri
                extracted_tags = query_params.get('tags[]', [])

                # Normalizziamo i tag (lowercase, strip)
                new_timbres = [t.lower().strip() for t in extracted_tags]

                if new_timbres:
                    logger.debug(f"Estratti timbri per {doc.get('_id')}: {new_timbres}")

                # --- C. Pulizia Category ---
                # La path contiene la categoria (es. /categories/analog-synth-bass)
                clean_slug = parsed.path.rstrip("/").split("/")[-1]

                # --- D. Aggiornamento Documento ---
                update_ops = {
                    "$set": {
                        "metadata.main_category": clean_slug
                    }
                }

                # Se abbiamo trovato timbri, li aggiungiamo a metadata.categories
                # Usiamo $addToSet con $each per evitare duplicati
                if new_timbres:
                    update_ops["$addToSet"] = {
                        "metadata.categories": {"$each": new_timbres}
                    }

                await collection.update_one(
                    {"_id": doc["_id"]},
                    update_ops
                )

                updated_count += 1
                if updated_count % 100 == 0:
                    logger.info(f"Processati {updated_count}/{count}...")

        except Exception as e:
            logger.error(f"Errore processando documento {doc.get('_id')}: {e}")

    logger.info(f"Finito! {updated_count} documenti aggiornati e puliti.")
    client.close()


if __name__ == "__main__":
    asyncio.run(fix_and_extract_timbres())