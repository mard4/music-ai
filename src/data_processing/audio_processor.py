import logging
from urllib.parse import urlparse, parse_qs
from core.infrastructure.database.dependecies import get_audio_repository

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def samplefocus_fix():
    """
    Script avanzato di manutenzione post estrazione:
    1. Trova le categorie 'sporche' (es. con query params).
    2. Estrae i tag semantici (timbri) dai parametri URL.
    3. Sposta i timbri nel campo 'metadata.categories'.
    4. Pulisce il campo 'metadata.main_category'.
    """

    try:
        repo = get_audio_repository()
        collection = repo.collection
        logger.info(f"Repository inizializzato su collection: {collection.name}")
        total_docs = await collection.count_documents({})
        logger.info(f"Totale documenti nella collezione '{collection.name}': {total_docs}")

    except Exception as e:
        logger.error(f"Errore inizializzazione repository: {e}")
        return

    # Query: Trova documenti con '?' in main_category
    query_filter = {"metadata.main_category": {"$regex": "\?"}}

    # Usiamo la collection raw per il find, poiché il repository standard filtra per uguaglianza
    cursor = collection.find(query_filter)
    count = await collection.count_documents(query_filter)

    logger.info(f"Trovati {count} documenti da processare. Inizio estrazione timbri...")

    updated_count = 0

    async for doc in cursor:
        try:
            doc_id = str(doc["_id"])
            metadata = doc.get('metadata', {})
            dirty_string = metadata.get('main_category', "")
            current_categories = metadata.get('categories', []) or []

            if '?' in dirty_string:
                # Gestione URL parziali
                if not dirty_string.startswith("http"):
                    fake_url = f"http://fake.com/{dirty_string}"
                else:
                    fake_url = dirty_string

                parsed = urlparse(fake_url)
                query_params = parse_qs(parsed.query)

                # --- B. Estrazione Timbri (Tags) ---
                extracted_tags = query_params.get('tags[]', [])
                new_timbres = [t.lower().strip() for t in extracted_tags]

                if new_timbres:
                    logger.debug(f"Estratti timbri per {doc_id}: {new_timbres}")

                # --- C. Pulizia Category ---
                clean_slug = parsed.path.rstrip("/").split("/")[-1]

                # --- D. Aggiornamento Documento ---

                # Unione e deduplicazione categorie esistenti + nuove
                final_categories = list(set(current_categories + new_timbres))

                update_data = {
                    "metadata.main_category": clean_slug,
                    "metadata.categories": final_categories
                }

                # Utilizzo del metodo standard del repository
                success = await repo.update_audio_file(doc_id, update_data)

                if success:
                    updated_count += 1
                    if updated_count % 100 == 0:
                        logger.info(f"Processati {updated_count}/{count}...")
                else:
                    logger.warning(f"Fallito aggiornamento per {doc_id}")

        except Exception as e:
            logger.error(f"Errore processando documento {doc.get('_id')}: {e}")

    logger.info(f"{updated_count} documenti aggiornati e puliti.")
