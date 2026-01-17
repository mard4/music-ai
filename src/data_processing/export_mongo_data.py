import asyncio
import os
import json
from datetime import datetime
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "audio_db"

# Export Audio (GridFS)
FS_BUCKET_NAME = "audio_files"  # La collezione GridFS (es. audio_files.files / audio_files.chunks)
OUTPUT_AUDIO_DIR = "audio_export"

# Export Metadati (Collection)
METADATA_COLLECTION_NAME = "audio_samples"
OUTPUT_JSON_FILE = "dataset.json"


# ==========================================
# 2. CLASSI DI UTILITÀ
# ==========================================
class MongoJSONEncoder(json.JSONEncoder):
    """
    Encoder personalizzato per convertire i tipi BSON di Mongo (ObjectId, datetime)
    in stringhe JSON standard compatibili con lo script di Colab.
    """

    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)  # Converte ObjectId("...") in stringa semplice
        if isinstance(o, datetime):
            return o.isoformat()  # Converte datetime in stringa ISO
        return json.JSONEncoder.default(self, o)


# ==========================================
# 3. FUNZIONI DI EXPORT
# ==========================================

async def export_audio_files():
    """Scarica i file audio da GridFS nella cartella locale."""
    print(f"\n--- Inizio Export Audio (Bucket: {FS_BUCKET_NAME}) ---")

    # Crea cartella output se non esiste
    if not os.path.exists(OUTPUT_AUDIO_DIR):
        os.makedirs(OUTPUT_AUDIO_DIR)
        print(f"Creata cartella: {OUTPUT_AUDIO_DIR}")

    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    fs = AsyncIOMotorGridFSBucket(db, bucket_name=FS_BUCKET_NAME)

    try:
        # Itera su tutti i file in GridFS
        cursor = fs.find({})
        count = 0
        new_downloads = 0

        async for grid_file in cursor:
            filename = grid_file.filename
            file_path = os.path.join(OUTPUT_AUDIO_DIR, filename)

            # Scarica solo se non esiste già per risparmiare tempo
            if not os.path.exists(file_path):
                print(f"Scaricamento: {filename}...")
                with open(file_path, "wb") as f:
                    await fs.download_to_stream(grid_file._id, f)
                new_downloads += 1

            count += 1

        print(f"✅ Audio Export completato.")
        print(f"   - Totale file nel DB: {count}")
        print(f"   - Nuovi file scaricati: {new_downloads}")
        print(f"   - Cartella destinazione: {OUTPUT_AUDIO_DIR}")

    except Exception as e:
        print(f"❌ Errore durante export audio: {e}")
    finally:
        client.close()


async def export_metadata_to_json():
    """Esporta i metadati della collezione in un file JSON."""
    print(f"\n--- Inizio Export Metadati (Collection: {METADATA_COLLECTION_NAME}) ---")

    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[METADATA_COLLECTION_NAME]

    try:
        # Recupera tutti i documenti
        # to_list(length=None) scarica tutto in memoria (ok per <100k items)
        documents = await collection.find({}).to_list(length=None)

        count = len(documents)
        print(f"Trovati {count} documenti. Scrittura su disco...")

        # Scrittura su file JSON
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            # indent=2 rende il file leggibile (pretty print)
            # cls=MongoJSONEncoder gestisce la conversione dei dati BSON
            json.dump(documents, f, cls=MongoJSONEncoder, indent=2)

        print(f"✅ Metadata Export completato.")
        print(f"   - File salvato come: {OUTPUT_JSON_FILE}")

    except Exception as e:
        print(f"❌ Errore durante export metadati: {e}")
    finally:
        client.close()


# ==========================================
# 4. MAIN ORCHESTRATOR
# ==========================================

async def main():
    print("🚀 AVVIO PROCEDURA DI ESPORTAZIONE DATASET PER COLAB 🚀")
    print(f"Database Target: {DB_NAME}")

    # Esegui le due operazioni sequenzialmente
    await export_audio_files()
    await export_metadata_to_json()

    print("\n" + "=" * 50)
    print("✨ OPERAZIONE COMPLETATA CON SUCCESSO! ✨")
    print("=" * 50)
    print("Prossimi step:")
    print(f"1. Crea un archivio ZIP contenente:")
    print(f"   - Il file '{OUTPUT_JSON_FILE}'")
    print(f"   - La cartella '{OUTPUT_AUDIO_DIR}'")
    print("2. Carica lo ZIP su Google Colab.")
    print("3. Esegui lo script 'main_processor_from_json' fornito in precedenza.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperazione interrotta dall'utente.")