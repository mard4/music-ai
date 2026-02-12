import asyncio
import os
import sys
import tempfile
import pandas as pd
from typing import List, Set
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from qdrant_client import QdrantClient

# Ensure project root is in sys.path
sys.path.append(os.path.abspath("."))
from config.settings import settings
from rag.clap.model_handler import create_clap_model

# --- CONFIGURATION ---
INDEX_COLLECTION_NAME = settings.QDRANT_ENRICHED_COLLECTION_NAME
TOP_K = 10
DB_NAME = "audio_db_test"
COLLECTION_NAME = "audio_samples"
# Se usi una config specifica per mongo, usala qui, altrimenti default locale
MONGODB_URI = "mongodb://localhost:27017/"


# --- METRIC UTILS ---
def calculate_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


def normalize_tags(tag_list: List) -> Set[str]:
    if not isinstance(tag_list, list):
        return set()
    return set(str(t).lower().strip() for t in tag_list if t)


# --- MAIN EVALUATION ---
async def run_audio_retrieval_evaluation():
    print(f"--- 2.2 AUDIO-TO-AUDIO RETRIEVAL EVALUATION ---")
    print(f"Index: {INDEX_COLLECTION_NAME}")
    print(f"Test DB: {DB_NAME} (Held-out samples)")

    # 1. INIT CLIENTS
    client_q = QdrantClient(host=settings.QDRANT_CONNECTION_HOST, port=settings.QDRANT_PORT)

    mongo_client = AsyncIOMotorClient(MONGODB_URI)
    test_db = mongo_client[DB_NAME]
    test_collection = test_db[COLLECTION_NAME]

    # GridFS Asincrono (Motor)
    gridfs = AsyncIOMotorGridFSBucket(test_db, bucket_name="audio_files")

    print("Loading CLAP Model...")
    clap = create_clap_model()

    # 2. FETCH TEST SAMPLES
    cursor = test_collection.find({})
    test_samples = await cursor.to_list(length=None)

    if not test_samples:
        print("❌ No samples found in the Test Database.")
        return

    print(f"Found {len(test_samples)} held-out test samples.")

    results_log = []
    from bson import ObjectId
    # 3. EVALUATION LOOP
    for i, sample in enumerate(test_samples):
        # --- FIX 1: ESTRAZIONE CORRETTA METADATI ---
        # Adattato alla struttura del tuo DB (vedi ingestion)
        sample_data = sample.get("sample", {})
        metadata = sample.get("metadata", {})

        file_name = sample_data.get("file_name", "unknown")
        file_id = sample.get("file_name")

        raw_id = sample.get("gridfs_file_id")

        if not raw_id:
            print(f"⚠️ Skipping {file_name}: Missing gridfs_file_id")
            continue

        try:
            # Forziamo la conversione in ObjectId.
            # Se è già ObjectId non fa danni, se è stringa lo corregge.
            file_id = ObjectId(raw_id)
        except Exception as e:
            print(f"⚠️ Invalid ID format for {file_name}: {raw_id}")
            continue

        query_category = (metadata.get("main_category") or "").lower().strip()
        query_tags = normalize_tags(metadata.get("categories", []))
        if not query_tags:
            query_tags = normalize_tags(metadata.get("tags", []))

        print(f"[{i + 1}/{len(test_samples)}] Processing: {file_name} ({query_category})")

        # --- FIX 2: GESTIONE FILE WINDOWS SAFE ---
        tmp_path = None
        try:
            # Creiamo il file ma NON lo apriamo con 'with' per evitare lock persistenti
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name

            # Scarichiamo stream nel file aperto
            # Motor richiede un oggetto file-like scrivibile
            await gridfs.download_to_stream(file_id, tmp)

            # CHIUDIAMO SUBITO il file affinché CLAP possa riaprirlo
            tmp.close()

            # Generazione Embedding (CLAP legge dal path su disco)
            audio_vec = clap.get_audio_embedding([tmp_path])[0]

        except Exception as e:
            print(f"  ❌ Error processing audio {file_name}: {e}")
            # Se fallisce qui, assicuriamoci di chiudere se era aperto
            try:
                tmp.close()
            except:
                pass
            continue

        finally:
            # Pulizia sicura
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass  # Ignora errori di cancellazione su Windows se capita

        # C. Retrieve Neighbors
        try:
            search_res = client_q.query_points(
                collection_name=INDEX_COLLECTION_NAME,
                query=audio_vec,
                using="audio_vector",
                limit=TOP_K,
                with_payload=True
            )
        except Exception as e:
            print(f"  ⚠️ Qdrant Error: {e}")
            continue

        # D. Calculate Metrics
        cat_hits = 0
        jaccard_sum = 0.0
        similarity_sum = 0.0

        for hit in search_res.points:
            payload = hit.payload or {}

            similarity_sum += hit.score

            res_category = (payload.get("main_category") or "").lower().strip()

            # Flexible Category Match
            if query_category and res_category:
                if query_category == res_category or query_category in res_category:
                    cat_hits += 1

            res_tags = normalize_tags(payload.get("ai_tags", []))
            if not res_tags:
                res_tags = normalize_tags(payload.get("original_tags", []))

            jaccard = calculate_jaccard_similarity(query_tags, res_tags)
            jaccard_sum += jaccard

        results_log.append({
            "Query File": file_name,
            "Category": query_category,
            "Hit Rate@K": cat_hits / TOP_K,
            "Tag Overlap@K": jaccard_sum / TOP_K,
            "Mean Similarity": similarity_sum / TOP_K
        })

    # 4. REPORTING
    df = pd.DataFrame(results_log)

    if df.empty:
        print("No results generated.")
        return

    print("\n--- 2.2 AUDIO RETRIEVAL RESULTS (Aggregated) ---")
    print(f"Total Samples Tested: {len(df)}")

    overall_hit_rate = df["Hit Rate@K"].mean()
    overall_jaccard = df["Tag Overlap@K"].mean()
    overall_similarity = df["Mean Similarity"].mean()

    print(f"Category Hit Rate@{TOP_K}:  {overall_hit_rate:.4f}")
    print(f"Tag Overlap (Jaccard)@{TOP_K}: {overall_jaccard:.4f}")
    print(f"Mean Acoustic Similarity: {overall_similarity:.4f}")

    if "Category" in df.columns:
        print("\n--- BREAKDOWN BY CATEGORY ---")
        print(df.groupby("Category")[["Hit Rate@K", "Tag Overlap@K", "Mean Similarity"]].mean())

    df.to_csv("audio_retrieval_evaluation.csv", index=False)
    print("\nDetailed results saved to 'audio_retrieval_evaluation.csv'")


if __name__ == "__main__":
    asyncio.run(run_audio_retrieval_evaluation())