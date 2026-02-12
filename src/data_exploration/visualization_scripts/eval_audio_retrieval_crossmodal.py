import asyncio
import os
import sys
import tempfile
import pandas as pd
import numpy as np
from typing import List, Set
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from qdrant_client import QdrantClient
from scipy.spatial.distance import cosine
from bson import ObjectId

# Ensure project root is in sys.path
sys.path.append(os.path.abspath("."))
from config.settings import settings
from rag.clap.model_handler import create_clap_model

# --- CONFIGURATION ---
INDEX_COLLECTION_NAME = settings.QDRANT_ENRICHED_COLLECTION_NAME
TOP_K = 10
DB_NAME = "audio_db_test"
COLLECTION_NAME = "audio_samples"
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


def get_cosine_similarity(vec1, vec2):
    """Calcola la similarità coseno (1 - distanza coseno)."""
    # CLAP vectors are normalized, but scipy cosine distance is robust
    return 1 - cosine(vec1, vec2)


# --- MAIN EVALUATION ---
async def run_audio_retrieval_evaluation():
    print(f"--- 2.2 AUDIO-TO-AUDIO RETRIEVAL EVALUATION (With Cross-Modal Check) ---")
    print(f"Index: {INDEX_COLLECTION_NAME}")
    print(f"Test DB: {DB_NAME} (Held-out samples)")

    # 1. INIT CLIENTS
    client_q = QdrantClient(host=settings.QDRANT_CONNECTION_HOST, port=settings.QDRANT_PORT)
    mongo_client = AsyncIOMotorClient(MONGODB_URI)
    test_db = mongo_client[DB_NAME]
    test_collection = test_db[COLLECTION_NAME]
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

    # 3. EVALUATION LOOP
    for i, sample in enumerate(test_samples):
        # Metadata Setup
        sample_data = sample.get("sample", {})
        metadata = sample.get("metadata", {})
        file_name = sample_data.get("file_name", "unknown")

        # GridFS ID Handling
        raw_id = sample.get("gridfs_file_id")
        if not raw_id:
            print(f"⚠️ Skipping {file_name}: Missing gridfs_file_id")
            continue
        try:
            file_id = ObjectId(raw_id)
        except Exception:
            continue

        # Extract Ground Truth Metadata
        query_category = (metadata.get("main_category") or "").lower().strip()
        query_tags = normalize_tags(metadata.get("categories", []))
        if not query_tags:
            query_tags = normalize_tags(metadata.get("tags", []))

        # --- CROSS-MODAL PREPARATION ---
        # Costruiamo il "Prompt Ideale" basato sui metadati reali del file di test
        # Questo ci serve per validare se l'audio recuperato "matcha" la descrizione testuale
        gt_description = f"{query_category} {' '.join(query_tags)}".strip()
        gt_text_vec = None
        if gt_description:
            gt_text_vec = clap.get_text_embedding([gt_description])[0]
        # -------------------------------

        print(f"[{i + 1}/{len(test_samples)}] Processing: {file_name}")

        # A. Audio Embedding Generation
        tmp_path = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            await gridfs.download_to_stream(file_id, tmp)
            tmp.close()
            audio_vec = clap.get_audio_embedding([tmp_path])[0]
        except Exception as e:
            print(f"  ❌ Error processing audio: {e}")
            try:
                tmp.close()
            except:
                pass
            continue
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass

        # B. Retrieve Neighbors
        try:
            search_res = client_q.query_points(
                collection_name=INDEX_COLLECTION_NAME,
                query=audio_vec,
                using="audio_vector",
                limit=TOP_K,
                with_payload=True,
                with_vectors=["audio_vector"]  # NECESSARIO: Recuperiamo i vettori per il calcolo cross-modale
            )
        except Exception as e:
            print(f"  ⚠️ Qdrant Error: {e}")
            continue

        # C. Calculate Metrics
        cat_hits = 0
        jaccard_sum = 0.0
        similarity_sum = 0.0  # Audio-to-Audio (Circular but useful for precision)
        cross_modal_sum = 0.0  # Input Text -> Output Audio (Validation Metric)

        for hit in search_res.points:
            payload = hit.payload or {}

            # 1. Audio-to-Audio Similarity (Qdrant Score)
            similarity_sum += hit.score

            # 2. Cross-Modal Alignment (Validation)
            # Confrontiamo: Vettore Testo dei Metadati Reali <-> Vettore Audio del Risultato
            if gt_text_vec is not None:
                # Recupera il vettore audio del risultato (potrebbe essere in dict o list)
                res_audio_vec = hit.vector.get("audio_vector") if isinstance(hit.vector, dict) else hit.vector
                cm_score = get_cosine_similarity(gt_text_vec, res_audio_vec)
                cross_modal_sum += cm_score

            # 3. Category Hit
            res_category = (payload.get("main_category") or "").lower().strip()
            if query_category and res_category:
                if query_category == res_category or query_category in res_category:
                    cat_hits += 1

            # 4. Tag Overlap
            res_tags = normalize_tags(payload.get("ai_tags", []))
            if not res_tags: res_tags = normalize_tags(payload.get("original_tags", []))
            if not res_tags: res_tags = normalize_tags(payload.get("categories", []))

            jaccard_sum += calculate_jaccard_similarity(query_tags, res_tags)

        results_log.append({
            "Query File": file_name,
            "Category": query_category,
            "Hit Rate@K": cat_hits / TOP_K,
            "Tag Overlap@K": jaccard_sum / TOP_K,
            "Mean Audio Similarity": similarity_sum / TOP_K,
            "Cross-Modal Alignment": cross_modal_sum / TOP_K if gt_text_vec is not None else 0.0
        })

    # 4. REPORTING
    df = pd.DataFrame(results_log)
    if df.empty:
        print("No results generated.")
        return

    print("\n--- AGGREGATED RESULTS ---")
    print(f"Total Samples: {len(df)}")
    print(f"Category Hit Rate@{TOP_K}:    {df['Hit Rate@K'].mean():.4f}")
    print(f"Tag Overlap@{TOP_K}:         {df['Tag Overlap@K'].mean():.4f}")
    print(f"Mean Audio Similarity:      {df['Mean Audio Similarity'].mean():.4f} (Internal Consistency)")
    print(f"Cross-Modal Alignment:      {df['Cross-Modal Alignment'].mean():.4f} (Semantic Validation)")

    # Save
    df.to_csv("audio_retrieval_evaluation_crossmodal.csv", index=False)
    print("\nSaved to 'audio_retrieval_evaluation_crossmodal.csv'")


if __name__ == "__main__":
    asyncio.run(run_audio_retrieval_evaluation())