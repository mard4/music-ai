import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from openai import OpenAI
import sys
import os
sys.path.append(os.path.abspath("."))
from config.settings import settings

"""
Metriche di Retrieval (Text-to-Audio)
- Precision@K (Category Consistency):Misura la purezza dei risultati. Se cerchi "Dark Techno Bass", qual è la percentuale dei primi $K$ file restituiti che appartengono effettivamente alla categoria "Bass"? Indica l'affidabilità di base del sistema.
- Recall@K (Target Retrieval):Misura la capacità di recupero. In uno scenario controllato (dove sappiamo che esiste un file specifico che corrisponde alla query), il sistema riesce a includerlo nei primi $K$ risultati?
- MRR (Mean Reciprocal Rank):Misura la tempestività del risultato. Indica quanto in alto appare il primo risultato rilevante. Un valore di 1.0 significa che il file perfetto è sempre al primo posto; valori più bassi indicano che l'utente deve scorrere la lista.
- nDCG@K (Ranking Quality):Valuta la qualità dell'ordinamento con rilevanza graduata. Premia il sistema se posiziona i risultati "perfetti" (match di categoria + timbro) prima dei risultati "parziali" (solo categoria). È la metrica più raffinata per la user experience.
"""

OUTPUT_PATH = "results"
COLLECTION_NAME = settings.QDRANT_ENRICHED_COLLECTION_NAME
TOP_K = 10



def dcg_at_k(r, k, method=0):
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max: return 0.
    return dcg_at_k(r, k, method) / dcg_max


def precision_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size == 0: return 0.0
    relevant_count = np.sum(r >= 1)
    return relevant_count / k


def mrr_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    for i, score in enumerate(r):
        if score >= 1: return 1 / (i + 1)
    return 0.0


# --- RELEVANCE SCORING ---

def calculate_relevance(query_targets: Dict, result_payload: Dict) -> int:
    """
    Graded Relevance (0-2):
    2: Matches Category AND >= 1 Descriptor
    1: Matches Category OR >= 1 Descriptor
    0: No match
    """
    target_cat = query_targets.get("category", "").lower().strip()
    target_tags = set(t.lower().strip() for t in query_targets.get("descriptors", []))

    res_cat = (result_payload.get("main_category") or "").lower().strip()

    res_tags_list = result_payload.get("ai_tags", []) or []
    if not isinstance(res_tags_list, list): res_tags_list = []

    orig_tags = result_payload.get("original_tags", []) or []
    if isinstance(orig_tags, list):
        res_tags_list.extend(orig_tags)

    res_tags = set(str(t).lower().strip() for t in res_tags_list)

    cat_match = (target_cat == res_cat) and (target_cat != "")
    tag_matches = target_tags.intersection(res_tags)
    has_tag_match = len(tag_matches) > 0

    if cat_match and has_tag_match:
        return 2
    elif cat_match or has_tag_match:
        return 1
    else:
        return 0


# --- QUERY GENERATION ---

def get_type2_natural_queries() -> List[Dict]:
    return [
        {
            "query_text": "granular dark bass for trap drop",
            "targets": {"category": "bass", "descriptors": ["granular", "dark", "trap", "heavy"]},
            "source_filename": None  # Nessun file sorgente da escludere
        },
        {
            "query_text": "bright analog synth pad",
            "targets": {"category": "synth", "descriptors": ["bright", "analog", "pad", "warm"]},
            "source_filename": None
        },
        {
            "query_text": "punchy kick drum dry",
            "targets": {"category": "drums", "descriptors": ["punchy", "kick", "dry", "hard"]},
            "source_filename": None
        },
        {
            "query_text": "atmospheric cinematic texture",
            "targets": {"category": "fx", "descriptors": ["atmospheric", "cinematic", "texture", "ambient"]},
            "source_filename": None
        },
        {
            "query_text": "distorted 808 bass",
            "targets": {"category": "bass", "descriptors": ["distorted", "808", "sub"]},
            "source_filename": None
        }
    ]


def generate_type1_metadata_queries(sample_pool: List[Dict]) -> List[Dict]:
    queries = []
    for sample in sample_pool:
        cat = sample.get("main_category", "").lower()
        tags = sample.get("ai_tags", [])
        fname = sample.get("original_filename")  # <--- Salviamo il filename

        if not cat or not tags: continue

        # Scegliamo tag casuali per creare la query
        selected_tags = tags[:2]
        query_text = f"{' '.join(selected_tags)} {cat}"

        queries.append({
            "query_text": query_text,
            "targets": {"category": cat, "descriptors": selected_tags},
            "source_filename": fname
        })
    return queries


# --- MAIN EVALUATION FLOW ---

async def run_evaluation():
    print("--- 2.1 RETRIEVAL EVALUATION (STRICT MODE) ---")
    print("Note: Excluding the source file itself from search results to avoid data leakage.")

    client_q = QdrantClient(host=settings.QDRANT_CONNECTION_HOST, port=settings.QDRANT_PORT)
    client_o = OpenAI(api_key=settings.MODEL_API_KEY)

    # 1. SETUP TEST QUERIES
    print("Fetching samples for Type 1 Query Generation...")
    scroll_res, _ = client_q.scroll(collection_name=COLLECTION_NAME, limit=20, with_payload=True)
    type1_pool = [hit.payload for hit in scroll_res]

    type1_queries = generate_type1_metadata_queries(type1_pool)
    type2_queries = get_type2_natural_queries()

    all_queries = [("Type 1 (Metadata)", q) for q in type1_queries] + \
                  [("Type 2 (Natural)", q) for q in type2_queries]

    results_log = []

    # 2. EVALUATION LOOP
    for q_type, q_data in all_queries:
        query_text = q_data["query_text"]
        targets = q_data["targets"]
        source_filename = q_data.get("source_filename")  # File da ignorare

        # A. Embed Query
        emb_res = client_o.embeddings.create(input=query_text, model=settings.MODEL_EMBEDDING_MODEL)
        query_vec = emb_res.data[0].embedding

        # B. Retrieve (Ask for MORE than TOP_K to handle exclusion)
        search_res = client_q.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            using="text_vector",
            limit=TOP_K + 5,  # Chiediamo 5 extra buffer
            with_payload=True
        )

        # C. Filter & Score
        relevance_scores = []

        filtered_count = 0
        for hit in search_res.points:
            # STOP condition: abbiamo riempito la top-k
            if filtered_count >= TOP_K:
                break

            payload = hit.payload or {}
            hit_filename = payload.get('original_filename')

            # --- SELF-EXCLUSION ---
            # Se il risultato è lo stesso file che ha generato la query, SALTALO.
            if source_filename and hit_filename == source_filename:
                continue
                # ------------------------------------

            score = calculate_relevance(targets, payload)
            relevance_scores.append(score)
            filtered_count += 1

        # D. Compute Metrics
        ndcg = ndcg_at_k(relevance_scores, TOP_K)
        precision = precision_at_k(relevance_scores, TOP_K)
        mrr = mrr_at_k(relevance_scores, TOP_K)


        results_log.append({
            "Type": q_type,
            "Query": query_text,
            "nDCG": ndcg,
            "Precision": precision,
            "MRR": mrr,
            "Targets": str(targets)
        })

    # 3. REPORTING
    df = pd.DataFrame(results_log)

    if df.empty:
        print("Nessun dato generato.")
        return

    grouped = df.groupby("Type")[["nDCG", "Precision", "MRR"]].mean()
    print(grouped)

    print("\n--- DETAILED LOG ---")
    print(df[["Type", "Query", "nDCG", "Precision"]])


    df.to_csv(f"{OUTPUT_PATH}/retrieval_evaluation.csv", index=False)

    grouped.to_csv(f"{OUTPUT_PATH}/retrieval_evaluation_summary.csv", index=False)
    print("\nResults saved to 'retrieval_evaluation_strict.csv'")


if __name__ == "__main__":
    asyncio.run(run_evaluation())