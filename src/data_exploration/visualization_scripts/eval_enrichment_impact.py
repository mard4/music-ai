import asyncio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict
from qdrant_client import QdrantClient
from openai import OpenAI
from scipy.spatial.distance import cosine
import sys
import os
sys.path.append(os.path.abspath("."))
from config.settings import settings

"""
3. Metriche di Validazione Semantica (Enrichment Impact)
Cross-Modal Alignment Score:
Una metrica di validazione incrociata. Misura la similarità tra il vettore Audio del risultato trovato e il vettore Testo della descrizione ideale (Ground Truth) del file cercato. Serve a provare che ciò che è simile acusticamente è anche coerente semanticamente.

Mean Semantic Gain (MSG):
Quantifica il valore aggiunto dell'AI. È la differenza tra la similarità della query con la ai_label (Enriched) rispetto alla similarità con l'original_filename (Raw). Un valore positivo dimostra matematicamente che l'arricchimento ha reso il file "più trovabile".


"""


COLLECTION_NAME = settings.QDRANT_ENRICHED_COLLECTION_NAME
TOP_K = 5

TEST_QUERIES = [
    "Dark cinematic drone with granular texture",
    "Punchy analog kick drum for techno",
    "Ethereal pad with shimmering highs",
    "Distorted 808 bass with long decay",
    "Dry acoustic snare drum jazz style"
]


def cosine_similarity(v1, v2):
    return 1 - cosine(v1, v2)


async def evaluate_enrichment_impact():
    print(f"--- 3. ENRICHMENT IMPACT ANALYSIS (Before vs After) ---")

    client_q = QdrantClient(host=settings.QDRANT_CONNECTION_HOST, port=settings.QDRANT_PORT)
    client_o = OpenAI(api_key=settings.MODEL_API_KEY)

    results_data = []

    print(f"Analyzing {len(TEST_QUERIES)} complex queries...")

    for query_text in TEST_QUERIES:
        print(f"\nQuery: '{query_text}'")

        # 1. Embed Query
        res_q = client_o.embeddings.create(input=query_text, model=settings.MODEL_EMBEDDING_MODEL)
        query_vec = res_q.data[0].embedding

        # 2. Retrieve results (using the ENRICHED vector, which is the current system)
        search_res = client_q.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            using="text_vector",
            limit=TOP_K,
            with_payload=True,
            with_vectors=["text_vector"]  # Recuperiamo il vettore enriched già calcolato
        )

        for hit in search_res.points:
            payload = hit.payload or {}

            # DATI "AFTER" (Enriched)
            ai_label = payload.get("ai_label", "N/A")
            vec_after = hit.vector.get("text_vector") if isinstance(hit.vector, dict) else hit.vector

            # DATI "BEFORE" (Original)
            # Dobbiamo gestire casi in cui il filename è sporco o vuoto
            orig_filename = payload.get("original_filename", "unknown.wav")
            orig_label = payload.get("original_label") or orig_filename

            # Calcoliamo l'embedding del "BEFORE" al volo per il confronto
            # (Simuliamo: "E se avessimo cercato solo nel filename?")
            res_orig = client_o.embeddings.create(input=orig_label, model=settings.MODEL_EMBEDDING_MODEL)
            vec_before = res_orig.data[0].embedding

            # 3. Calcolo Similarità
            sim_before = cosine_similarity(query_vec, vec_before)  # Query vs Filename
            sim_after = cosine_similarity(query_vec, vec_after)  # Query vs AI Description

            gain = sim_after - sim_before

            results_data.append({
                "Query": query_text,
                "Filename (Before)": orig_filename,
                "AI Label (After)": ai_label,
                "Sim_Before": sim_before,
                "Sim_After": sim_after,
                "Gain": gain
            })

            print(f"  -> Match: {orig_filename}")
            print(f"     [Before: {sim_before:.3f}] -> [After: {sim_after:.3f}] (Gain: +{gain:.3f})")

    # --- REPORTING & VISUALIZATION ---
    df = pd.DataFrame(results_data)

    if df.empty:
        print("No data found.")
        return

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Average Similarity BEFORE (Raw Data): {df['Sim_Before'].mean():.4f}")
    print(f"Average Similarity AFTER (Enriched):  {df['Sim_After'].mean():.4f}")
    print(f"AVERAGE SEMANTIC GAIN:               +{df['Gain'].mean():.4f}")

    # Save CSV
    df.to_csv("enrichment_impact_analysis.csv", index=False)
    print("Saved to 'enrichment_impact_analysis.csv'")

    # --- PLOT: SLOPE CHART (Before vs After) ---
    print("\nGenerating Slope Chart...")

    fig = go.Figure()

    # Per non affollare il grafico, prendiamo la media per ogni Query o i top 10 campioni totali
    # Qui plottiamo i singoli punti per vedere le linee

    subset = df.head(15)  # Primi 15 esempi per chiarezza visiva

    for i, row in subset.iterrows():
        color = 'green' if row['Gain'] > 0 else 'red'

        # Linea che collega Before e After
        fig.add_trace(go.Scatter(
            x=['Original Metadata', 'AI Enriched'],
            y=[row['Sim_Before'], row['Sim_After']],
            mode='lines+markers',
            line=dict(color='gray', width=1),
            marker=dict(size=8, color=[color, color]),
            showlegend=False,
            hoverinfo='text',
            text=f"File: {row['Filename (Before)']}<br>Query: {row['Query']}<br>Gain: {row['Gain']:.3f}"
        ))

    # Aggiungi le medie come linee spesse
    avg_before = df['Sim_Before'].mean()
    avg_after = df['Sim_After'].mean()

    fig.add_trace(go.Scatter(
        x=['Original Metadata', 'AI Enriched'],
        y=[avg_before, avg_after],
        mode='lines+markers',
        name='AVERAGE IMPROVEMENT',
        line=dict(color='blue', width=4),
        marker=dict(size=12, color='blue')
    ))

    fig.update_layout(
        title="<b>Semantic Enrichment Impact</b><br>Similarity to User Query (Raw Filename vs AI Label)",
        yaxis_title="Cosine Similarity with Query",
        template="plotly_white",
        width=800, height=600
    )

    fig.show()


if __name__ == "__main__":
    asyncio.run(evaluate_enrichment_impact())