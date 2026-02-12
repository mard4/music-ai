import asyncio
import pandas as pd
from qdrant_client import QdrantClient
import sys
import os

# Setup Path
sys.path.append(os.path.abspath("."))
from config.settings import settings
from rag.clap.model_handler import create_clap_model

# --- CONFIGURAZIONE DEL TEST ---
# 1. Scegli un file audio REALE che hai sul disco (es. un Kick Drum)
# Se non ne hai uno, scaricane uno al volo o usa un path assoluto
QUERY_AUDIO_PATH = "C:/Users/marti/Desktop/dark-distorted-hard-trap-bass_F_minor.wav"

REAL_AUDIO_PATH = "C:/Users/marti/Desktop/dark-distorted-hard-trap-bass_F_minor.wav"

# 2. Dagli un nome FALSO e ingannevole
FAKE_FILENAME = "Romantic_Violin_Concerto_in_D_Major.wav"
FAKE_LABEL = "Violin Solo Classical"


# --- ESECUZIONE ---
async def run_misleading_test():
    print(f"--- 🚫 MISLEADING METADATA TEST ---")
    print(f"Subject: Audio Analysis Capabilities")
    print(f"Real Audio Content: KICK DRUM (Perceptual Reality)")
    print(f"Fake Metadata Label: '{FAKE_LABEL}' (Textual Deception)")
    print("-" * 60)

    if not os.path.exists(REAL_AUDIO_PATH):
        print(f"❌ ERRORE: Non trovo il file audio '{REAL_AUDIO_PATH}'")
        print("   Per favore modifica lo script inserendo il percorso di un file .wav o .mp3 esistente.")
        return

    # 1. Init
    client_q = QdrantClient(host=settings.QDRANT_CONNECTION_HOST, port=settings.QDRANT_PORT)
    clap = create_clap_model()

    # 2. "ASCOLTO" (Generazione Embedding Audio)
    print("🎧 System is listening to the input file...")
    # Il modello CLAP guarda SOLO l'audio, ignora il nome del file
    audio_vec = clap.get_audio_embedding([REAL_AUDIO_PATH])[0]

    # 3. RICERCA (Retrieval)
    print("🔍 Searching database based on acoustic features...")
    search_res = client_q.query_points(
        collection_name=settings.QDRANT_ENRICHED_COLLECTION_NAME,
        query=audio_vec,
        using="audio_vector",  # Usiamo il vettore audio
        limit=5,
        with_payload=True
    )

    # 4. RISULTATI
    results_data = []
    for rank, hit in enumerate(search_res.points, start=1):
        payload = hit.payload or {}
        filename = payload.get("original_filename", "Unknown")
        ai_desc = payload.get("ai_label", "N/A")
        category = payload.get("main_category", "N/A")

        results_data.append({
            "Rank": rank,
            "Retrieved File": filename,
            "Category": category,
            "AI Description": ai_desc,
            "Similarity": f"{hit.score:.4f}"
        })

    # 5. VISUALIZZAZIONE DELLA PROVA
    df = pd.DataFrame(results_data)

    print("\n" + "=" * 80)
    print(f"TEST RESULTS: Input labelled as '{FAKE_LABEL}'")
    print("=" * 80)

    # Mostriamo la tabella
    pd.set_option('display.max_colwidth', None)
    display(df) if 'display' in locals() else print(df.to_string())

    print("\n" + "=" * 80)
    # VERDETTO FINALE AUTOMATICO
    # Contiamo quanti risultati sono "Violin" vs quanti sono "Drums/Kick/Bass"
    violins_found = df['Category'].str.contains('violin', case=False).sum()

    if violins_found == 0:
        print("✅ SUCCESS: The system IGNORED the fake metadata.")
        print("   It retrieved acoustically similar files (e.g., Drums/Bass) based on the audio content.")
        print("   -> PROOF: The system 'listens' and does not just read.")
    else:
        print("❌ FAILURE: The system was fooled by the metadata.")


if __name__ == "__main__":
    asyncio.run(run_misleading_test())