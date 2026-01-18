import asyncio
from rag.orchestrator import AudioAgentOrchestrator
import os


if __name__ == "__main__":
    orchestrator = AudioAgentOrchestrator()

    path_test = r"C:/Users/marti/Desktop/aggressive-high-pitched-dubstep-bass-loop.mp3"

    # Simula se il file esiste davvero, altrimenti test generico
    if os.path.exists(path_test):
        print(f"\n--- TEST: Analisi {os.path.basename(path_test)} ---")
        res = asyncio.run(orchestrator.process_request(f"Analizza il {path_test}"))
        print(res)
    else:
        print("\n[INFO] File di test non trovato sul disco, impossibile eseguire demo analisi.")