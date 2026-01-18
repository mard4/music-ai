import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.orchestrator import AudioAgentOrchestrator

if __name__ == "__main__":
    orchestrator = AudioAgentOrchestrator()

    path_test = r"C:/Users/marti/Desktop/aggressive-high-pitched-dubstep-bass-loop.mp3"

    if os.path.exists(path_test):
        print(f"\n--- TEST: Analisi {os.path.basename(path_test)} ---")
        res = asyncio.run(orchestrator.process_request(f"Analizza il {path_test}"))
        print(res)
    else:
        print("\n[INFO] File di test non trovato sul disco, impossibile eseguire demo analisi.")