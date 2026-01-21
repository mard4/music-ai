import os
import logging
from typing import Dict, Any, List, Optional
from rag.agents.AgentBase import AgentBase
from rag.clap.model_handler import create_clap_model

logger = logging.getLogger(__name__)


class AudioAnalystAgent(AgentBase):
    """
    AGENT: Audio Analyst
    Ruolo: Gestione Input Audio (Specialist).

    Nel nuovo workflow, questo agente NON fa più retrieval o sintesi.
    Si limita a:
    1. Validare l'esistenza del file.
    2. Usare il modello CLAP per generare l'embedding (il "DNA" del suono).
    """

    def __init__(self):
        super().__init__(context={"agent_name": "AudioAnalyst"})
        logger.info("Caricamento CLAP Model...")
        self.ear = create_clap_model()

    def get_embedding(self, file_path: str) -> Optional[List[float]]:
        """
        Metodo pulito per ottenere il vettore audio da un file.
        Usato dal Workflow nella FASE 1 (Retrieval).
        """
        clean_path = file_path.strip('"').strip("'")

        if not os.path.exists(clean_path):
            logger.error(f"File non trovato: {clean_path}")
            return None

        try:
            embedding_np = self.ear.get_audio_embedding([clean_path])[0]
            return embedding_np.tolist()

        except Exception as e:
            logger.error(f"Errore critico generazione embedding CLAP: {e}")
            return None