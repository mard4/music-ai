import os
from typing import Dict, Any
from rag.agents.AgentBase import AgentBase
from rag.tools.audio_analysis import LabelEnricherTool
from rag.tools.retrieval import RetrievalTool
from rag.utils import logger


class AudioAnalystAgent(AgentBase):
    """
    AGENT: Audio Analyst ("Il Medico").
    Ruolo: Analizza file locali e trova corrispondenze.
    """

    def __init__(self):
        super().__init__(agent_name="AudioAnalyst")

        # 2. Carica i Tools specifici
        self.analyser = LabelEnricherTool()
        self.retriever = RetrievalTool()

    def run(self, file_path: str) -> Dict[str, Any]:
        clean_path = file_path.strip('"').strip("'")

        if not os.path.exists(clean_path):
            return {"error": f"File non trovato: {clean_path}"}

        filename = os.path.basename(clean_path)
        self.logger.info(f"Analyst: Inizio diagnosi per {filename}")

        try:
            # FASE 1: Diagnosi (Analisi del file)
            # L'agente usa il tool per capire cos'è il file
            analysis = self.analyser.enrich_and_verify(filename, clean_path)

            clean_caption = analysis["caption"]
            self.logger.info(f"Analyst: File identificato come '{clean_caption}'")

            # FASE 2: Consultazione (Ricerca simili)
            # L'agente usa il tool di ricerca con la diagnosi appena fatta
            similar_samples = self.retriever.search_similar_audio(clean_caption, limit=3)

            return {
                "source": "LOCAL_FILE",
                "filename": filename,
                "analysis": {
                    "description": clean_caption,
                    "confidence": analysis["status"],
                    "tags": analysis["raw_tags"]
                },
                "recommendations": similar_samples
            }

        except Exception as e:
            self.logger.error(f"Analyst Error: {e}", exc_info=True)
            return {"error": "Analisi fallita"}