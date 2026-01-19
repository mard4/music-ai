import os
import logging
from typing import Dict, Any

from rag.agents.AgentBase import AgentBase
from rag.clap.model_handler import create_clap_model
from rag.tools.retrieval import RetrievalTool
from rag.agents.LabelEnricherAgent import LabelEnricher

logger = logging.getLogger(__name__)


class AudioAnalystAgent(AgentBase):
    def __init__(self):
        super().__init__(context={"agent_name": "AudioAnalyst"})
        logger.info("Caricamento CLAP...")
        self.ear = create_clap_model()
        self.retrieval_tool = RetrievalTool()
        self.synthesizer = LabelEnricher()

    async def run(self, file_path: str) -> Dict[str, Any]:
        clean_path = file_path.strip('"').strip("'")
        if not os.path.exists(clean_path):
            return {"error": f"File non trovato: {clean_path}"}

        filename = os.path.basename(clean_path)

        try:
            # 1. Genera vettore CLAP
            audio_vector = self.ear.get_audio_embedding([clean_path])[0].tolist()

            # 2. Usa il tool corretto
            similar_samples = await self.retrieval_tool.search_similar_audio_audio_vector(
                vector=audio_vector,
            )

            if not similar_samples:
                return {"error": "Nessun vicino trovato."}

            # 3. Sintesi
            synthesis_result = self.synthesizer.run(
                filename=filename,
                audio_path=clean_path,
                neighbors=similar_samples
            )

            return {
                "source": "LOCAL_FILE_ANALYSIS",
                "target_file": filename,
                "generated_analysis": synthesis_result,
                "evidence": similar_samples
            }

        except Exception as e:
            logger.error(f"Errore Analyst: {e}")
            return {"error": str(e)}