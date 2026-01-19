import logging
from typing import List, Dict, Any
from rag.agents.AgentBase import AgentBase
from rag.tools.audio_analysis import LabelEnricherTool

logger = logging.getLogger(__name__)


class LabelEnricher(AgentBase):
    """
    AGENT: Label Enricher / Synthesizer.
    Responsabilità: Coordinare il tool di analisi audio per generare descrizioni.
    """

    def __init__(self):
        super().__init__(context={"agent_name": "LabelEnricher"})

        self.tool = LabelEnricherTool()

    def run(self, filename: str, audio_path: str, neighbors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """        Esegue sintesi e verifica.
        Requires: audio_path (per il check CLAP).
        """
        return self.tool.predict_label_from_neighbors(
            filename=filename,
            audio_path=audio_path,
            neighbors=neighbors
        )