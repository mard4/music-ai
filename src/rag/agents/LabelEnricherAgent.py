import logging
from typing import List, Dict, Any
from rag.agents.AgentBase import AgentBase
from rag.tools.audio_analysis import LabelEnricherTool

logger = logging.getLogger(__name__)


class LabelEnricher(AgentBase):
    """
    AGENT: Label Enricher (Semantic Brain).

    Nel nuovo workflow, viene chiamato dopo il Retrieval.
    Il suo compito è guardare i 'neighbors' trovati e dire:
    "Visto che assomigli a questi 5 file, allora sei probabilmente un Acid Bass".
    """

    def __init__(self):
        super().__init__(context={"agent_name": "LabelEnricher"})
        self.tool = LabelEnricherTool()

    def run(self, filename: str, audio_path: str, neighbors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Esegue la 'Sintesi dei Vicini' (Analysis Synthesis).

        Args:
            filename: Nome del file target (o query utente).
            audio_path: Path del file (opzionale, serve solo per check allucinazioni).
            neighbors: La lista dei file simili trovati dal RetrievalAgent.
        """
        # Questo metodo usa il prompt 'analysis_synthesis.txt' internamente al tool
        return self.tool.predict_label_from_neighbors(
            filename=filename,
            audio_path=audio_path,
            neighbors=neighbors
        )