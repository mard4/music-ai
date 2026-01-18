from typing import List, Dict, Any
from rag.agents.AgentBase import AgentBase
from rag.tools.retrieval import RetrievalTool


class AudioRetriever(AgentBase):
    """
    AGENT: Audio Retriever ("Il Bibliotecario").
    Ruolo: Riceve la richiesta di ricerca dal Workflow e usa il Tool per eseguirla.
    """

    def __init__(self):
        super().__init__(agent_name="AudioRetriever")
        # Inizializza il Tool (il "Sistema di Ricerca")
        self.tool = RetrievalTool()

    async def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Interfaccia pubblica chiamata dal Workflow.
        """
        self.logger.info(f"AudioRetriever: Ricevuta richiesta per '{query}'")

        # Delega il lavoro sporco al Tool
        # Nota: usiamo await perché il tool ora è asincrono
        results = await self.tool.search_similar_audio(query, limit=k)

        return results