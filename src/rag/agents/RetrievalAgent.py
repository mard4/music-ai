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
        self.tool = RetrievalTool()

    async def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Ricerca di audio simili a partire dal label
        """
        self.logger.info(f"AudioRetriever: Ricevuta richiesta per '{query}'")

        results = await self.tool.search_similar_audio_text_vector(query)

        return results

