from typing import List, Dict, Any, Optional
from rag.agents.AgentBase import AgentBase
from rag.tools.retrieval import RetrievalTool
from rag.utils import logger


class AudioRetriever(AgentBase):
    """
    AGENT: Audio Retriever
    Ruolo: Riceve la richiesta (testo o vettore) e usa il Tool per cercare nel DB.
    Ora supporta la ricerca ibrida (Text-to-Audio e Audio-to-Audio).
    """

    def __init__(self):
        super().__init__(
            prompt_file="humanizer_retrieval.txt",
            agent_name="RetrievalAgent"
        )
        self.tool = RetrievalTool()

    async def retrieve(self, query: str = None, audio_vector: List[float] = None) -> List[Dict[str, Any]]:
        """
        Router di ricerca unificato.
        - Se c'è un audio_vector -> Cerca per similarità acustica (collection: audio_vector).
        - Se c'è una query testo -> Cerca per similarità semantica (collection: text_vector).
        """

        if audio_vector:
            logger.info("AudioRetriever: Esecuzione ricerca per VETTORE AUDIO (CLAP)")
            return await self.tool.search_similar_audio_audio_vector(vector=audio_vector)

        elif query:
            logger.info(f"AudioRetriever: Esecuzione ricerca per TESTO '{query}'")
            return await self.tool.search_similar_audio_text_vector(query_text=query)

        logger.warning("AudioRetriever: Nessun input valido fornito (né query né vector).")
        return []

    def format_html_response(self, samples: List[Dict[str, Any]]) -> str:
        """
        Genera l'HTML per la lista sample.
        """
        if not samples:
            return ""
        try:
            return self.render_prompt(samples=samples)
        except Exception as e:
            self.logger.error(f"Errore rendering HTML samples: {e}")
            return ""