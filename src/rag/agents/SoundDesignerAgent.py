from typing import Dict, Any, Optional
from rag.agents.AgentBase import AgentBase
from rag.tools.dsp_search import DSPSearchTool


class SoundDesignerAgent(AgentBase):
    """
    AGENT: Sound Designer ("L'Ingegnere").
    Ruolo: Tradurre aggettivi (es. "Crunchy") in parametri tecnici (es. Drive=50%).
    """

    def __init__(self):
        super().__init__(agent_name="SoundDesigner")

        # Inizializza il tool specifico
        # (Possiamo passare i client se l'AgentBase li esponesse, altrimenti il tool se li crea)
        self.dsp_tool = DSPSearchTool()

    async def run(self, query: str) -> Dict[str, Any]:
        """Esegue la ricerca dei parametri."""
        self.logger.info(f"Sound Designer: Ricerca ricetta tecnica per '{query}'...")

        if "," in query:
            keywords = [k.strip() for k in query.split(",") if k.strip()]
            if len(keywords) > 1:
                # Usa blend_parameters se hai più keyword
                dsp_result = await self.dsp_tool.blend_parameters(keywords)
            else:
                dsp_result = await self.dsp_tool.find_parameters(keywords[0])
        else:
            dsp_result = await self.dsp_tool.find_parameters(query)

        if not dsp_result:
            # Ritorna un dict vuoto ma "safe" per l'Humanizer
            return {"found": False, "message": f"Nessun parametro trovato per '{query}'"}

        return {
            "found": True,
            "source": "SocialFX_DB",
            "concept": dsp_result.get("descriptor"),
            "effect": dsp_result.get("effect_type"),
            "values": dsp_result.get("params"),
            "keys": dsp_result.get("keys"),  # Importante passarlo all'Humanizer!
            "explanation": f"Parametri per {dsp_result.get('descriptor')}"
        }