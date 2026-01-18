import asyncio
import logging
import os
from rag.agents import AudioFinderAgent, SoundDesignerAgent, AudioAnalystAgent, HumanizerAgent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioAgentOrchestrator:
    def __init__(self):
        logger.info("Avvio Orchestrator...")
        self.finder = AudioFinderAgent()
        self.designer = SoundDesignerAgent()
        self.analyst = AudioAnalystAgent()
        self.writer = HumanizerAgent()
        logger.info("Sistema pronto.")

    async def process_request(self, user_input: str) -> str:
        intent = self._classify_intent(user_input)
        logger.info(f"Intent: {intent}")
        
        context_data = {"user_query": user_input, "intent": intent}

        if intent == "ANALYSIS":
            # 1. Estrazione del Path dal comando utente
            # Rimuove "Analizza il", "vedi", ecc.
            clean_input = user_input.lower()
            for kw in ["analizza", "analyze", "check", "controlla", "vedi", "il file", "l'audio", "il"]:
                clean_input = clean_input.replace(kw, "")
            
            # Il path è ciò che rimane (es. "C:\Users\file.mp3")
            target_path = clean_input.strip()
            
            # Esegue l'analisi sul file locale
            result = self.analyst.run(target_path)
            context_data["analysis_result"] = result

        else: 
            # RETRIEVAL_MIX (Ricerca Audio + Parametri)
            audio_results = self.finder.run(user_input)
            
            clean_query = user_input.lower()
            for kw in ["trova", "find", "cerca", "get", "un", "una", "dammi"]:
                clean_query = clean_query.replace(kw, "")
            params_result = self.designer.run(clean_query.strip())

            context_data["found_audio_samples"] = audio_results
            context_data["suggested_dsp_parameters"] = params_result

        # Generazione Risposta
        return self.writer.generate_response(
            user_query=user_input, 
            intent=intent, 
            data=context_data
        )

    def _classify_intent(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["analizza", "analyze", "check", "controlla", "descrivi"]):
            return "ANALYSIS"
        return "RETRIEVAL_MIX"

if __name__ == "__main__":
    orchestrator = AudioAgentOrchestrator()
    
    # Esempio pratico: Analisi file locale -> Ricerca Simili
    path_test = r"C:/Users/marti/Desktop/aggressive-high-pitched-dubstep-bass-loop.mp3"
    
    # Simula se il file esiste davvero, altrimenti test generico
    if os.path.exists(path_test):
        print(f"\n--- TEST: Analisi {os.path.basename(path_test)} ---")
        res = asyncio.run(orchestrator.process_request(f"Analizza il {path_test}"))
        print(res)
    else:
        print("\n[INFO] File di test non trovato sul disco, impossibile eseguire demo analisi.")