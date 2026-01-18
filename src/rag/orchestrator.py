import asyncio
import logging
import os
# Importiamo anche IntentClassifierAgent
from rag.agents import AudioFinderAgent, SoundDesignerAgent, AudioAnalystAgent, HumanizerAgent, IntentClassifierAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioAgentOrchestrator:
    def __init__(self):
        logger.info("Avvio Orchestrator...")
        self.classifier = IntentClassifierAgent()
        self.finder = AudioFinderAgent()
        self.designer = SoundDesignerAgent()
        self.analyst = AudioAnalystAgent()
        self.writer = HumanizerAgent()
        logger.info("Sistema pronto.")

    async def process_request(self, user_input: str) -> str:

        classification = self.classifier.run(user_input)
        intent = classification["intent"]
        clean_params = classification["params"]

        logger.info(f"Intent: {intent} | Clean Params: '{clean_params}'")

        context_data = {"user_query": user_input, "intent": intent}

        if intent == "ANALYSIS":
            # Esegue l'analisi sul path pulito (es. "C:\files\loop.mp3")
            result = self.analyst.run(clean_params)
            context_data["analysis_result"] = result

        else:
            # RETRIEVAL_MIX (Ricerca Audio + Parametri)
            # Usiamo la query pulita per entrambi gli agenti per migliorare la precisione
            # (es. rimuove "trova un" lasciando solo "kick drum")

            # 1. Finder Agent
            audio_results = self.finder.run(clean_params)

            # 2. Sound Designer Agent
            params_result = self.designer.run(clean_params)

            context_data["found_audio_samples"] = audio_results
            context_data["suggested_dsp_parameters"] = params_result

        # Generazione Risposta
        return self.writer.generate_response(
            user_query=user_input,
            intent=intent,
            data=context_data
        )

    def _classify_intent(self, query: str) -> str:
        # Deprecato: Logica spostata in IntentClassifierAgent
        pass


if __name__ == "__main__":
    orchestrator = AudioAgentOrchestrator()

    path_test = r"C:/Users/marti/Desktop/aggressive-high-pitched-dubstep-bass-loop.mp3"

    if os.path.exists(path_test):
        print(f"\n--- TEST: Analisi {os.path.basename(path_test)} ---")
        res = asyncio.run(orchestrator.process_request(f"Analizza il {path_test}"))
        print(res)
    else:
        print("\n[INFO] File di test non trovato sul disco.")