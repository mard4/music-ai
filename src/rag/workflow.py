import asyncio
from rag.agents.IntentClassifierAgent import IntentClassifierAgent
from rag.agents.AudioAnalystAgent import AudioAnalystAgent
from rag.agents.HumanizerAgent import HumanizerAgent
from rag.agents.LabelEnricherAgent import LabelEnricher
from rag.agents.RetrievalAgent import AudioRetriever
from rag.agents.SoundDesignerAgent import SoundDesignerAgent

from rag.utils import logger

class Workflow:
    def __init__(self):
        logger.info("Inizializzazione Workflow AI...")

        # 1. CERVELLO: Capisce l'intenzione
        self.classifier = IntentClassifierAgent()

        # 2. RAMO ANALISI: Gestisce file
        self.analyst = AudioAnalystAgent()

        self.label_enricher = LabelEnricher()

        # 3. RAMO RICERCA (Audio): Trova sample nel DB
        self.audio_retriever = AudioRetriever()

        # 4. RAMO RICERCA (Parametri): Trova impostazioni DSP
        self.sound_designer = SoundDesignerAgent()

        # 5. VOCE: Genera la risposta finale
        self.humanizer = HumanizerAgent()

        logger.info("Workflow pronto.")

    async def run(self, user_input: str) -> str:
        """
        Esegue il flusso principale gestendo il routing della richiesta.
        """
        # STEP 1: Classificazione
        classification = self.classifier.run(user_input)
        intent = classification.get("intent")
        clean_params = classification.get("params")

        logger.info(f"Routing attivo: {intent} | Target: {clean_params}")

        context_data = {
            "user_query": user_input,
            "intent": intent,
            "results": {}
        }

        # STEP 2: Branching (Il Bivio)
        if intent == "ANALYSIS":
            await self.intent_analysis(clean_params, context_data)

        elif intent == "RETRIEVAL":
            await self.intent_retrieval(clean_params, context_data)

        else:
            context_data["results"] = {"error": "Intent not recognized."}

        # STEP 3: Generazione Risposta
        final_response = self.humanizer.generate_response(
            user_query=user_input,
            intent=intent,
            data=context_data
        )

        return final_response

    async def intent_analysis(self, clean_params, context_data):
        """PERCORSO A: Analisi File Locale
        L'Analyst si occupa di tutto: arricchimento e ricerca similare
        clean_params qui è il path del file
        """
        analysis_result = await self.analyst.run(clean_params)
        context_data["results"] = analysis_result

    async def intent_retrieval(self,clean_params, context_data):
        """"PERCORSO B: Ricerca nel Database ---
        # Eseguiamo in parallelo o sequenziale la ricerca di audio e parametri
        # clean_params qui sono le keyword (es. "kick drum distorto")
        """

        # A. Cerca Audio     e Cerca Parametri
        audio_hits, dsp_params = await asyncio.gather(
            self.audio_retriever.retrieve(clean_params),
            self.sound_designer.run(clean_params)
        )

        context_data["results"] = {
            "found_samples": audio_hits,
            "suggested_parameters": dsp_params
        }

if __name__ == "__main__":


    async def main():
        wf = Workflow()

        print("TEST RICERCA")
        res = await wf.run("Find me a bright acid bass")
        print(res)


    asyncio.run(main())




