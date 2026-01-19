import asyncio
from rag.agents.IntentClassifierAgent import IntentClassifierAgent
from rag.agents.AudioAnalystAgent import AudioAnalystAgent
from rag.agents.HumanizerAgent import HumanizerAgent
from rag.agents.RetrievalAgent import AudioRetriever
from rag.agents.SoundDesignerAgent import SoundDesignerAgent

from rag.utils import logger

class Workflow:
    def __init__(self):
        logger.info("Inizializzazione Workflow AI...")

        # 1. CERVELLO: Capisce l'intenzione
        # Input: Query utente -> Output: Intent + Parametri puliti
        self.classifier = IntentClassifierAgent()

        # 2. RAMO ANALISI: Gestisce file
        # Contiene internamente LabelEnricher e logica di confronto
        self.analyst = AudioAnalystAgent()

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
        clean_params = classification.get("params")  # Es. path del file o keywords di ricerca

        logger.info(f"Routing attivo: {intent} | Target: {clean_params}")

        context_data = {
            "user_query": user_input,
            "intent": intent,
            "results": {}
        }

        # STEP 2: Branching (Il Bivio)
        if intent == "ANALYSIS":
            # --- PERCORSO A: Analisi File Locale ---
            # L'Analyst si occupa di tutto: arricchimento e ricerca similare
            # clean_params qui è il path del file
            analysis_result = self.analyst.run(clean_params)
            context_data["results"] = analysis_result

        elif intent == "RETRIEVAL":
            # --- PERCORSO B: Ricerca nel Database ---
            # Eseguiamo in parallelo o sequenziale la ricerca di audio e parametri
            # clean_params qui sono le keyword (es. "kick drum distorto")

            # A. Cerca Audio     e Cerca Parametri
            audio_hits, dsp_params = await asyncio.gather(
                self.audio_retriever.retrieve(clean_params, k=3),
                self.sound_designer.run(clean_params)
            )

            context_data["results"] = {
                "found_samples": audio_hits,
                "suggested_parameters": dsp_params
            }

        else:
            # Fallback per intenti sconosciuti
            context_data["results"] = {"error": "Intent not recognized."}

        # STEP 3: Generazione Risposta
        final_response = self.humanizer.generate_response(
            user_query=user_input,
            intent=intent,
            data=context_data
        )

        return final_response


if __name__ == "__main__":


    async def main():
        wf = Workflow()

        # Test Ricerca
        print("\n--- TEST RICERCA ---")
        res = await wf.run("Find me a bright acid bass")
        print(res)


    asyncio.run(main())




