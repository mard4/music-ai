import asyncio
from rag.agents.IntentClassifierAgent import IntentClassifierAgent
from rag.agents.AudioAnalystAgent import AudioAnalystAgent
from rag.agents.HumanizerAgent import HumanizerAgent
from rag.agents.LabelEnricherAgent import LabelEnricher
from rag.agents.RetrievalAgent import AudioRetriever
from rag.agents.SoundDesignerAgent import SoundDesignerAgent

from rag.utils import logger, FoundSimilarAudios


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

        self.memory = {
            "last_analysis": None
        }

        logger.info("Workflow pronto.")

    async def run(self, user_input: str, file_path: str = None) -> str:
        """
        Esegue il flusso principale. Se c'è un file_path, forza l'analisi.
        """

        intent = None
        clean_params = None
        has_memory = bool(self.memory["last_analysis"])
        context_data = {
            "user_query": user_input,
            "intent": "",
            "results": {}
        }


        # CASO A: Nuovo File Caricato -> Analisi + Memorizzazione
        if file_path:
            logger.info(f"File audio rilevato. Intent -> ANALYSIS")
            intent = "ANALYSIS"
            clean_params = file_path
            if not user_input.strip():
                context_data["user_query"] = "Analizza questo file audio."

        # CASO B: Solo testo, usiamo il Classificatore
        else:
            # Passiamo has_memory al classifier!
            classification = self.classifier.run(user_input, has_context=has_memory)

            intent = classification.get("intent")
            clean_params = classification.get("params")

            # --- LOGICA GESTITA DAL PROMPT ---
            # Se il classifier ci dice "USE_LAST_ANALYSIS", attiviamo la memoria
            if intent == "RETRIEVAL" and clean_params == "USE_LAST_ANALYSIS":
                logger.info("Classifier triggered Contextual Search.")

                if has_memory:
                    last_ana = self.memory["last_analysis"]
                    tags = last_ana.get("smart_tags", [])
                    desc = last_ana.get("description", "")

                    # Costruiamo la query reale
                    clean_params = f"{desc} {', '.join(tags)}"
                    context_data["is_contextual_search"] = True
                else:
                    # Caso difensivo (non dovrebbe succedere se il prompt funziona bene)
                    return "No analysis in memory."

            logger.info(f"Routing attivo: {intent} | Target: {clean_params}")

        context_data["intent"] = intent

        if intent == "ANALYSIS":
            analysis_result = await self.intent_analysis(clean_params, context_data)

            if analysis_result and "analysis" in analysis_result:
                self.memory["last_analysis"] = analysis_result["analysis"]
                logger.info("Analisi salvata in memoria per richieste future.")

        elif intent == "RETRIEVAL":
            await self.intent_retrieval(clean_params, context_data)

        else:
            context_data["results"] = {"error": "Intent not recognized."}

        # STEP 3: Generazione Risposta (LLM)
        # L'Humanizer ora riceverà in context_data["results"] l'analisi tecnica del file
        # e genererà una risposta discorsiva.
        final_response = self.humanizer.generate_response(
            user_query=context_data["user_query"],
            intent=intent,
            data=context_data
        )

        return final_response

    async def intent_analysis(self, clean_params, context_data):
        # 1. Esegui l'analisi (che trova i neighbors internamente)
        analysis_result = await self.analyst.run(clean_params)

        # 2. RECUPERA GLI AUDIO PER I NEIGHBORS (La parte nuova)
        # 'analysis_result' ha una chiave 'recommendations' che contiene i vicini
        neighbors = analysis_result.get("recommendations", [])

        processed_neighbors = []
        for sample in neighbors:
            # Usiamo la TUA utility per generare l'URL pubblico
            web_link = await FoundSimilarAudios().prepare_audio_for_web(sample)
            if web_link:
                sample['web_url'] = web_link
            processed_neighbors.append(sample)

        # Aggiorniamo la lista con i link
        analysis_result["recommendations"] = processed_neighbors

        context_data["results"] = analysis_result
        return analysis_result

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

        processed_hits = []
        for hit in audio_hits:
            web_link = await FoundSimilarAudios().prepare_audio_for_web(hit)

            if web_link:
                hit['web_url'] = web_link

            processed_hits.append(hit)

        context_data["results"] = {
            "found_samples": processed_hits,
            "suggested_parameters": dsp_params
        }

if __name__ == "__main__":


    async def main():
        wf = Workflow()

        print("TEST RICERCA")
        res = await wf.run("Find me a bright acid bass")
        print(res)


    asyncio.run(main())




