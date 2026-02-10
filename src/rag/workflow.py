import asyncio
import os
from rag.agents.IntentClassifierAgent import IntentClassifierAgent
from rag.agents.AudioAnalystAgent import AudioAnalystAgent
from rag.agents.HumanizerAgent import HumanizerAgent
from rag.agents.LabelEnricherAgent import LabelEnricher
from rag.agents.RetrievalAgent import AudioRetriever

from rag.utils import logger, FoundSimilarAudios


class Workflow:
    def __init__(self):
        logger.info("Inizializzazione Workflow AI (Mirror Pipeline)...")

        # 1. CERVELLO: Capisce l'intenzione (usato per saluti/fallback)
        self.classifier = IntentClassifierAgent()

        # 2. AUDIO PROCESSOR: Lo usiamo per accedere al modello CLAP (self.analyst.ear)
        self.analyst = AudioAnalystAgent()

        # 3. SEMANTIC BRAIN: Sintetizza i tag dai vicini
        self.label_enricher = LabelEnricher()

        # 4. UNIVERSAL FINDER: Cerca sample (Audio o Testo)
        self.audio_retriever = AudioRetriever()

        # 5. VOCE: Genera la risposta finale
        self.humanizer = HumanizerAgent()

        self.memory = {
            "last_analysis": None
        }

        logger.info("Workflow pronto.")

    async def run(self, user_input: str, file_path: str = None) -> str:
        """
        Esegue la Pipeline Unificata ("The Mirror Pipeline").
        Non c'è più distinzione rigida tra Analisi e Ricerca: tutto passa dal Retrieval.
        """

        # --- STEP 0: CLASSIFICAZIONE BASE ---
        # Serve ancora per gestire saluti ("Ciao") o richieste fuori contesto.
        has_memory = bool(self.memory["last_analysis"])
        classification = self.classifier.run(user_input, has_context=has_memory)
        intent = classification.get("intent")

        # Override dell'intent se c'è un file (Priorità assoluta all'audio)
        if file_path:
            intent = "ANALYSIS"

        # Uscita rapida per chat generica
        if intent == "OTHER":
            return self.humanizer.generate_response(user_input, intent, {"results": classification})

        logger.info(f"Avvio Pipeline | Input: {'Audio File' if file_path else 'Text'} | Intent Derivato: {intent}")

        # Contenitore per i risultati che passeremo all'Humanizer
        pipeline_results = {
            "filename": os.path.basename(file_path) if file_path else "User Query",
            "intent": intent
        }

        # --- FASE 1: DUAL-PATH RETRIEVAL (Trova i simili) ---
        neighbors = []

        if file_path:
            # PERCORSO A: Audio-to-Audio
            # Sfruttiamo il modello CLAP già caricato nell'Analyst per ottenere il vettore
            try:
                # [0] perché get_audio_embedding ritorna una lista di vettori
                audio_vector = self.analyst.ear.get_audio_embedding([file_path])[0].tolist()

                # Cerchiamo usando il vettore audio
                neighbors = await self.audio_retriever.retrieve(audio_vector=audio_vector)
            except Exception as e:
                logger.error(f"Errore generazione embedding audio: {e}")
                return "C'è stato un problema tecnico nell'analizzare il file audio."
        else:
            # PERCORSO B: Text-to-Audio
            # Usiamo i parametri puliti dal classificatore o l'input utente raw
            search_query = classification.get("params") or user_input

            # Gestione contesto
            if intent == "RETRIEVAL" and search_query == "USE_LAST_ANALYSIS" and has_memory:
                last_tags = self.memory["last_analysis"].get("ai_tags", [])
                search_query = " ".join(last_tags)
                pipeline_results["is_contextual"] = True

            neighbors = await self.audio_retriever.retrieve(query=search_query)

        # Arricchimento Neighbors con Link Web (per il player frontend)
        processed_neighbors = []
        for hit in neighbors:
            web_link = await FoundSimilarAudios().prepare_audio_for_web(hit)
            if web_link:
                hit['web_url'] = web_link
            processed_neighbors.append(hit)

        pipeline_results["found_samples"] = processed_neighbors  # Per Humanizer Retrieval
        pipeline_results["recommendations"] = processed_neighbors  # Per Humanizer Analysis (compatibilità)

        # --- FASE 2: SEMANTIC SYNTHESIS (Capisci cos'è) ---
        # Usiamo i vicini trovati per distillare una descrizione e dei tag
        # Se siamo partiti da testo, questo step serve a "raffinare" la richiesta per il DSP

        target_name = os.path.basename(file_path) if file_path else user_input

        synthesis = self.label_enricher.run(
            filename=target_name,
            audio_path=file_path,  # Passiamo None se è solo testo (skip check allucinazioni)
            neighbors=processed_neighbors
        )

        pipeline_results["analysis"] = {
            "description": synthesis.get("generated_label", "N/A"),
            "confidence": synthesis.get("confidence", "Medium"),
            "ai_tags": synthesis.get("ai_tags", []),
            "reasoning": synthesis.get("reasoning", "")
        }

        # Se era un'analisi file, aggiorniamo la memoria a lungo termine
        if file_path:
            self.memory["last_analysis"] = pipeline_results["analysis"]

        # --- FASE 4: HUMANIZER (Presentazione) ---

        context_data = {
            "user_query": user_input,
            "intent": intent,
            "results": pipeline_results
        }

        final_response = self.humanizer.generate_response(
            user_query=user_input,
            intent=intent,
            data=context_data
        )

        return final_response


if __name__ == "__main__":
    async def main():
        wf = Workflow()
        # Test Text Flow
        print("--- TEST TEXT ---")
        res = await wf.run("Find me a bright acid bass")
        print(res)

        # Test Audio Flow (simulation)
        # res_audio = await wf.run("", file_path="path/to/test.wav")


    asyncio.run(main())