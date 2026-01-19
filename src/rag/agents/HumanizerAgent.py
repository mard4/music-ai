import json
from typing import Dict, Any, List
from rag.agents.AgentBase import AgentBase
from rag.utils import read_prompt, logger


class HumanizerAgent(AgentBase):
    """
    AGENT: Humanizer ("Il Volto").
    Ruolo: Sintetizzare dati tecnici (JSON) in una risposta naturale utile all'utente.
    """

    def __init__(self):
        super().__init__(agent_name="Humanizer")

        # cambia dinamicamente in base all'intent Cache dei template per evitare I/O ripetuto
        self.templates = {
            "ANALYSIS": read_prompt("humanizer_analysis.txt"),
            "RETRIEVAL": read_prompt("humanizer_retrieval.txt"),
            "DEFAULT": read_prompt("humanizer_default.txt")
        }

    def generate_response(self, user_query: str, intent: str, data: Dict[str, Any]) -> str:
        """
        Genera la risposta finale basata sui dati raccolti dagli altri agenti.
        """
        self.logger.info(f"Humanizer: Generazione risposta per intent '{intent}'")

        # 1. Selezione del Template in base all'intento
        # Se l'intent non è mappato, usa DEFAULT
        template_content = self.templates.get(intent, self.templates["DEFAULT"])

        # Sovrascriviamo il template corrente dell'AgentBase
        self.prompt = template_content

        # 2. Preparazione dei messaggi
        # Usiamo render_prompt per iniettare i dati nel template Jinja2
        try:
            # Passiamo 'results' che corrisponde alla chiave usata nel prompt Jinja (es. results.found_samples)
            # 'data' solitamente contiene {"results": ...} dal workflow, o lo passiamo direttamente.
            # Assumiamo dal workflow.py che data["results"] contenga i dati veri.
            context_data = data.get("results", {}) if "results" in data else data

            dynamic_system_prompt = self.render_prompt(
                user_query=user_query,
                results=context_data
            )

            messages = [
                {"role": "system", "content": dynamic_system_prompt},
                {"role": "user", "content": "Generate the final response for the user"}
            ]

            # 3. Chiamata LLM
            response = self.call_llm(messages, temperature=0.7)

            return response if response else "Non sono riuscito a generare una descrizione."

        except Exception as e:
            self.logger.error(f"Errore generazione risposta: {e}", exc_info=True)
            return "Mi dispiace, ho riscontrato un errore nel sintetizzare i risultati."