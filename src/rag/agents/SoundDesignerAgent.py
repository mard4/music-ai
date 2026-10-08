import logging
import asyncio
from typing import Dict, Any

from rag.agents.AgentBase import AgentBase
from rag.tools.dsp_search import DSPSearchTool
from rag.utils import read_prompt


class SoundDesignerAgent(AgentBase):
    """
    AGENT: Sound Designer.
    Genera ricette DSP usando i dati del DB e un prompt rigoroso per la formattazione.
    """

    def __init__(self):
        super().__init__(agent_name="SoundDesigner")
        self.dsp_tool = DSPSearchTool()

    async def run(self, query: str) -> Dict[str, Any]:
        self.logger.info(f"DSP Search for: '{query}'")

        # 1. Recupero Parametri
        channel_strip_slots = [
            ("Tone", "equalizer frequency response"),
            ("Dynamics", "compressor settings"),
            ("FX", "distortion saturation reverb parameters")
        ]

        tasks = []
        for role, suffix in channel_strip_slots:
            targeted_query = f"{query} {suffix}".strip()
            tasks.append(self.dsp_tool.search_parameters(targeted_query, limit=1))

        results_list = await asyncio.gather(*tasks)

        # 2. Costruzione Chain
        recipe_chain = []
        used_effects = set()

        for i, (role, _) in enumerate(channel_strip_slots):
            found_effects = results_list[i]
            if not found_effects: continue

            best_match = found_effects[0]
            effect_type = best_match.get("effect_type", "").title()

            if effect_type in used_effects: continue

            raw_keys = best_match.get("param_keys", [])
            raw_values = best_match.get("param_values", [])
            formatted_params = []

            for k, v in zip(raw_keys, raw_values):
                clean_val = str(v)
                if isinstance(v, (int, float)):
                    if "freq" in k.lower() or v > 1000:
                        clean_val = f"{int(v)} Hz"
                    elif "gain" in k.lower() or "threshold" in k.lower():
                        clean_val = f"{v:.1f} dB"
                    elif "time" in k.lower() or "attack" in k.lower():
                        clean_val = f"{v:.2f} ms"
                    else:
                        clean_val = f"{v:.2f}"

                formatted_params.append({"name": k.replace("_", " ").title(), "value": clean_val})

            recipe_chain.append({
                "role": role,
                "effect_name": effect_type,
                "params": formatted_params
            })
            used_effects.add(effect_type)

        if not recipe_chain:
            return {"found": False, "html_output": ""}

        # 3. Generazione Output con LLM (Semi-Strict)
        try:
            prompt_text = read_prompt("humanizer_sound_design.txt")
            if not prompt_text:
                prompt_text = "Display these parameters: {{ recipe_chain }}"

            self.prompt = prompt_text

            rendered_msg = self.render_prompt(
                concept=query,
                recipe_chain=recipe_chain
            )

            messages = [
                {"role": "system",
                 "content": "You are a Technical Data Reporter. Follow the template provided strictly."},
                {"role": "user", "content": rendered_msg}
            ]

            # Temperatura moderata: permette di variare leggermente le frasi di raccordo,
            ai_commentary = self.call_llm(messages, temperature=0.4)

            if isinstance(ai_commentary, dict):
                html_output = ai_commentary.get("content", "")
            else:
                html_output = str(ai_commentary)

            html_output = html_output.replace("```html", "").replace("```", "").strip()

        except Exception as e:
            self.logger.error(f"Errore generazione LLM Sound Design: {e}")
            html_output = ""

        return {
            "found": True,
            "concept": query,
            "recipe_chain": recipe_chain,
            "html_output": html_output
        }