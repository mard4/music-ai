import asyncio
import sys
import os
from pathlib import Path
from collections import Counter

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
from core.infrastructure.database.dependecies import get_socialfx_repository
from core.domain.audio import SocialFXEntry

# CONFIGURAZIONE SOGLIA
MIN_FREQUENCY = 8
output_file = project_root / "data_exploration" / "descriptors_list.txt"

async def extract_descriptors_via_repo(output_file):
    print(f"--- Estrazione Descrittori via Repository (Clean Arch) ---")

    try:

        repo = get_socialfx_repository()
        entries = await repo.find_all()

        if not entries:
            print("Nessuna entry trovata.")
            return

        print(f"Entità recuperate: {len(entries)}")


        tag_counter = Counter()

        for entry in entries:

            raw_text = entry.descriptor

            if raw_text:
                parts = raw_text.split(',')
                clean_parts = [p.strip().lower() for p in parts if p.strip()]
                tag_counter.update(clean_parts)

        frequent_terms = {
            term: count
            for term, count in tag_counter.items()
            if count >= MIN_FREQUENCY
        }

        sorted_terms = sorted(frequent_terms.keys())
        print(f"Termini unici finali (Freq >= {MIN_FREQUENCY}): {len(sorted_terms)}")

        with open(output_file, "w", encoding="utf-8") as f:
            for term in sorted_terms:
                f.write(f"{term}\n")

        print(f"Lista salvata in: {output_file}")

        for term, count in tag_counter.most_common(10):
            print(f"{term}: {count}")

    except Exception as e:
        print(f"Errore critico: {e}")


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(extract_descriptors_via_repo(output_file=output_file))