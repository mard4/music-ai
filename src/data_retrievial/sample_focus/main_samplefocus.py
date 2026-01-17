import asyncio
import os
from SampleFocusExtractor import download_by_category_to_mongo
from config.settings import settings, mongo_config
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


async def main():

    TARGET_PER_INTERSECTION = 250

    # ASSE X: Sorgenti Sonore (Strumenti/Famiglie)
    instruments = [
        "analog-synth-bass", "bass-line", "sub-bass",  # Bass Family
        "kicks", "toms", "claps", "rides", "snare",  # Drums Family
        "synth pad", "synth lead", "pluck"  # Synth Family
    ]

    # ASSE Y: Qualità Timbriche (Attributi Percettivi)
    timbres = [
        "warm","cold","soft","happy","heavy","airy", "bright", "dark"
    ]

    # ==========================================
    # BLACKLIST SEMANTICA (Filtro Qualità)
    # ==========================================
    impossible_combinations = {
        ("sub bass", "airy"),  # Il sub bass non ha alte frequenze per essere "airy"
        ("sub bass", "bright"),  # Contraddizione fisica
        ("kicks", "airy"),  # Un kick solitamente è solido, non arioso
        ("rides", "warm"),  # I piatti sono metallici (freddi/bright) per definizione
        ("claps", "dark"),  # I clap sono transienti sulle medie-alte
        ("bass line", "airy"),  # Raro, rischia di portare rumore
    }

    total_successes = 0

    # ==========================================
    # 4. ESECUZIONE MATRICE (Loop Incrociato)
    # ==========================================
    print(f"--- Inizio raccolta dati Matrix Strategy ---")
    print(f"Target per cella: {TARGET_PER_INTERSECTION}")

    for category in instruments:
        for timbre in timbres:

            # 1. Controllo Blacklist
            if (category, timbre) in impossible_combinations:
                logging.warning(f"SKIP: Combinazione semanticamente improbabile: '{category}' + '{timbre}'")
                continue

            # 2. Costruzione URL Filtrata (Category + Tag)
            # Pattern: https://samplefocus.com/categories/{category}?tags[]={timbre}
            base_url = f"https://samplefocus.com/categories/{category}"
            query_params = f"?tags[]={timbre}&min_tempo=0&max_tempo=200"

            intersection_url = base_url + query_params

            print(f"\nProcessing Intersezione: [{category.upper()}] + [{timbre.upper()}]")
            print(f"URL: {intersection_url}")

            # 3. Download con Cap
            results = await download_by_category_to_mongo(
                category_url=intersection_url,
                max_samples=TARGET_PER_INTERSECTION,
                mongo_config=mongo_config
            )

            success_count = sum(1 for r in results if r)
            total_successes += success_count
            print(f" -> Scaricati: {success_count}/{TARGET_PER_INTERSECTION} files per questa cella.")

            await asyncio.sleep(2)

    print(f"\n==========================================")
    print(f"RACCOLTA COMPLETATA")
    print(f"Totale campioni nel dataset: {total_successes}")
    print(f"==========================================")


if __name__ == "__main__":
    asyncio.run(main())