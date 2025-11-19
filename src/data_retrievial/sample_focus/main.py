from SampleFocusExtractor import SampleFocusExtractor, download_by_category

def main():
    automator = SampleFocusExtractor()
    
    # Configurazione
    OUTPUT_DIR = "./data/downloads"
    categories = {
        "bass": "https://samplefocus.com/categories/bass",
        "vocals": "https://samplefocus.com/categories/vocals",
        "drums": "https://samplefocus.com/categories/drums", 
        "synths": "https://samplefocus.com/categories/synths",
    }
    
    sample_urls = download_by_category(categories["drums"], max_samples=5, output_dir=OUTPUT_DIR)
    # se download_by_category ha già processato e restituito (url, success), prendi solo gli URL
    if sample_urls and isinstance(sample_urls[0], tuple):
        sample_urls = [u for u, _ in sample_urls]
    results = automator.process_multiple_samples(sample_urls, OUTPUT_DIR)
    
    # Riepilogo
    print("\n📊 RIEPILOGO DOWNLOAD:")
    success_count = sum(1 for _, success in results if success)
    print(f"✅ Successi: {success_count}/{len(results)}")
    print(f"❌ Falliti: {len(results) - success_count}/{len(results)}")


if __name__ == "__main__":
    import random
    main()
