import requests
import cloudscraper
import re
import os
import time
from privacy import get_random_user_agent, HumanBehavior, RateLimiter

class SampleFocusAutomator:
    def __init__(self):
        self.scraper = cloudscraper.create_scraper()
        self.human_behavior = HumanBehavior()
        self.rate_limiter = RateLimiter()
        
    def get_download_headers(self, referer_url):
        """Headers specifici per il download MP3"""
        return {
            'User-Agent': get_random_user_agent(),
            'Accept': '*/*',
            'Accept-Encoding': 'identity',  # Importante per download MP3
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': referer_url,  # CRITICO: deve matchare la pagina di origine
            'Origin': 'https://samplefocus.com',
            'Sec-Fetch-Dest': 'audio',
            'Sec-Fetch-Mode': 'no-cors',
            'Sec-Fetch-Site': 'cross-site',
            'Connection': 'keep-alive',
        }
    
    def extract_mp3_url(self, url):
        """Estrae URL MP3 dalla pagina - versione semplificata e efficace"""
        try:
            print(f"🔍 Analizzando: {url}")
            response = self.scraper.get(url)
            print(f"📄 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                # Pattern più diretto e efficace
                patterns = [
                    r'https://d9olupt5igjta\.cloudfront\.net/samples/sample_files/\d+/[a-f0-9]+/mp3/[^\s"\'<>]+\.mp3\?[^\s"\'<>]+',
                    r'contentUrl["\']?\s*content=["\']([^"\']+\.mp3[^"\']*)["\']',
                    r'"sample_mp3_url":"([^"]+)"',
                    r'mp3_url["\']?\s*:\s*["\']([^"\']+\.mp3[^"\']*)["\']'
                ]
                
                for i, pattern in enumerate(patterns):
                    matches = re.findall(pattern, response.text)
                    if matches:
                        mp3_url = matches[0].replace('\\/', '/')
                        print(f"URL MP3 trovato (Pattern {i+1})")
                        print(f"URL: {mp3_url}")
                        return mp3_url
                
                # Fallback: cerca qualsiasi URL MP3 nella pagina
                mp3_pattern = r'https?://[^\s"\'<>]+\.mp3(?:\?[^\s"\'<>]*)?'
                fallback_matches = re.findall(mp3_pattern, response.text)
                if fallback_matches:
                    mp3_url = fallback_matches[0].replace('\\/', '/')
                    print(f"URL: {mp3_url}")
                    return mp3_url
                
                print("❌ Nessun URL MP3 trovato")
                
            else:
                print(f"❌ Errore HTTP: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Errore nell'estrazione: {e}")
            
        return None

    def download_file(self, mp3_url, page_url, output_dir="downloads"):
        """Scarica il file MP3 - versione semplificata e funzionante"""
        try:
            # Crea directory se non esiste
            os.makedirs(output_dir, exist_ok=True)
            
            # Genera nome file
            sample_name = page_url.split('/')[-1]
            filename = f"{sample_name}.mp3"
            filepath = os.path.join(output_dir, filename)
            
            print(f"⬇️ Scaricando: {filename} da: {mp3_url}")
            
            headers = self.get_download_headers(page_url)
            
            # Fai la richiesta
            response = requests.get(
                mp3_url, 
                headers=headers, 
                stream=True, 
                timeout=30
            )
            
            print(f"Download Status: {response.status_code}")
            
            if response.status_code == 200:
                total_size = 0
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            total_size += len(chunk)
                
                file_size_mb = total_size / (1024 * 1024)
                print(f"Download completato: {filename} ({file_size_mb:.2f} MB)")
                return True
            else:
                print(f"❌ Errore nel download: Status {response.status_code}")
                return False
            
        except Exception as e:
            print(f"❌ Errore nel download: {e}")
            return False
    
    def process_single_sample(self, page_url, output_dir="downloads"):
        """Processa un singolo sample"""
        print(f"\n🎵 Processing: {page_url}")
        
        # Estrai URL MP3
        mp3_url = self.extract_mp3_url(page_url)
        
        if mp3_url:
            # Download diretto
            return self.download_file(mp3_url, page_url, output_dir)
        else:
            print("❌ Impossibile procedere con il download")
            return False
    
    def process_multiple_samples(self, url_list, output_dir="downloads"):
        """Processa multipli samples"""
        results = []
        
        for i, url in enumerate(url_list, 1):
            print(f"\n{'='*50}")
            print(f"🎵 Processing {i}/{len(url_list)}: {url}")
            
            success = self.process_single_sample(url, output_dir)
            results.append((url, success))
            
            # Aspetta tra i download
            if i < len(url_list):
                delay = random.uniform(2, 5)
                print(f"⏳ Attesa di {delay:.1f} secondi...")
                time.sleep(delay)
        
        return results


def main():
    automator = SampleFocusAutomator()
    
    # Configurazione
    OUTPUT_DIR = "./data/downloads"
    
    print("=== SAMPLEFOCUS DOWNLOADER ===")
    
    # 📥 SINGOLO DOWNLOAD
    print("\n1. DOWNLOAD SINGOLO")
    single_url = "https://samplefocus.com/samples/woman-harmonizing-vocal"
    success = automator.process_single_sample(single_url, OUTPUT_DIR)
    
    if success:
        print("🎉 Download singolo completato con successo!")
    else:
        print("💥 Download singolo fallito")
    
    # 📥 MULTIPLI DOWNLOAD (opzionale)
    print("\n2. DOWNLOAD MULTIPLI")
    sample_urls = [
        "https://samplefocus.com/samples/woman-harmonizing-vocal",
        # Aggiungi altri URL qui se vuoi testare multipli
        # "https://samplefocus.com/samples/your-sample-url",
    ]
    
    if len(sample_urls) > 1:
        results = automator.process_multiple_samples(sample_urls, OUTPUT_DIR)
        
        # Riepilogo
        print("\n📊 RIEPILOGO DOWNLOAD:")
        success_count = sum(1 for _, success in results if success)
        print(f"✅ Successi: {success_count}/{len(results)}")
        print(f"❌ Falliti: {len(results) - success_count}/{len(results)}")


if __name__ == "__main__":
    import random
    main()