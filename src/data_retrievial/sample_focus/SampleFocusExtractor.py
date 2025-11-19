import requests
import cloudscraper
import re
import os
import time
from privacy_utils import get_random_user_agent, HumanBehavior, RateLimiter
from metadata import extract_sample_metadata
from typing import List, Dict
import random
from pathlib import Path

class SampleFocusExtractor:
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
            success = self.download_file(mp3_url, page_url, output_dir)
            if success:
                # Estrai e salva i metadati
                metadata = extract_sample_metadata(page_url, self.scraper)
                if metadata:
                    sample_name = page_url.split('/')[-1]
                    self.save_metadata(metadata, output_dir, sample_name)
            return True
        else:
            print("❌ Impossibile procedere con il download")
            return False
    
    def process_multiple_samples(self, url_list, output_dir="downloads",delay=3):
        """Processa multipli samples"""
        results = []
        
        for i, url in enumerate(url_list, 1):
            print(f"🎵 Processing {i}/{len(url_list)}: {url}")
            
            # Ensure we are passing only the URL string, not a tuple
            success = self.process_single_sample(url, output_dir)
            results.append((url, success))
            
            # Aspetta tra i download
            if i < len(url_list):
                delay = random.uniform(2, 5)
                print(f"⏳ Attesa di {delay:.1f} secondi...")
                time.sleep(delay)
        
        return results
    
    
    def extract_from_sample_list(self, list_url, max_pages=1):
        """Estrae URLs di samples da una pagina di lista (categoria, ricerca, etc.)"""
        sample_urls = []
        
        try:
            print(f"📄 Estraendo samples da: {list_url}")
            response = self.scraper.get(list_url)
            if response.status_code != 200:
                print(f"❌ Errore nell'accedere alla lista: {response.status_code}")
                return sample_urls

            # Pattern per trovare links ai samples individuali
            # Cerchiamo URL che contengono /samples/ seguito da uno slug
            pattern = r'https://samplefocus\.com/samples/[a-zA-Z0-9-]+'
            found_urls = re.findall(pattern, response.text)
            
            # Rimuovi duplicati e filtra solo quelli validi
            unique_urls = list(set(found_urls))
            
            # Filtra ulteriormente: solo URL che hanno il formato corretto (escludi altri percorsi)
            sample_urls = [url for url in unique_urls if re.match(r'https://samplefocus\.com/samples/[a-zA-Z0-9-]+$', url)]
            
            print(f"📋 Trovati {len(sample_urls)} samples unici")
            
            # Se vogliamo gestire paginazione, possiamo cercare il link alla prossima pagina
            # Ma per ora restituiamo solo la prima pagina
            return sample_urls

        except Exception as e:
            print(f"❌ Errore nell'estrazione della lista: {e}")
            return []
        
        
    def save_metadata(self, metadata: Dict, output_dir: str, filename: str):
        """Salva i metadati in un file JSON"""
        import json
        filepath = os.path.join(output_dir, f"{filename}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"💾 Metadati salvati in: {filepath}")

def download_by_category(category_url, max_samples=10, output_dir="category_downloads"):
    """Scarica samples da una categoria specifica"""
    automator = SampleFocusExtractor()
    
    print(f"🎯 Scaricando dalla categoria: {category_url}")
    
    # Estrai URLs dalla pagina della categoria
    sample_urls = automator.extract_from_sample_list(category_url)
    
    if sample_urls:
        # Limita il numero di samples
        sample_urls = sample_urls[:max_samples]
        print(f"🎵 Scaricando {len(sample_urls)} samples...")
        
        results = automator.process_multiple_samples(sample_urls, output_dir, delay=4)
        return results
    else:
        print("❌ Nessun sample trovato in questa categoria")
        return []

