import requests
import cloudscraper
import re
import os
import time
import json
import logging
import asyncio
from typing import List, Dict, Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorGridFSBucket

from infrastructure.database.repositories import MongoAudioFilesRepository
from privacy_utils import get_random_user_agent, HumanBehavior, RateLimiter
from metadata import extract_sample_metadata
from core.domain.audio import AudioFile, AudioMetadata, EnrichedAudioFile, Sample
# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger('pymongo').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


class SampleFocusExtractor:
    def __init__(self, repository: MongoAudioFilesRepository = None,
                 fs_bucket=None, category_url: str = None):
        self.scraper = cloudscraper.create_scraper()
        self.human_behavior = HumanBehavior()
        self.rate_limiter = RateLimiter()
        self.repository = repository
        self.fs_bucket = fs_bucket
        self.category_url = category_url

    def get_download_headers(self, referer_url):
        """Headers specifici per il download MP3"""
        return {
            'User-Agent': get_random_user_agent(),
            'Accept': '*/*',
            'Accept-Encoding': 'identity',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': referer_url,
            'Origin': 'https://samplefocus.com',
            'Sec-Fetch-Dest': 'audio',
            'Sec-Fetch-Mode': 'no-cors',
            'Sec-Fetch-Site': 'cross-site',
            'Connection': 'keep-alive',
        }

    def extract_mp3_url(self, url):
        try:
            response = self.scraper.get(url)

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
                        logging.info(f"URL MP3 trovato (Pattern {i + 1})")
                        return mp3_url

                # Fallback: cerca qualsiasi URL MP3 nella pagina
                mp3_pattern = r'https?://[^\s"\'<>]+\.mp3(?:\?[^\s"\'<>]*)?'
                fallback_matches = re.findall(mp3_pattern, response.text)
                if fallback_matches:
                    mp3_url = fallback_matches[0].replace('\\/', '/')
                    return mp3_url

                logging.debug("Nessun URL MP3 trovato")

            else:
                logging.debug(f"Errore HTTP: {response.status_code}")

        except Exception as e:
            logging.debug(f"Errore nell'estrazione: {e}")

        return None

    async def download_and_save_to_mongo(self, mp3_url: str, page_url: str,
                                         metadata: Dict) -> bool:
        """Scarica e salva direttamente in MongoDB."""
        try:
            if not self.repository or not self.fs_bucket:
                logging.error("Repository MongoDB non inizializzato")
                return False

            sample_name = page_url.split('/')[-1]
            filename = f"{sample_name}.mp3"

            logging.info(f"Scaricando e salvando in MongoDB: {filename}")

            headers = self.get_download_headers(page_url)

            # Scarica il file
            response = requests.get(mp3_url, headers=headers, stream=True, timeout=30)

            if response.status_code == 200:
                # Leggi i dati audio
                audio_data = b""
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        audio_data += chunk

                # Crea i modelli Pydantic
                sample_obj = Sample(
                    file_name=filename,
                    file_type="mp3",
                    label=metadata.get('title', ''),
                    source="samplefocus"
                )

                metadata_obj = AudioMetadata(
                    categories=metadata.get('categories', []),
                    bpm=metadata.get('bpm'),
                    duration=metadata.get('duration'),
                    key=metadata.get('key'),
                    main_category=self.category_url.rstrip("/").split("/")[-1] if self.category_url else None
                )

                audio_file = AudioFile(
                    sample=sample_obj,
                    metadata=metadata_obj
                )

                # Salva in GridFSBucket
                upload_stream = self.fs_bucket.open_upload_stream(
                    filename,
                    metadata=audio_file.model_dump()
                )
                await upload_stream.write(audio_data)
                await upload_stream.close()

                file_id = upload_stream._id

                # Aggiorna il documento con file_id
                audio_file_dict = audio_file.model_dump()
                audio_file_dict['gridfs_file_id'] = str(file_id)

                await self.repository.insert_audio_file(AudioFile(**audio_file_dict))

                file_size_mb = len(audio_data) / (1024 * 1024)
                logging.info(f"Salvato in MongoDB: {filename} ({file_size_mb:.2f} MB)")
                return True
            else:
                logging.error(f"Errore nel download: Status {response.status_code}")
                return False

        except Exception as e:
            logging.error(f"Errore nel download/salvataggio MongoDB: {e}")
            return False

    async def process_single_sample(self, page_url: str) -> bool:
        """Processa un singolo sample."""
        logging.info(f"Processing: {page_url}")

        mp3_url = self.extract_mp3_url(page_url)

        if mp3_url:
            # Estrai metadati
            metadata = extract_sample_metadata(page_url, self.scraper)
            if metadata:
                logging.debug(f"Metadati estratti: {metadata}")
                # Scarica e salva in MongoDB
                success = await self.download_and_save_to_mongo(mp3_url, page_url, metadata)
                return success
            else:
                logging.error("Impossibile estrarre metadati")
                return False
        else:
            logging.error("Impossibile trovare URL MP3")
            return False

    async def process_multiple_samples(self, url_list: List[str]) -> List[bool]:
        """Processa multipli samples e salva in MongoDB."""
        results = []

        for i, url in enumerate(url_list, 1):
            logging.info(f"Processing {i}/{len(url_list)}: {url}")

            success = await self.process_single_sample(url)
            results.append(success)

            if i < len(url_list):
                wait_time = self.human_behavior.random_delay()
                time.sleep(wait_time)

        return results

    def extract_from_sample_list(self, list_url: str, max_pages: int = 1) -> List[str]:
        """Estrae URLs di samples da una pagina di lista."""
        sample_urls = []

        try:
            logging.info(f"Estraendo samples da: {list_url}")
            response = self.scraper.get(list_url)
            if response.status_code != 200:
                logging.error(f"Errore nell'accedere alla lista: {response.status_code}")
                return sample_urls

            # Pattern per trovare links ai samples individuali
            pattern = r'https://samplefocus\.com/samples/[a-zA-Z0-9-]+'
            found_urls = re.findall(pattern, response.text)

            # Rimuovi duplicati e filtra solo quelli validi
            unique_urls = list(set(found_urls))

            # Filtra ulteriormente: solo URL che hanno il formato corretto
            sample_urls = [url for url in unique_urls if re.match(
                r'https://samplefocus\.com/samples/[a-zA-Z0-9-]+$', url)]

            logging.info(f"Trovati {len(sample_urls)} samples unici")

            return sample_urls

        except Exception as e:
            logging.error(f"Errore nell'estrazione della lista: {e}")
            return []


async def download_by_category_to_mongo(category_url: str, max_samples: int = 10,
                                        mongo_config: dict = None) -> List[bool]:
    """Scarica samples da una categoria specifica e salva in MongoDB."""

    # Se non viene fornita una configurazione, usa le settings di default
    if mongo_config is None:
        from config.settings import settings
        mongo_config = {
            "connection_string": settings.database.mongodb_connection_string,
            "database_name": settings.database.mongodb_database_name,
            "audio_collection": settings.database.mongodb_audio_collection,
            "fs_collection": settings.database.mongodb_fs_collection,
            "clean_labels_collection": settings.database.mongodb_clean_labels_collection
        }

    # Inizializza MongoDB
    client = AsyncIOMotorClient(mongo_config["connection_string"])
    db = client[mongo_config["database_name"]]
    fs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name=mongo_config["fs_collection"])
    collection = db[mongo_config["audio_collection"]]
    repository = MongoAudioFilesRepository(collection)

    extractor = SampleFocusExtractor(repository, fs_bucket, category_url)

    logging.info(f"Scaricando dalla categoria: {category_url}")

    # Estrai URLs dalla pagina della categoria
    sample_urls = extractor.extract_from_sample_list(category_url)

    if sample_urls:
        # Limita il numero di samples
        sample_urls = sample_urls[:max_samples]
        logging.info(f"Scaricando {len(sample_urls)} samples in MongoDB...")

        results = await extractor.process_multiple_samples(sample_urls)

        # Chiudi connessione
        client.close()

        return results
    else:
        logging.error("Nessun sample trovato in questa categoria")
        client.close()
        return []


async def create_samplefocus_extractor(mongo_config: dict = None) -> SampleFocusExtractor:
    """Factory per creare SampleFocusExtractor con dipendenze MongoDB."""

    client = AsyncIOMotorClient(mongo_config["connection_string"])
    db = client[mongo_config["database_name"]]
    fs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name=mongo_config["fs_collection"])
    collection = db[mongo_config["audio_collection"]]
    repository = MongoAudioFilesRepository(collection)

    return SampleFocusExtractor(repository, fs_bucket)