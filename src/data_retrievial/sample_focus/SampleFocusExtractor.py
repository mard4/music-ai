import requests
import cloudscraper
import re
import time
import logging
from typing import List, Dict, Optional
from core.infrastructure.database.dependecies import (
    get_audio_repository,
    get_gridfs_handler,
    get_mongo_client
)
from core.infrastructure.storage.gridfs_handler import GridFSHandler
from core.interfaces.repositories import AudioFilesRepository

from privacy_utils import get_random_user_agent, HumanBehavior, RateLimiter
from metadata import extract_sample_metadata
from core.domain.audio import AudioFile, AudioMetadata, Sample
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger('pymongo').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


class SampleFocusExtractor:
    def __init__(self,
                 repository: AudioFilesRepository,
                 gridfs_handler: GridFSHandler,
                 category_url: Optional[str] = None):
        self.scraper = cloudscraper.create_scraper()
        self.human_behavior = HumanBehavior()
        self.rate_limiter = RateLimiter()
        self.repository = repository
        self.gridfs_handler = gridfs_handler  # Usa l'handler astratto
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
                for pattern in patterns:
                    matches = re.findall(pattern, response.text)
                    if matches:
                        return matches[0].replace('\\/', '/')

                mp3_pattern = r'https?://[^\s"\'<>]+\.mp3(?:\?[^\s"\'<>]*)?'
                fallback_matches = re.findall(mp3_pattern, response.text)
                if fallback_matches:
                    return fallback_matches[0].replace('\\/', '/')
            else:
                logging.debug(f"Errore HTTP: {response.status_code}")
        except Exception as e:
            logging.debug(f"Errore nell'estrazione: {e}")
        return None

    async def download_and_save_to_mongo(self, mp3_url: str, page_url: str,
                                         metadata: Dict) -> bool:
        """Scarica e salva usando GridFSHandler e Repository."""
        try:
            sample_name = page_url.split('/')[-1]
            filename = f"{sample_name}.mp3"

            logging.info(f"Scaricando e salvando: {filename}")
            headers = self.get_download_headers(page_url)
            response = requests.get(mp3_url, headers=headers, stream=True, timeout=30)

            if response.status_code == 200:
                audio_data = b""
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        audio_data += chunk

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

                # Costruiamo l'oggetto AudioFile parziale per usarlo nei metadata di GridFS
                audio_file_temp = AudioFile(sample=sample_obj, metadata=metadata_obj)

                # 1. Upload su GridFS usando l'handler
                file_id = await self.gridfs_handler.upload_file(
                    file_data=audio_data,
                    filename=filename,
                    metadata=audio_file_temp.model_dump()
                )

                # 2. Aggiornamento modello e inserimento nel Repository
                audio_file_temp.gridfs_file_id = file_id
                await self.repository.insert_audio_file(audio_file_temp)

                file_size_mb = len(audio_data) / (1024 * 1024)
                logging.info(f"Salvato: {filename} ({file_size_mb:.2f} MB)")
                return True
            else:
                logging.error(f"Errore download: {response.status_code}")
                return False

        except Exception as e:
            logging.error(f"Errore nel processo di salvataggio: {e}")
            return False

    async def process_single_sample(self, page_url: str) -> bool:
        """Processa un singolo sample."""
        logging.info(f"Processing: {page_url}")
        mp3_url = self.extract_mp3_url(page_url)
        if mp3_url:
            metadata = extract_sample_metadata(page_url, self.scraper)
            if metadata:
                return await self.download_and_save_to_mongo(mp3_url, page_url, metadata)
        return False

    async def process_multiple_samples(self, url_list: List[str]) -> List[bool]:
        """Processa multipli samples e salva in MongoDB."""
        results = []
        for i, url in enumerate(url_list, 1):
            logging.info(f"Processing {i}/{len(url_list)}: {url}")
            success = await self.process_single_sample(url)
            results.append(success)
            if i < len(url_list):
                time.sleep(self.human_behavior.random_delay())
        return results

    def extract_from_sample_list(self, list_url: str, max_pages: int = 1) -> List[str]:
        try:
            response = self.scraper.get(list_url)
            if response.status_code != 200: return []
            pattern = r'https://samplefocus\.com/samples/[a-zA-Z0-9-]+'
            found_urls = re.findall(pattern, response.text)
            unique_urls = list(set(found_urls))
            return [url for url in unique_urls if re.match(r'https://samplefocus\.com/samples/[a-zA-Z0-9-]+$', url)]
        except Exception:
            return []


async def download_by_category_to_mongo(category_url: str, max_samples: int = 10,
                                        mongo_config: dict = None) -> List[bool]:
    """Scarica samples da una categoria specifica e salva in MongoDB."""

    repository = get_audio_repository()
    gridfs_handler = get_gridfs_handler()

    extractor = SampleFocusExtractor(repository, gridfs_handler, category_url)

    logging.info(f"Scaricando dalla categoria: {category_url}")
    sample_urls = extractor.extract_from_sample_list(category_url)

    if sample_urls:
        sample_urls = sample_urls[:max_samples]
        results = await extractor.process_multiple_samples(sample_urls)

        return results
    else:
        return []


async def create_samplefocus_extractor(mongo_config: dict = None) -> SampleFocusExtractor:
    return SampleFocusExtractor(
        repository=get_audio_repository(),
        gridfs_handler=get_gridfs_handler()
    )