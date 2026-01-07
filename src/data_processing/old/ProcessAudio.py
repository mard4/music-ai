import os
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm
from bson import ObjectId
import numpy as np
from motor.motor_asyncio import AsyncIOMotorGridFSBucket, AsyncIOMotorDatabase, AsyncIOMotorCollection
from motor.motor_asyncio import AsyncIOMotorClient

from infrastructure.database.repositories import MongoAudioFilesRepository
from data_processing.old.AudioFeatureExtractor import AudioFeatureExtractor
from core.domain.audio import AudioFile, EnrichedAudioFile

import logging

logging.getLogger('pymongo').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('librosa').setLevel(logging.WARNING)


class AudioProcessor:
    """Processes audio files from MongoDB, extracts features, and updates metadata."""

    def __init__(
            self,
            repository: MongoAudioFilesRepository,
            gridfs_bucket: AsyncIOMotorGridFSBucket,
            database: AsyncIOMotorDatabase
    ):
        self.repository = repository
        self.gridfs_bucket = gridfs_bucket
        self.database = database
        self.extractor = AudioFeatureExtractor()

    async def process_audio_files(self) -> None:
        """Process all audio files from MongoDB."""

        audio_files = await self._fetch_audio_files()
        enriched_collection = self.database.enriched_audio_samples

        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            await self._process_single_file(audio_file, enriched_collection)

    async def _fetch_audio_files(self) -> List[AudioFile]:
        """Fetch all audio files from repository."""
        return await self.repository.find_with_gridfs()

    async def _process_single_file(self, audio_file: AudioFile,
                                   enriched_collection: AsyncIOMotorCollection) -> None:
        """Process a single audio file."""

        if not audio_file.gridfs_file_id:
            return

        temp_path = await self._download_gridfs_file(audio_file.gridfs_file_id)
        if not temp_path:
            return

        try:
            await self._extract_and_save_features(
                audio_file, temp_path, enriched_collection
            )
        finally:
            self._cleanup_temp_file(temp_path)

    async def _download_gridfs_file(self, gridfs_id: str) -> Optional[Path]:
        """Download file from GridFS to temporary location."""

        try:
            grid_out = await self.gridfs_bucket.open_download_stream(ObjectId(gridfs_id))
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(await grid_out.read())
            return temp_path
        except Exception:
            return None

    async def _extract_and_save_features(
            self,
            audio_file: AudioFile,
            temp_path: Path,
            enriched_collection: AsyncIOMotorCollection
    ) -> None:
        """Extract features and save enriched data."""

        features = self.extractor.extract_audio_features(temp_path)
        if not features:
            return

        technical_desc = self.extractor.generate_technical_description(features)
        semantic_desc = self.extractor.generate_semantic_description(features)
        text_descriptions = self.extractor.create_descriptions(temp_path)

        enriched_audio = self._create_enriched_audio_file(
            audio_file, features, technical_desc, semantic_desc, text_descriptions
        )

        await self._save_enriched_data(enriched_audio, enriched_collection)

    def _create_enriched_audio_file(
            self,
            audio_file: AudioFile,
            features: Dict[str, Any],
            technical_desc: List[str],
            semantic_desc: List[str],
            text_descriptions: List[str]
    ) -> EnrichedAudioFile:
        """Create enriched audio file object with extracted features."""

        # Converti tipi numpy a Python nativi
        spectral_centroid = features.get('spectral_centroid')
        rms = features.get('rms')
        duration = features.get('duration')

        # Crea l'oggetto EnrichedAudioFile
        enriched_audio = EnrichedAudioFile(
            sample=audio_file.sample,
            metadata=audio_file.metadata,
            gridfs_file_id=audio_file.gridfs_file_id,
            text_descriptions=text_descriptions,
            technical_descriptions=technical_desc,
            semantic_descriptions=semantic_desc,
            spectral_centroid=(
                float(spectral_centroid)
                if spectral_centroid is not None and isinstance(spectral_centroid, (np.floating, np.integer))
                else None
            ),
            rms_energy=(
                float(rms)
                if rms is not None and isinstance(rms, (np.floating, np.integer))
                else None
            ),
            duration_seconds=(
                float(duration)
                if duration is not None and isinstance(duration, (np.floating, np.integer))
                else None
            )
        )

        return enriched_audio

    async def _save_enriched_data(
            self,
            enriched_audio: EnrichedAudioFile,
            enriched_collection: AsyncIOMotorCollection
    ) -> None:
        """Save enriched audio data to collection."""

        # Converti l'oggetto EnrichedAudioFile in dizionario serializzabile
        enriched_data = self._convert_enriched_audio_to_dict(enriched_audio)

        await enriched_collection.update_one(
            {"gridfs_file_id": enriched_audio.gridfs_file_id},
            {"$set": enriched_data},
            upsert=True
        )

    def _convert_enriched_audio_to_dict(self, enriched_audio: EnrichedAudioFile) -> Dict[str, Any]:
        """Convert EnrichedAudioFile object to serializable dictionary."""

        # Usa model_dump per Pydantic v2 o dict per v1
        if hasattr(enriched_audio, 'model_dump'):
            data = enriched_audio.model_dump()
        else:
            data = enriched_audio.dict()

        # Assicurati che tutti i valori siano serializzabili
        return self._make_serializable(data)

    def _make_serializable(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy types and other non-serializable objects to Python native types."""

        def convert_value(value):
            if isinstance(value, (np.floating, np.float32, np.float64)):
                return float(value)
            elif isinstance(value, (np.integer, np.int32, np.int64)):
                return int(value)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            else:
                return value

        return {k: convert_value(v) for k, v in data.items()}

    def _cleanup_temp_file(self, temp_path: Path) -> None:
        """Remove temporary file."""
        if temp_path.exists():
            os.unlink(temp_path)


async def create_audio_processor(mongo_config: dict = None) -> AudioProcessor:
    """Factory to create audio processor using application settings."""


    client = AsyncIOMotorClient(mongo_config["connection_string"])
    db = client[mongo_config["database_name"]]
    gridfs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name=mongo_config["fs_collection"])
    collection = db[mongo_config["audio_collection"]]
    repository = MongoAudioFilesRepository(collection)

    return AudioProcessor(repository, gridfs_bucket, db)