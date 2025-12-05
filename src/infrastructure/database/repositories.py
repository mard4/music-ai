"""
Implementazioni concrete dei repository per MongoDB.
"""
import logging
from typing import List, Optional, Dict, Any
from bson import ObjectId

from motor.motor_asyncio import AsyncIOMotorCollection
from core.interfaces.repositories import (
    AudioFilesRepository,
    CleanLabelsRepository,
    EnrichedAudioRepository
)
from core.domain.audio import AudioFile, Sample, AudioMetadata

logger = logging.getLogger(__name__)


class MongoAudioFilesRepository(AudioFilesRepository):
    """Implementazione MongoDB per repository audio."""

    def __init__(self, collection: AsyncIOMotorCollection):
        self.collection = collection

    async def insert_audio_file(self, audio_data: AudioFile) -> str:
        """
        Inserisce un nuovo file audio.
        """
        audio_dict = audio_data.model_dump()
        result = await self.collection.insert_one(audio_dict)
        return str(result.inserted_id)

    async def update_audio_file(self, audio_id: str, update_data: dict) -> bool:
        """
        Aggiorna un file audio esistente.
        """
        try:
            result = await self.collection.update_one(
                {"_id": ObjectId(audio_id)},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating audio file {audio_id}: {e}")
            return False

    async def find_audio_by_filter(
            self,
            file_type: Optional[str] = None,
            label: Optional[str] = None,
            source: Optional[str] = None,
            categories: Optional[List[str]] = None,
            key: Optional[str] = None,
            bpm: Optional[str] = None,
            duration: Optional[str] = None
    ) -> List[AudioFile]:
        """
        Trova file audio per filtri.
        """
        query = {}

        if file_type:
            query['sample.file_type'] = file_type
        if label:
            query['sample.label'] = {'$regex': label, '$options': 'i'}
        if source:
            query['sample.source'] = source
        if categories:
            query['metadata.categories'] = {'$in': categories}
        if key:
            query['metadata.key'] = key
        if bpm:
            query['metadata.bpm'] = bpm
        if duration:
            query['metadata.duration'] = duration

        cursor = self.collection.find(query)
        docs = await cursor.to_list(length=None)

        audio_files = []
        for doc in docs:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
            audio_files.append(AudioFile(**doc))

        return audio_files

    async def get_audio_by_id(self, audio_id: str) -> Optional[AudioFile]:
        """
        Ottiene un file audio per ID.
        """
        try:
            doc = await self.collection.find_one({"_id": ObjectId(audio_id)})
            if doc:
                doc['_id'] = str(doc['_id'])
                return AudioFile(**doc)
            return None
        except Exception as e:
            logger.error(f"Error getting audio by ID {audio_id}: {e}")
            return None

    async def find_all(self) -> List[AudioFile]:
        """
        Ottiene tutti i file audio.
        """
        try:
            cursor = self.collection.find({})
            docs = await cursor.to_list(length=None)

            audio_files = []
            for doc in docs:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                audio_files.append(AudioFile(**doc))

            return audio_files
        except Exception as e:
            logger.error(f"Error finding all audio files: {e}")
            return []

    async def find_audio_by_metadata(
            self,
            categories: Optional[List[str]] = None,
            key: Optional[str] = None,
            bpm: Optional[int] = None,
            duration: Optional[str] = None,
            main_category: Optional[str] = None,
    ) -> List[AudioFile]:
        """
        Trova file audio per metadata.
        """
        return await self.find_audio_by_filter(
            categories=categories,
            key=key,
            bpm=bpm,
            duration=duration
        )

    async def find_by_gridfs_id(self, gridfs_id: str) -> Optional[AudioFile]:
        """
        Trova file audio per ID GridFS.
        """
        try:
            doc = await self.collection.find_one({"gridfs_file_id": gridfs_id})
            if doc:
                doc['_id'] = str(doc['_id'])
                return AudioFile(**doc)
            return None
        except Exception as e:
            logger.error(f"Error finding by GridFS ID {gridfs_id}: {e}")
            return None

    async def find_with_gridfs(self) -> List[AudioFile]:
        """
        Trova file audio che hanno riferimento GridFS.
        """
        try:
            cursor = self.collection.find({"gridfs_file_id": {"$exists": True, "$ne": None}})
            docs = await cursor.to_list(length=None)

            audio_files = []
            for doc in docs:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                audio_files.append(AudioFile(**doc))

            return audio_files
        except Exception as e:
            logger.error(f"Error finding files with GridFS: {e}")
            return []


class MongoCleanLabelsRepository(CleanLabelsRepository):
    """Implementazione MongoDB per repository clean labels."""

    def __init__(self, collection: AsyncIOMotorCollection):
        self.collection = collection

    async def get_clean_label(self, audio_id: str) -> Optional[str]:
        """
        Ottiene il clean label per un file audio.
        """
        try:
            doc = await self.collection.find_one({"gridfs_file_id": audio_id})
            if doc:
                return doc.get('label')
            return None
        except Exception as e:
            logger.error(f"Error getting clean label for {audio_id}: {e}")
            return None

    async def update_clean_label(self, audio_id: str, label: str) -> bool:
        """
        Aggiorna un clean label.
        """
        try:
            result = await self.collection.update_one(
                {"gridfs_file_id": audio_id},
                {"$set": {"label": label}},
                upsert=True
            )
            return result.modified_count > 0 or result.upserted_id is not None
        except Exception as e:
            logger.error(f"Error updating clean label for {audio_id}: {e}")
            return False


class MongoEnrichedAudioRepository(EnrichedAudioRepository):
    """Implementazione MongoDB per repository audio arricchiti."""

    def __init__(self, collection: AsyncIOMotorCollection):
        self.collection = collection

    async def insert_enriched_audio(self, enriched_data: Dict[str, Any]) -> str:
        """
        Inserisce dati audio arricchiti.
        """
        result = await self.collection.insert_one(enriched_data)
        return str(result.inserted_id)

    async def update_enriched_audio(
            self,
            gridfs_file_id: str,
            update_data: Dict[str, Any]
    ) -> bool:
        """
        Aggiorna dati audio arricchiti.
        """
        try:
            result = await self.collection.update_one(
                {"gridfs_file_id": gridfs_file_id},
                {"$set": update_data},
                upsert=True
            )
            return result.modified_count > 0 or result.upserted_id is not None
        except Exception as e:
            logger.error(f"Error updating enriched audio for {gridfs_file_id}: {e}")
            return False

    async def get_enriched_by_gridfs_id(
            self,
            gridfs_file_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Ottiene dati arricchiti per ID GridFS.
        """
        try:
            doc = await self.collection.find_one({"gridfs_file_id": gridfs_file_id})
            if doc and '_id' in doc:
                doc['_id'] = str(doc['_id'])
            return doc
        except Exception as e:
            logger.error(f"Error getting enriched audio for {gridfs_file_id}: {e}")
            return None