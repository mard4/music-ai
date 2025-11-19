""" 
Concrete implementations of repository interfaces using Pydantic models.
"""

from typing import List, Optional
from functools import lru_cache
from fastapi import Depends
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorCollection
from bson import ObjectId

from commons.state_persistent_storage import AudioFilesRepository
from commons.mongo_dependecies import get_audiofiles_collection
from commons.data_models.audio_models import Sample, AudioFiles, Metadata

class MongoAudioFilesRepository(AudioFilesRepository):

    def __init__(self, collection: AsyncIOMotorCollection):
        """
        Initialize repository with MongoDB collection.
        
        Args:
            collection: MongoDB collection for audio operations
        """
        self.collection = collection

    async def insert_audio_file(self, audio_data: AudioFiles) -> str:
        """
        Insert a new audio file document using Pydantic model.
        
        Args:
            audio_data: Audio file data using Pydantic model
            
        Returns:
            Inserted document ID
        """
        audio_dict = audio_data.model_dump()
        
        result = await self.collection.insert_one(audio_dict)
        return str(result.inserted_id)

    async def find_audio_by_filter(
        self,
        file_type: Optional[str] = None,
        label: Optional[str] = None,
        source: Optional[str] = None,
        categories: Optional[List[str]] = None,
        key: Optional[str] = None,
        bpm: Optional[int] = None,
        duration: Optional[str] = None
    ) -> List[AudioFiles]:
        """
        Retrieve audio files by different filters using Pydantic models.
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
        docs = await cursor.to_list(length=100)
        
        audio_files = []
        for doc in docs:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
            audio_files.append(AudioFiles(**doc))
        
        return audio_files

    async def get_audio_by_id(self, audio_id: str) -> Optional[AudioFiles]:
        """
        Get audio file by ID.
        
        Args:
            audio_id: Audio file ID
            
        Returns:
            Audio file document or None
        """
        try:
            doc = await self.collection.find_one({"_id": ObjectId(audio_id)})
            if doc:
                doc['_id'] = str(doc['_id'])
                return AudioFiles(**doc)
            return None
        except Exception:
            return None

@lru_cache(maxsize=1)
def get_audio_files_repository(
    collection: AsyncIOMotorCollection = Depends(get_audiofiles_collection)
) -> AudioFilesRepository:
    """
    Dependency provider for AudioFilesRepository.
    
    Args:
        collection: MongoDB collection from dependency injection
        
    Returns:
        AudioFilesRepository: Configured audio files repository instance
    """
    return MongoAudioFilesRepository(collection)