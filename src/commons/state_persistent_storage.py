from abc import ABC, abstractmethod
from typing import List, Optional
from data_models.audio_models import Sample, Metadata, AudioFiles

class AudioFilesRepository(ABC):    
    """
    Abstract repository for audio data operations.
    """

    @abstractmethod
    async def insert_audio_file(self, audio_data: AudioFiles) -> str:
        """
        Insert a new audio file document.
        
        Args:
            audio_data: Audio file data using Pydantic model
            
        Returns:
            Inserted document ID
        """
        pass

    @abstractmethod
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
        Retrieve audio files by different filters
        
        Args:
            file_type: Optional[str]: mp3, wav
            label: Optional[str]
            source: Optional[str]: fsd50k, samplefocus
            categories: Optional[List[str]]
            key: Optional[str]
            bpm: Optional[int]
            duration: Optional[str]
            
        Returns:
            List of audio file documents
        """
        pass

    @abstractmethod
    async def get_audio_by_id(self, audio_id: str) -> Optional[AudioFiles]:
        """
        Get audio file by ID.
        
        Args:
            audio_id: Audio file ID
            
        Returns:
            Audio file document or None
        """
        pass