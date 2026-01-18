"""
Interfacce astratte per i repository.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from core.domain.audio import AudioFile, SocialFXEntry, EnrichedAudioFile
from core.domain.text import TextDocument


class AudioFilesRepository(ABC):
    """Repository per la gestione di file audio."""

    @abstractmethod
    async def insert_audio_file(self, audio_data: AudioFile) -> str:
        """
        Inserisce un nuovo file audio.
        """
        pass

    @abstractmethod
    async def update_audio_file(self, audio_id: str, update_data: dict) -> bool:
        """
        Aggiorna un file audio esistente.
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
        bpm: Optional[str] = None,
        duration: Optional[str] = None
    ) -> List[AudioFile]:
        """
        Trova file audio per filtri.
        """
        pass

    @abstractmethod
    async def get_audio_by_id(self, audio_id: str) -> Optional[AudioFile]:
        """
        Ottiene un file audio per ID.
        """
        pass

    @abstractmethod
    async def find_all(self) -> List[AudioFile]:
        """
        Ottiene tutti i file audio.
        """
        pass

    @abstractmethod
    async def find_audio_by_metadata(
        self,
        categories: Optional[List[str]] = None,
        key: Optional[str] = None,
        bpm: Optional[int] = None,
        duration: Optional[str] = None,
        main_category: Optional[str] = None

    ) -> List[AudioFile]:
        """
        Trova file audio per metadata.
        """
        pass

    @abstractmethod
    async def find_by_gridfs_id(self, gridfs_id: str) -> Optional[AudioFile]:
        """
        Trova file audio per ID GridFS.
        """
        pass

    @abstractmethod
    async def find_with_gridfs(self) -> List[AudioFile]:
        """
        Trova file audio che hanno riferimento GridFS.
        """
        pass


class TextFilesRepository(ABC):
    """Repository per la gestione di file di testo."""

    @abstractmethod
    async def insert_text_file(self, text_data: TextDocument) -> str:
        """
        Inserisce un nuovo file di testo.
        """
        pass

    @abstractmethod
    async def find_text_by_filter(
        self,
        file_type: Optional[str] = None,
        label: Optional[str] = None,
        source: Optional[str] = None,
        categories: Optional[List[str]] = None
    ) -> List[TextDocument]:
        """
        Trova file di testo per filtri.
        """
        pass

    @abstractmethod
    async def get_text_by_id(self, text_id: str) -> Optional[TextDocument]:
        """
        Ottiene un file di testo per ID.
        """
        pass


class CleanLabelsRepository(ABC):
    """Repository per label puliti."""

    @abstractmethod
    async def get_clean_label(self, audio_id: str) -> Optional[str]:
        """
        Ottiene il clean label per un file audio.
        """
        pass

    @abstractmethod
    async def update_clean_label(self, audio_id: str, label: str) -> bool:
        """
        Aggiorna un clean label.
        """
        pass


class EnrichedAudioRepository(ABC):
    """Repository per file audio arricchiti."""

    @abstractmethod
    async def insert_enriched_audio(self, enriched_data: EnrichedAudioFile) -> str:
        """
        Inserisce dati audio arricchiti.
        """
        pass

    @abstractmethod
    async def update_enriched_audio(
        self,
        gridfs_file_id: str,
        update_data: Dict[str, Any]
    ) -> bool:
        """
        Aggiorna dati audio arricchiti.
        """
        pass

    @abstractmethod
    async def get_enriched_by_gridfs_id(
        self,
        gridfs_file_id: str
    ) -> Optional[EnrichedAudioFile]:
        """
        Ottiene dati arricchiti per ID GridFS.
        """
        pass
    
class SocialFxAudioRepository(ABC):
    """Repository per file audio con metriche social."""

    @abstractmethod
    async def insert_social_fx_audio(self, social_fx_data: SocialFXEntry) -> str:
        """
        Inserisce dati audio con metriche social.
        """
        pass

    @abstractmethod
    async def get_social_fx_by_descriptor(
        self,
        descriptor: str
    ) -> SocialFXEntry:
        """
        Ottiene dati audio social per descrittore.
        """
        pass

    @abstractmethod
    async def find_all(self) -> List[SocialFXEntry]:
        """
        Ottiene tutte le entry SocialFX.
        """
        pass