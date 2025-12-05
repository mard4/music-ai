from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class Sample(BaseModel):
    """Modello per campione audio."""

    file_name: str = Field(..., description="Nome del file audio")
    file_type: Optional[str] = Field(None, description="Tipo di file: mp3, wav, etc.")
    label: str = Field(..., description="Etichetta descrittiva")
    source: Optional[str] = Field(None, description="Origine del campione")


class AudioMetadata(BaseModel):
    """Metadata per file audio."""

    categories: Optional[List[str]] = Field(None, description="Categorie musicali")
    key: Optional[str] = Field(None, description="Tonalità musicale")
    bpm: Optional[str] = Field(None, description="Battiti per minuto")
    duration: Optional[str] = Field(None, description="Durata in secondi")
    split: Optional[str] = Field(None, description="Split del dataset")
    original_split: Optional[str] = Field(None, description="Split originale")
    main_category: str = Field(None, description="Nome del main category")


class AudioFile(BaseModel):
    """File audio completo con metadata."""

    sample: Sample
    metadata: AudioMetadata
    gridfs_file_id: Optional[str] = Field(None, description="ID GridFS")

    class Config:
        arbitrary_types_allowed = True


class EnrichedAudioFile(AudioFile):
    """File audio arricchito con features."""

    text_descriptions: Optional[List[str]] = Field(None, description="Descrizioni testuali")
    technical_descriptions: Optional[List[str]] = Field(None, description="Descrizioni tecniche")
    semantic_descriptions: Optional[List[str]] = Field(None, description="Descrizioni semantiche")
    spectral_centroid: Optional[float] = Field(None, description="Centroide spettrale")
    rms_energy: Optional[float] = Field(None, description="Energia RMS")
    duration_seconds: Optional[float] = Field(None, description="Durata in secondi")