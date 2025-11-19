"""
Modelli Pydantic per la gestione dei file audio.
"""

from typing import Optional, List, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime

class Sample(BaseModel):
    file_name: str = Field(..., description="Nome del file audio")
    file_type: Optional[str] = Field(None, description="Tipo di file: mp3, wav, etc.")
    label: str = Field(..., description="Etichetta descrittiva del file audio")
    source: Optional[str] = Field(None, description="Origine: fsd50k, samplefocus, etc.")

class Metadata(BaseModel):
    categories: Optional[List[str]] = Field(None, description="Categorie musicali")
    key: Optional[str] = Field(None, description="Tonalità musicale")
    bpm: Optional[int] = Field(None, description="Battiti per minuto")
    duration: Optional[str] = Field(None, description="Durata nel formato HH:MM:SS")
    split: Optional[str] = Field(None, description="Suddivisione del dataset: dev, eval, etc.")
    original_split: Optional[str] = Field(None, description="Suddivisione originale del dataset: train, val, etc.")

class FileAudio(BaseModel):
    file_name: str = Field(..., description="Nome del file")
    file: bytes = Field(..., description="File audio in bytes")

class AudioFiles(BaseModel):
    sample: Sample
    metadata: Metadata

class EnrichedAudioFile(Sample):
    text_description: Optional[List[str]] = Field(None, description="Descrizioni testuali")
    technical_description: Optional[List[str]] = Field(None, description="Descrizioni tecniche")
    semantic_description: Optional[List[str]] = Field(None, description="Descrizioni semantiche")
    spectral_centroid: Optional[float] = Field(None, description="Centroide spettrale")
