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

class EffectParams(BaseModel):
    """Modello generico per i parametri di un effetto (EQ, Comp, Reverb)."""
    param_values: List[float] = Field(..., description="Valori numerici dei parametri")
    param_keys: List[str] = Field(..., description="Nomi dei parametri (es. 'Low', 'Threshold')")

class SocialFXEntry(BaseModel):
    """
    Modello per una voce della Knowledge Base SocialFX.
    Rappresenta il mapping tra un descrittore semantico e i parametri tecnici.
    """
    descriptor: str = Field(..., description="Aggettivo semantico normalizzato (es. 'warm')")
    effect_type: str = Field("eq", description="Tipo di effetto (es. 'eq', 'comp')")
    parameters: EffectParams = Field(..., description="I parametri tecnici associati")
    sample_count: int = Field(..., description="Numero di campioni originali aggregati per calcolare la media")
    source: str = Field("socialfx-original", description="Dataset di origine")


