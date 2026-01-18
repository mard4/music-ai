import os
from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).parent.parent
ENV_FILE_PATH = BASE_DIR / ".env"


class DatabaseSettings(BaseSettings):
    """
    Configurazioni del database MongoDB.
    """
    mongodb_connection_string: str = Field(
        default="mongodb://localhost:27017/",
        description="Connection string per MongoDB"
    )
    mongodb_database_name: str = Field(
        default="audio_db",
        description="Nome del database"
    )
    mongodb_audio_collection: str = Field(
        default="audio_samples",
        description="Collection per i metadata audio"
    )
    mongodb_gridfs_bucket: str = Field(
        default="fs",
        description="Bucket name per GridFS"
    )
    mongodb_socialfx_collection: str = Field(
        default="socialfx_collection",
        description="Collection per SocialFX parameters"
    )


class CLAPSettings(BaseSettings):
    """
    Configurazioni specifiche per CLAP (Inference).
    """
    # Importante: enable_fusion=False per usare il modello 'HTSAT-base' con checkpoint '630k-audioset-best.pt'
    enable_fusion: bool = Field(default=False)
    device: str = Field(default="cuda")
    model_name: str = "HTSAT-tiny"
    use_cuda: bool = Field(default=True)
    audio_sample_rate: int = 48000

class VectorDatabaseSettings(BaseSettings):
    QDRANT_CONNECTION_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_AUDIO_COLLECTION_NAME: str = Field(default="audio_vectors")
    QDRANT_PARAMETERS_COLLECTION_NAME: str = Field(default="socialfx_vectors")

class Settings(BaseSettings):
    """
    Configurazioni globali dell'applicazione.
    Aggrega tutte le sotto-configurazioni e le variabili d'ambiente piatte.
    """

    # App Meta
    app_name: str = "Music AI RAG"
    environment: str = Field(default="development")
    debug: bool = Field(default=False)

    # Moduli
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    clap: CLAPSettings = Field(default_factory=CLAPSettings)

    # OpenAI
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    OPENAI_MODEL: str = Field(default="gpt-4o")

    # Qdrant
    QDRANT_CONNECTION_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_AUDIO_COLLECTION_NAME: str = Field(default="audio_vectors")
    QDRANT_PARAMETERS_COLLECTION_NAME: str = Field(default="socialfx_vectors")

    # External Datasets
    socialfx_dataset_name: str = "seungheondoh/socialfx-original"

    # Paths
    base_dir: Path = BASE_DIR
    log_dir: Path = base_dir.parent / "logs"
    checkpoint_dir: Path = base_dir.parent / "checkpoints"

    @field_validator("log_dir")
    def create_dirs(cls, v: Path) -> Path:
        """Crea automaticamente la directory dei log se non esiste."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    class Config:
        env_file = str(ENV_FILE_PATH)
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Singleton instance da importare nel progetto
settings = Settings()

mongo_config = {
    "connection_string": settings.database.mongodb_connection_string,
    "database_name": settings.database.mongodb_database_name,
    "audio_collection": settings.database.mongodb_audio_collection,
    "socialfx_collection": settings.database.mongodb_socialfx_collection
}