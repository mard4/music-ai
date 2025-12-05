"""
Configurazioni centralizzate dell'applicazione.
Carica variabili d'ambiente e fornisce configurazioni validate.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings
import os

class DatabaseSettings(BaseSettings):
    """Configurazioni del database."""

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
    mongodb_fs_collection: str = Field(
        default="audio_files",
        description="Collection GridFS per file audio"
    )
    mongodb_clean_labels_collection: str = Field(
        default="clean_audio_labels",
        description="Collection per label puliti"
    )

    class Config:
        env_prefix = "MONGODB_"
        case_sensitive = False


class TrainingSettings(BaseSettings):
    """Configurazioni per il training."""

    learning_rate: float = Field(default=1e-4)
    weight_decay: float = Field(default=0.1)
    batch_size: int = Field(default=16)
    num_workers: int = Field(default=0)
    epochs: int = Field(default=10)
    patience: int = Field(default=10)
    log_interval: int = Field(default=10)
    checkpoint_interval: int = Field(default=5)

    # Split ratios
    train_ratio: float = Field(default=0.7)
    val_ratio: float = Field(default=0.15)
    test_ratio: float = Field(default=0.15)

    # Audio processing
    target_sample_rate: int = Field(default=48000)
    max_duration_seconds: float = Field(default=10.0)

    class Config:
        env_prefix = "TRAINING_"
        case_sensitive = False


class CLAPSettings(BaseSettings):
    """Configurazioni specifiche per CLAP."""

    enable_fusion: bool = Field(default=False)
    logit_scale: float = Field(default=100.0)
    device: str = Field(default="cuda")

    class Config:
        env_prefix = "CLAP_"
        case_sensitive = False


class AudioSettings(BaseSettings):
    """Configurazioni audio."""
    target_sample_rate: int = Field(default=48000)
    max_duration_seconds: float = Field(default=10.0)

    class Config:
        env_prefix = "AUDIO_"

class Settings(BaseSettings):
    """Configurazioni globali dell'applicazione."""

    # Environment
    environment: str = Field(default="development")
    debug: bool = Field(default=False)

    # Configurations
    database: DatabaseSettings = DatabaseSettings(mongodb_database_name=os.environ.get("MONGODB_DATABASE_NAME"),
                                                  mongodb_audio_collection=os.environ.get("MONGODB_AUDIO_COLLECTION"),
                                                  mongodb_fs_collection=os.environ.get("MONGODB_FS_COLLECTION"),
                                                  mongodb_connection_string=os.environ.get("MONGODB_CONNECTION_STRING"),
                                                  mongodb_clean_labels_collection=os.environ.get("MONGODB_CLEAN_LABELS_COLLECTION"),
                                                  )
    training: TrainingSettings = TrainingSettings()
    clap: CLAPSettings = CLAPSettings()
    audio: AudioSettings = AudioSettings()

    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    checkpoint_dir: Path = base_dir / "checkpoints"
    log_dir: Path = base_dir / "logs"

    @validator("checkpoint_dir", "log_dir", pre=True)
    def create_dirs(cls, v: Path) -> Path:
        """Crea directory se non esistono."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"

# Singleton instance
settings = Settings()

mongo_config = {
    "connection_string": settings.database.mongodb_connection_string,
    "database_name": settings.database.mongodb_database_name,
    "audio_collection": settings.database.mongodb_audio_collection,
    "fs_collection": settings.database.mongodb_fs_collection,
    "clean_labels_collection": settings.database.mongodb_clean_labels_collection
}