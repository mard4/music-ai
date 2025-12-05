"""
Modelli di configurazione Pydantic.
"""
from pydantic import BaseModel, Field
from typing import Optional
from .settings import settings


class MongoDBConfig(BaseModel):
    """Configurazione MongoDB."""

    connection_string: str = Field(
        default=settings.database.mongodb_connection_string
    )
    database_name: str = Field(
        default=settings.database.mongodb_database_name
    )
    audio_collection: str = Field(
        default=settings.database.mongodb_audio_collection
    )
    fs_collection: str = Field(
        default=settings.database.mongodb_fs_collection
    )
    clean_labels_collection: str = Field(
        default=settings.database.mongodb_clean_labels_collection
    )