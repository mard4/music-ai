"""
Dependency injection per database connections.
"""
import logging
from functools import lru_cache
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from motor.motor_asyncio import AsyncIOMotorGridFSBucket

from config.settings import settings
from core.interfaces.repositories import AudioFilesRepository, SocialFxAudioRepository
from core.infrastructure.database.repositories import (
    MongoAudioFilesRepository,
    MongoCleanLabelsRepository,
    MongoEnrichedAudioRepository, MongoSocialFxRepository
)
from core.infrastructure.storage.gridfs_handler import GridFSHandler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger('pymongo').setLevel(logging.WARNING)


@lru_cache(maxsize=1)
def get_mongo_client() -> AsyncIOMotorClient:
    """
    Returns a cached MongoDB client instance.
    """
    mongo_uri = settings.database.mongodb_connection_string
    logger.info(f"Connecting to MongoDB at: {mongo_uri}")
    return AsyncIOMotorClient(mongo_uri)


@lru_cache(maxsize=1)
def get_mongo_database(
    client: Optional[AsyncIOMotorClient] = None
) -> AsyncIOMotorDatabase:
    """
    Returns a cached MongoDB database instance.
    """
    if client is None:
        client = get_mongo_client()

    db_name = settings.database.mongodb_database_name
    logger.debug(f"Getting database: {db_name}")
    return client[db_name]


@lru_cache(maxsize=1)
def get_gridfs_bucket(
    database: Optional[AsyncIOMotorDatabase] = None
) -> AsyncIOMotorGridFSBucket:
    """
    Returns a cached GridFS bucket instance.
    """
    if database is None:
        database = get_mongo_database()

    bucket_name = settings.database.mongodb_fs_collection
    logger.debug(f"Getting GridFS bucket: {bucket_name}")
    return AsyncIOMotorGridFSBucket(database, bucket_name=bucket_name)


@lru_cache(maxsize=1)
def get_gridfs_handler(
    fs_bucket: Optional[AsyncIOMotorGridFSBucket] = None
) -> GridFSHandler:
    """
    Returns a cached GridFS handler instance.
    """
    if fs_bucket is None:
        fs_bucket = get_gridfs_bucket()

    return GridFSHandler(fs_bucket)


@lru_cache(maxsize=1)
def get_audio_collection(
    database: Optional[AsyncIOMotorDatabase] = None
) -> AsyncIOMotorCollection:
    """
    Returns a cached audio collection instance.
    """
    if database is None:
        database = get_mongo_database()

    collection_name = settings.database.mongodb_audio_collection
    logger.debug(f"Getting audio collection: {collection_name}")
    return database[collection_name]


@lru_cache(maxsize=1)
def get_audio_repository(
    collection: Optional[AsyncIOMotorCollection] = None
) -> AudioFilesRepository:
    """
    Returns a cached audio repository instance.
    """
    if collection is None:
        collection = get_audio_collection()

    return MongoAudioFilesRepository(collection)


@lru_cache(maxsize=1)
def get_enriched_audio_collection(
    database: Optional[AsyncIOMotorDatabase] = None
) -> AsyncIOMotorCollection:
    """
    Returns a cached enriched audio collection instance.
    """
    if database is None:
        database = get_mongo_database()

    collection_name = "enriched_audio_samples"
    logger.debug(f"Getting enriched audio collection: {collection_name}")
    return database[collection_name]


@lru_cache(maxsize=1)
def get_enriched_audio_repository(
    collection: Optional[AsyncIOMotorCollection] = None
) -> MongoEnrichedAudioRepository:
    """
    Returns a cached enriched audio repository instance.
    """
    if collection is None:
        collection = get_enriched_audio_collection()

    return MongoEnrichedAudioRepository(collection)


@lru_cache(maxsize=1)
def get_clean_labels_collection(
    database: Optional[AsyncIOMotorDatabase] = None
) -> AsyncIOMotorCollection:
    """
    Returns a cached clean labels collection instance.
    """
    if database is None:
        database = get_mongo_database()

    collection_name = settings.database.mongodb_clean_labels_collection
    logger.debug(f"Getting clean labels collection: {collection_name}")
    return database[collection_name]


@lru_cache(maxsize=1)
def get_clean_labels_repository(
    collection: Optional[AsyncIOMotorCollection] = None
) -> MongoCleanLabelsRepository:
    """
    Returns a cached clean labels repository instance.
    """
    if collection is None:
        collection = get_clean_labels_collection()

    return MongoCleanLabelsRepository(collection)

@lru_cache(maxsize=1)
def get_socialfx_collection(
    database: Optional[AsyncIOMotorDatabase] = None
) -> AsyncIOMotorCollection:
    """
    Returns a cached SocialFX collection instance.
    """
    if database is None:
        database = get_mongo_database()

    collection_name = getattr(settings.database, "mongodb_socialfx_collection")
    logger.debug(f"Getting SocialFX collection: {collection_name}")
    return database[collection_name]


@lru_cache(maxsize=1)
def get_socialfx_repository(
    collection: Optional[AsyncIOMotorCollection] = None
) -> SocialFxAudioRepository:
    """
    Returns a cached SocialFX repository instance.
    """
    if collection is None:
        collection = get_socialfx_collection()

    return MongoSocialFxRepository(collection)