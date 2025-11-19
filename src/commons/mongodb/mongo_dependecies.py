"""
MongoDB dependency injection container for FastAPI.
 
This module provides cached dependencies for MongoDB connections and collections
following the dependency injection pattern. It handles connection pooling,
environment configuration, and resource management for the application.
 
Dependencies:
- get_mongo_client: Provides AsyncIOMotorClient instance
- get_mongo_database: Provides database instance  
- get_audiofiles_collection: Provides audiofiles collection instance
"""
 
import os
import logging
from typing import Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from functools import lru_cache
from fastapi import Depends

logger = logging.getLogger(__name__)
 
def get_mongo_client() -> AsyncIOMotorClient:
    mongo_uri = os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017")
    return AsyncIOMotorClient(mongo_uri)

def get_mongo_database(client: AsyncIOMotorClient) -> AsyncIOMotorDatabase:
    mongo_db = os.getenv("MONGODB_DATABASE_NAME", "audio_db")
    return client[mongo_db]

def get_audiofiles_collection(database: AsyncIOMotorDatabase) -> AsyncIOMotorCollection:
    collection_name = os.getenv("MONGODB_AUDIO_COLLECTION", "audio_samples")
    return database[collection_name]