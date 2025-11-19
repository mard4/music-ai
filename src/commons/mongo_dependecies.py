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
 
#@lru_cache(maxsize=1)
# def get_mongo_client() -> AsyncIOMotorClient:
#     """
#     Provide singleton MongoDB client instance.
    
#     Returns:
#         AsyncIOMotorClient: Configured MongoDB client
        
#     Raises:
#         ValueError: If MONGODB_URI environment variable is not set
#     """
#     mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
#     logger.info(f"Connecting to MongoDB at {mongo_uri}")
#     return AsyncIOMotorClient(mongo_uri)
 
# @lru_cache(maxsize=1)
# def get_mongo_database(
#     client: AsyncIOMotorClient = Depends(get_mongo_client)
# ) -> AsyncIOMotorDatabase:
#     """
#     Provide MongoDB database instance.
    
#     Args:
#         client: MongoDB client from dependency injection
        
#     Returns:
#         AsyncIOMotorDatabase: Database instance
#     """
#     mongo_db = os.getenv("MONGODB_DB", "music_ai")
#     return client[mongo_db]
 
# def get_audiofiles_collection(
#     database: AsyncIOMotorDatabase = Depends(get_mongo_database)
# ) -> AsyncIOMotorCollection:
#     """
#     Provide audiofiles collection instance.
    
#     Args:
#         database: MongoDB database from dependency injection
        
#     Returns:
#         AsyncIOMotorCollection: Audiofiles collection instance
#     """
#     collection_name = os.getenv("MONGODB_AUDIO_COLLECTION", "audiofiles")
#     return database[collection_name]


#"""Versione sincrona per uso outside FastAPI"""
def get_mongo_client() -> AsyncIOMotorClient:
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    return AsyncIOMotorClient(mongo_uri)

def get_mongo_database(client: AsyncIOMotorClient) -> AsyncIOMotorDatabase:
    mongo_db = os.getenv("MONGODB_DB", "music_ai")
    return client[mongo_db]

def get_audiofiles_collection(database: AsyncIOMotorDatabase) -> AsyncIOMotorCollection:
    collection_name = os.getenv("MONGODB_AUDIO_COLLECTION", "audiofiles")
    return database[collection_name]