from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
load_dotenv()

class MongoDBConfig(BaseModel):
    connection_string: str = Field(os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017/"))
    database_name: str = Field(os.getenv("MONGODB_DATABASE_NAME", "fsd50k_db"))
    audio_collection: str = Field(os.getenv("MONGODB_AUDIO_COLLECTION", "audio_samples"))
    fs_collection: str = Field(os.getenv("MONGODB_FS_COLLECTION", "audio_files"))