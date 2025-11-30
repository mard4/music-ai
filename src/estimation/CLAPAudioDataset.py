import logging
import pandas as pd
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import io
from bson import ObjectId

from commons.data_models.models import MongoDBConfig
from commons.mongodb.mongo_dependecies import get_mongo_client, get_mongo_database, get_audiofiles_collection
from commons.mongodb.mongo_repositories import MongoAudioFilesRepository
from motor.motor_asyncio import AsyncIOMotorGridFSBucket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLAPAudioDataset(Dataset):
    def __init__(self, repository: MongoAudioFilesRepository, fs: AsyncIOMotorGridFSBucket, db, target_sr=48000, max_duration=10.0):
        self.repository = repository
        self.fs = fs
        self.db = db
        self.target_sr = target_sr
        self.max_duration = max_duration
        self.valid_data = [] 
        self.audio_cache = {} 

    def pre_process_audio(self, audio):
        """Pre-processa l'audio"""
        if audio is None:
            return None
            
        # Normalizzazione
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Troncamento/padding
        target_length = int(self.target_sr * self.max_duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        return audio
    
    async def download_audio_from_gridfs(self, gridfs_id):
        """Scarica file audio da GridFS (ASYNC)"""
        try:
            grid_out = await self.fs.open_download_stream(ObjectId(gridfs_id))
            audio_data = await grid_out.read()
            
            audio, sr = librosa.load(io.BytesIO(audio_data), sr=self.target_sr, duration=self.max_duration)
            return audio
            
        except Exception as e:
            logger.error(f"Errore download {gridfs_id}: {e}")
            return None

    async def load_from_mongodb(self):
        """Carica i file audio da MongoDB"""
        audio_files = await self.repository.find_audio_by_filter()
        
        for audio_file in audio_files:
            if audio_file.gridfs_file_id:
                # Pre-carica l'audio (ASYNC)
                audio_data = await self.download_audio_from_gridfs(audio_file.gridfs_file_id)
                
                if audio_data is not None:
                    # Pre-processing
                    processed_audio = self.pre_process_audio(audio_data)
                    
                    if processed_audio is not None:
                        # Salva in cache
                        self.audio_cache[audio_file.gridfs_file_id] = {
                            'audio': torch.FloatTensor(processed_audio),
                            'text': audio_file.sample.label,
                            'file_name': audio_file.sample.file_name
                        }
                        
                        self.valid_data.append({
                            'gridfs_id': audio_file.gridfs_file_id,
                            'labels': audio_file.sample.label,
                            'file_name': audio_file.sample.file_name
                        })

        logger.info(f"Caricati {len(self.valid_data)} esempi da MongoDB")

    def __len__(self):
        return len(self.valid_data)
    
    def __getitem__(self, idx):
        item = self.valid_data[idx]
        
        cached = self.audio_cache[item['gridfs_id']]
        return {
            'audio': cached['audio'],
            'text': cached['text'],
            'audio_path': cached['file_name']
        }

    async def download_audio_from_gridfs(self, gridfs_id): 
        """Scarica file audio da GridFS"""
        try:
            grid_out = await self.fs.open_download_stream(ObjectId(gridfs_id))  
            audio_data = await grid_out.read() 
            
            audio, sr = librosa.load(io.BytesIO(audio_data), sr=self.target_sr, duration=self.max_duration)
            return audio 
            
        except Exception as e:
            logger.error(f"Errore download {gridfs_id}: {e}")
            return None

async def create_processor_extractor(mongo_config: MongoDBConfig) -> CLAPAudioDataset:
    """Factory per creare il dataset da MongoDB"""
    client = get_mongo_client() 
    db = get_mongo_database(client) 
    fs = AsyncIOMotorGridFSBucket(db, bucket_name=mongo_config.fs_collection)
    collection = get_audiofiles_collection(db)  
    repository = MongoAudioFilesRepository(collection)
    
    dataset = CLAPAudioDataset(repository, fs, db)
    await dataset.load_from_mongodb()  # Pre-carica tutto async
    
    return dataset

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

