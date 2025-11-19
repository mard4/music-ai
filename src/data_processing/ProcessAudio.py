import logging
import os
import tempfile
from pathlib import Path
from tqdm import tqdm
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorGridFSBucket

from commons.data_models.models import MongoDBConfig
from commons.data_models.audio_models import EnrichedAudioFile
from commons.mongodb.mongo_dependecies import get_mongo_client, get_mongo_database, get_audiofiles_collection
from commons.mongodb.mongo_repositories import MongoAudioFilesRepository
from data_processing.AudioFeatureExtractor import AudioFeatureExtractor

class ProcessAudio:
    def __init__(self, repository: MongoAudioFilesRepository, fs, db):
        self.fs = fs
        self.repository = repository
        self.db = db

    async def process_audio_files_from_mongo(self):
        """
        Process audio files from MongoDB, extract features, and update metadata in a separate collection.
        """
        extractor = AudioFeatureExtractor()
        
        # Fetch all audio files
        logging.info("Fetching audio files from MongoDB...")
        audio_files = await self.repository.find_audio_by_filter() 
        
        logging.info(f"Found {len(audio_files)} files to process.")
        
        # Get the enriched audio collection
        enriched_collection = self.db.enriched_audio_samples
        
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            try:
                if not audio_file.gridfs_file_id:
                    logging.warning(f"No GridFS ID for file: {audio_file.sample.file_name}")
                    continue
                    
                # Create temp file
                with tempfile.NamedTemporaryFile(suffix=Path(audio_file.sample.file_name).suffix, delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                
                try:
                    # Download from GridFS
                    grid_out = await self.fs.open_download_stream(ObjectId(audio_file.gridfs_file_id))
                    with open(temp_path, "wb") as f:
                        f.write(await grid_out.read())
                    
                    # Extract features
                    features = extractor.extract_audio_features(temp_path)
                    
                    if features:
                        technical_desc = extractor.generate_technical_description(features)
                        semantic_desc = extractor.generate_semantic_description(features)
                        text_descriptions = extractor.create_descriptions(temp_path)
                        
                        # Create EnrichedAudioFile object
                        enriched_audio = EnrichedAudioFile(
                            sample=audio_file.sample,
                            metadata=audio_file.metadata,
                            gridfs_file_id=audio_file.gridfs_file_id,
                            text_description=text_descriptions,
                            technical_description=technical_desc,
                            semantic_description=semantic_desc,
                            spectral_centroid=features.get('spectral_centroid'),
                            rms=features.get('rms'),
                            duration=features.get('duration')
                        )
                        
                        # Convert to dictionary for MongoDB
                        enriched_data = enriched_audio.model_dump(exclude_none=True)
                        
                        # Insert into enriched collection (upsert to avoid duplicates)
                        await enriched_collection.update_one(
                            {"gridfs_file_id": audio_file.gridfs_file_id},
                            {"$set": enriched_data},
                            upsert=True
                        )
                        
                        logging.info(f"Saved enriched data for: {audio_file.sample.file_name}")
                        
                    else:
                        logging.warning(f"Could not extract features for: {audio_file.sample.file_name}")

                except Exception as e:
                    logging.error(f"Error processing {audio_file.sample.file_name}: {e}")
                finally:
                    # Cleanup temp file
                    if temp_path.exists():
                        os.unlink(temp_path)
                        
            except Exception as e:
                logging.error(f"General error for file: {e}")

async def create_processor_extractor(mongo_config: MongoDBConfig) -> ProcessAudio:
    """Factory per creare l'extractor con le dipendenze già iniettate"""
    client = get_mongo_client()
    db = get_mongo_database(client=client)
    fs = AsyncIOMotorGridFSBucket(db, bucket_name=mongo_config.fs_collection)

    collection = get_audiofiles_collection(db)
    repository = MongoAudioFilesRepository(collection)
    
    return ProcessAudio(repository, fs, db)
