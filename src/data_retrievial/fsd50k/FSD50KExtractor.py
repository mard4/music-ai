import zipfile
import pandas as pd
from pathlib import Path
import logging 
import io
from typing import List, Optional, Dict, Any
import gridfs
from tqdm import tqdm
import asyncio

from commons.data_models.models import MongoDBConfig
from commons.data_models.audio_models import Metadata, AudioFiles, Sample
from commons.mongodb.mongo_repositories import MongoAudioFilesRepository
from commons.mongodb.mongo_dependecies import get_mongo_client, get_mongo_database, get_audiofiles_collection

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger('pymongo').setLevel(logging.WARNING)

class FSD50KExtractor:
    def __init__(self, data_dir: Path, repository: MongoAudioFilesRepository, fs):
        self.data_dir = data_dir
        self.fs = fs
        self.repository = repository

    async def _initialize_repository(self):
        # Se repository e fs sono già passati nel costruttore, questo metodo potrebbe non servire
        if self.repository and self.fs:
            return True
        return False

    def select_samples(self, categories: List[str], max_samples: int) -> List[Dict[str, Any]]:
        selected_samples = []
        
        dev_csv_path = "FSD50K.ground_truth/dev.csv"
        eval_csv_path = "FSD50K.ground_truth/eval.csv"
        
        try:
            with zipfile.ZipFile(self.data_dir, 'r') as zip_ref:
                if dev_csv_path in zip_ref.namelist():
                    with zip_ref.open(dev_csv_path) as dev_file:
                        dev_df = pd.read_csv(dev_file)
                        for _, row in dev_df.iterrows():
                            if any(category in str(row['labels']) for category in categories):
                                selected_samples.append({
                                    'fname': row['fname'],
                                    'labels': row['labels'],
                                    'split': row.get('split', 'eval'),
                                    'source': 'dev'
                                })
                
                if eval_csv_path in zip_ref.namelist():
                    with zip_ref.open(eval_csv_path) as eval_file:
                        eval_df = pd.read_csv(eval_file)
                        for _, row in eval_df.iterrows():
                            if any(category in str(row['labels']) for category in categories):
                                selected_samples.append({
                                    'fname': row['fname'],
                                    'labels': row['labels'], 
                                    'split': 'eval',
                                    'source': 'eval'
                                })
        except Exception as e:
            logging.error(f"Errore lettura CSV dallo ZIP: {e}")
            return []
        
        if len(selected_samples) > max_samples:
            import random
            selected_samples = random.sample(selected_samples, max_samples)
        
        logging.info(f"Selezionati {len(selected_samples)} campioni")
        return selected_samples

    def create_audio_samples(self, categories: List[str], max_samples: int) -> List[AudioFiles]:
        selected_samples = self.select_samples(categories, max_samples)
        
        if not selected_samples:
            return []
        
        audio_files_list = []
        for sample in selected_samples:
            fname = str(sample['fname'])
            labels = sample['labels']
            original_split = sample['split']
            audio_split = 'dev' if original_split in ['train', 'val'] else 'eval'
            main_category = next((cat for cat in categories if cat in labels), categories[0])
            
            sample_obj = Sample(
                file_name=f"{fname}.wav",
                file_type="wav",
                label=labels,
                source="fsd50k"
            )
            
            metadata_obj = Metadata(
                categories=[main_category],
                split=audio_split,
                original_split=original_split
            )
            
            audio_files = AudioFiles(
                sample=sample_obj,
                metadata=metadata_obj
            )
            
            audio_files_list.append(audio_files)
        
        logging.info(f"Creati {len(audio_files_list)} campioni audio")
        return audio_files_list

    
    async def extract_and_save_audio(self, audio_files_list: List[AudioFiles]) -> bool:
        try:
            # Pulisci GridFS - metodo corretto per GridFSBucket
            cursor = self.fs.find()
            async for gridfs_file in cursor:
                await self.fs.delete(gridfs_file._id)

            extracted_count = 0
            with zipfile.ZipFile(self.data_dir, 'r') as zip_ref:
                zip_contents = set(zip_ref.namelist())
                logging.info(f"Contenuti ZIP caricati, totale file: {len(zip_contents)}")
                
                with tqdm(total=len(audio_files_list), desc="Estrazione audio") as pbar:
                    for audio_files in audio_files_list:
                        file_name = audio_files.sample.file_name
                        split = audio_files.metadata.split

                        internal_path=f"FSD50K.{split}_audio_16k/{file_name}",
                                                
                        try:
                            with zip_ref.open(internal_path) as source_file:
                                audio_data = source_file.read()
                            
                            # Salva in GridFSBucket
                            upload_stream = self.fs.open_upload_stream(
                                file_name,
                                metadata=audio_files.model_dump()
                            )
                            await upload_stream.write(audio_data)
                            await upload_stream.close()
                            
                            file_id = upload_stream._id
                            
                            # Aggiorna il documento con file_id
                            audio_files_dict = audio_files.model_dump()
                            audio_files_dict['gridfs_file_id'] = str(file_id)
                            
                            # Inserisci in MongoDB
                            await self.repository.insert_audio_file(AudioFiles(**audio_files_dict))
                            
                            extracted_count += 1
                            pbar.update(1)
                            
                        except Exception as e:
                            logging.debug(f"Errore estrazione {file_name}: {e}")
                            continue
                        


            logging.info(f"Estratti {extracted_count}/{len(audio_files_list)} file")
            return extracted_count > 0
            
        except Exception as e:
            logging.error(f"Errore estrazione audio: {e}")
            return False
        
    async def full_pipeline(self, categories: List[str], max_samples: int = 100) -> bool:
        """Esegue la pipeline completa"""
        try:
            if not await self._initialize_repository():
                return False
            
            logging.info("Creazione campioni audio...")
            audio_files_list = self.create_audio_samples(categories, max_samples)
            
            if not audio_files_list:
                return False
            
            logging.info("Estrazione e salvataggio audio...")
            success = await self.extract_and_save_audio(audio_files_list)
            
            audio_files = await self.repository.find_audio_by_filter(source="fsd50k")
            logging.info(f"Pipeline completata: {len(audio_files)} campioni salvati")
            
            return success
            
        except Exception as e:
            logging.error(f"Errore pipeline: {e}")
            return False
    
async def create_fsd50k_extractor(data_dir: Path, mongo_config: MongoDBConfig) -> FSD50KExtractor:
    from motor.motor_asyncio import AsyncIOMotorGridFSBucket

    """Factory per creare l'extractor con le dipendenze già iniettate"""
    client = get_mongo_client()
    db = get_mongo_database(client)
    fs = AsyncIOMotorGridFSBucket(db, collection=mongo_config.fs_collection)  # GridFS asincrono

    collection = get_audiofiles_collection(db)
    repository = MongoAudioFilesRepository(collection)
    
    return FSD50KExtractor(data_dir, repository, fs)

async def full_pipeline_mongo_async(data_dir: Path, mongo_config: MongoDBConfig, 
                                   categories: List[str], max_samples: int = 100) -> bool:
    """Versione asincrona"""
    extractor = await create_fsd50k_extractor(data_dir, mongo_config)
    return await extractor.full_pipeline(categories, max_samples)

def full_pipeline_mongo(data_dir: Path, mongo_config: MongoDBConfig, 
                       categories: List[str], max_samples: int = 100) -> bool:
    """Versione sincrona per compatibilità"""
    return asyncio.run(full_pipeline_mongo_async(data_dir, mongo_config, categories, max_samples))