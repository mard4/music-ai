import asyncio
import logging
import pandas as pd
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import io
from bson import ObjectId
import os
from torch.utils.data import Dataset, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from commons.data_models.models import MongoDBConfig
from commons.mongodb.mongo_dependecies import get_mongo_client, get_mongo_database, get_audiofiles_collection
from commons.mongodb.mongo_repositories import MongoAudioFilesRepository
from motor.motor_asyncio import AsyncIOMotorGridFSBucket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLAPAudioDataset(Dataset):
    def __init__(self, repository: MongoAudioFilesRepository, fs: AsyncIOMotorGridFSBucket, db, target_sr=48000, max_duration=10.0, use_clean_labels=True):
        self.repository = repository
        self.fs = fs
        self.db = db
        self.target_sr = target_sr
        self.max_duration = max_duration
        self.use_clean_labels = use_clean_labels
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

    async def load_clean_labels(self):
        """Carica i clean labels dalla collection separata"""
        clean_labels = {}
        if self.use_clean_labels:
            try:
                clean_labels_collection = self.db.get_collection("clean_audio_labels")
                clean_labels_cursor = clean_labels_collection.find({})
                async for doc in clean_labels_cursor:
                    clean_labels[doc['gridfs_file_id']] = doc.get('label', '')
                logger.info(f"Caricati {len(clean_labels)} clean labels")
            except Exception as e:
                logger.warning(f"Impossibile caricare clean labels: {e}")
        return clean_labels

    async def load_from_mongodb(self):
        """Carica i file audio da MongoDB"""
        # Carica clean labels
        clean_labels = await self.load_clean_labels()

        # Carica file audio
        audio_files = await self.repository.find_audio_by_filter()

        for audio_file in audio_files:
            if audio_file.gridfs_file_id:
                audio_data = await self.download_audio_from_gridfs(audio_file.gridfs_file_id)

                if audio_data is not None:
                    processed_audio = self.pre_process_audio(audio_data)

                    if processed_audio is not None:
                        # Scegli il label: clean se disponibile, altrimenti originale
                        gridfs_id = audio_file.gridfs_file_id
                        text_label = clean_labels.get(gridfs_id, audio_file.sample.label)

                        self.audio_cache[gridfs_id] = {
                            'audio': torch.FloatTensor(processed_audio),
                            'text': text_label
                        }

                        self.valid_data.append({
                            'gridfs_id': gridfs_id,
                            'labels': text_label
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


def get_clap_dataset() -> CLAPAudioDataset:
    logger.info("Creazione dataset CLAP...")
    mongo_config = MongoDBConfig(
        connection_string=os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017"),
        database_name=os.getenv("MONGODB_DATABASE_NAME", "audio_db"),
        audio_collection=os.getenv("MONGODB_AUDIO_COLLECTION", "audio_samples"),
        fs_collection=os.getenv("MONGODB_FS_COLLECTION", "fs_audio_files")
    )

    return asyncio.run(create_processor_extractor(mongo_config))

def stratify_category(dataset):
    ## la categoria é in metadati --> main_category
    try:
        # Prova a estrarre la categoria principale dal label
        categories = []
        for item in dataset.valid_data:
            label = item['labels'].lower()
            # Cerca categorie comuni nei label
            if 'bass' in label:
                categories.append('bass')
            elif 'kick' in label:
                categories.append('kick')
            elif 'drum' in label or 'drums' in label:
                categories.append('drums')
            elif 'synth' in label:
                categories.append('synth')
            elif 'pad' in label:
                categories.append('pad')
            elif 'vocal' in label:
                categories.append('vocal')
            else:
                categories.append('other')

        if len(set(categories)) > 1:  # Se abbiamo abbastanza categorie diverse
            stratify = categories
            return stratify
        else:
            return None
    except:
        return None

def split_dataset(dataset: CLAPAudioDataset,
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  random_seed: int = 42,
                  stratify_by_category: bool = True):
    """
    Divide il dataset in train, validation e test sets

    Args:
        dataset: Dataset completo da suddividere
        train_ratio: Percentuale per training
        val_ratio: Percentuale per validation
        test_ratio: Percentuale per test
        random_seed: Seed per riproducibilità
        stratify_by_category: Se True, stratifica per categoria

    Returns:
        train_dataset, val_dataset, test_dataset: Sottodataset
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, \
        "Le percentuali devono sommare a 1.0"

    n_samples = len(dataset)
    indices = np.arange(n_samples)

    # Estrai categorie per stratificazione (se richiesto)
    if stratify_by_category and len(dataset.valid_data) > 0:
        stratify = stratify_category(dataset)
    else:
        stratify = None

    # Prima split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=val_ratio + test_ratio,
        random_state=random_seed,
        stratify=stratify if stratify else None
    )

    # Secondo split: val vs test
    if stratify:
        # Estrai categorie per il subset temporaneo
        temp_categories = [stratify[i] for i in temp_indices]
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=random_seed,
            stratify=temp_categories
        )
    else:
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=random_seed
        )

    # Crea i subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    logger.info(f"Dataset split completato:")
    logger.info(f"  Training samples: {len(train_dataset)} ({len(train_dataset) / n_samples:.1%})")
    logger.info(f"  Validation samples: {len(val_dataset)} ({len(val_dataset) / n_samples:.1%})")
    logger.info(f"  Test samples: {len(test_dataset)} ({len(test_dataset) / n_samples:.1%})")

    return train_dataset, val_dataset, test_dataset


def get_train_val_test_datasets(train_ratio: float = 0.7,
                                val_ratio: float = 0.15,
                                test_ratio: float = 0.15,
                                random_seed: int = 42) -> tuple:
    """
    Funzione completa che carica il dataset e lo divide

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Carica dataset completo
    logger.info("Caricamento dataset completo...")
    full_dataset = get_clap_dataset()

    # Applica split
    logger.info("Applicazione split train/val/test...")
    train_ds, val_ds, test_ds = split_dataset(
        full_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        stratify_by_category=False
    )

    # Verifica distribuzione
    def analyze_split(subset, name):
        if len(subset) == 0:
            return

        # Estrai alcuni label per vedere la distribuzione
        labels = []
        for i in range(min(50, len(subset))):
            try:
                item = subset[i]
                # Gestisci sia Subset che Dataset diretto
                if isinstance(item, dict):
                    labels.append(item['text'])
                else:
                    labels.append(item['text'])
            except:
                continue

        logger.info(f"\n{name} split analysis (primi {len(labels)} campioni):")
        if labels:
            logger.info(f"  Sample labels: {labels[:3]}...")
            # Conta categorie principali
            category_counts = {}
            for label in labels:
                label_lower = label.lower()
                if 'bass' in label_lower:
                    category_counts['bass'] = category_counts.get('bass', 0) + 1
                elif 'kick' in label_lower:
                    category_counts['kick'] = category_counts.get('kick', 0) + 1
                elif 'drum' in label_lower or 'drums' in label_lower:
                    category_counts['drums'] = category_counts.get('drums', 0) + 1
                elif 'synth' in label_lower:
                    category_counts['synth'] = category_counts.get('synth', 0) + 1
                elif 'pad' in label_lower:
                    category_counts['pad'] = category_counts.get('pad', 0) + 1
                elif 'vocal' in label_lower:
                    category_counts['vocal'] = category_counts.get('vocal', 0) + 1
                else:
                    category_counts['other'] = category_counts.get('other', 0) + 1

            logger.info(f"  Category distribution: {category_counts}")

    analyze_split(train_ds, "Training")
    analyze_split(val_ds, "Validation")
    analyze_split(test_ds, "Test")

    return train_ds, val_ds, test_ds

def create_dataloaders(batch_size: int = 16,
                       num_workers: int = 0,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15):
    """
    Crea DataLoader per training, validation e test
    """
    from torch.utils.data import DataLoader

    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Per batch size consistente
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(f"Dataloader creati:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader

