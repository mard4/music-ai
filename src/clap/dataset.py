"""
Dataset CLAP per audio e testo.
Gestisce il caricamento da MongoDB, preprocessing e split dei dati.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split
from bson import ObjectId
import io

from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from core.domain.audio import AudioFile
from config.settings import settings
from core.infrastructure.database.dependecies import get_mongo_database, get_mongo_client, get_gridfs_bucket, \
    get_audio_repository
from core.infrastructure.database.repositories import AudioFilesRepository
logger = logging.getLogger(__name__)


@dataclass
class AudioSample:
    """Campione audio processato."""
    audio_tensor: torch.Tensor
    text_label: str
    gridfs_id: str
    metadata: Dict[str, Any]


class CLAPAudioDataset(Dataset):
    """Dataset per training CLAP con audio da MongoDB GridFS."""

    def __init__(
            self,
            repository: AudioFilesRepository,
            fs_bucket: AsyncIOMotorGridFSBucket,
            db,
            target_sample_rate: int = None,
            max_duration: float = None,
            use_clean_labels: bool = True,
            enable_caching: bool = True,
            load_async: bool = False

    ):
        self.repository = repository
        self.fs_bucket = fs_bucket
        self.db = db
        self.target_sample_rate = target_sample_rate or settings.audio.target_sample_rate
        self.max_duration = max_duration or settings.audio.max_duration_seconds
        self.use_clean_labels = use_clean_labels
        self.enable_caching = enable_caching

        self.samples: List[AudioSample] = []
        self.audio_cache: Dict[str, torch.Tensor] = {}
        self._clean_labels_cache: Dict[str, str] = {}

    async def load_data(self):
        """Carica dati da MongoDB in modo asincrono."""
        logger.info("Caricamento dati da MongoDB...")

        # Carica clean labels se necessario
        if self.use_clean_labels:
            await self._load_clean_labels()

        # Carica file audio
        audio_files = await self.repository.find_all()

        for audio_file in audio_files:
            if audio_file.gridfs_file_id:
                await self._process_audio_file(audio_file)

        logger.info(f"Caricati {len(self.samples)} campioni validi")

    async def _async_load_data(self):
        """Carica dati da MongoDB in modo asincrono."""
        logger.info("Caricamento dati da MongoDB...")

        # Carica clean labels se necessario
        if self.use_clean_labels:
            await self._load_clean_labels()

        # Carica file audio
        audio_files = await self.repository.find_all()

        for audio_file in audio_files:
            if audio_file.gridfs_file_id:
                await self._process_audio_file(audio_file)

        logger.info(f"Caricati {len(self.samples)} campioni validi")

    async def _load_clean_labels(self):
        """Carica clean labels dalla collection dedicata."""
        try:
            collection = self.db[settings.database.mongodb_clean_labels_collection]
            cursor = collection.find({})

            async for doc in cursor:
                gridfs_id = doc.get('gridfs_file_id')
                label = doc.get('label', '')
                if gridfs_id:
                    self._clean_labels_cache[gridfs_id] = label

            logger.info(f"Caricati {len(self._clean_labels_cache)} clean labels")
        except Exception as e:
            logger.warning(f"Impossibile caricare clean labels: {e}")

    async def _process_audio_file(self, audio_file: AudioFile):
        """Processa un singolo file audio."""
        try:
            # Scarica audio da GridFS
            audio_data = await self._download_from_gridfs(audio_file.gridfs_file_id)
            if audio_data is None:
                return

            # Preprocess audio
            processed_audio = self._preprocess_audio(audio_data)
            if processed_audio is None:
                return

            # Ottieni label (clean se disponibile, altrimenti originale)
            text_label = self._get_text_label(
                audio_file.gridfs_file_id,
                audio_file.sample.label
            )

            # Crea campione
            sample = AudioSample(
                audio_tensor=torch.FloatTensor(processed_audio),
                text_label=text_label,
                gridfs_id=audio_file.gridfs_file_id,
                metadata={
                    'file_name': audio_file.sample.file_name,
                    'source': audio_file.sample.source,
                    'categories': audio_file.metadata.categories
                }
            )

            self.samples.append(sample)

            # Cache se abilitato
            if self.enable_caching:
                self.audio_cache[audio_file.gridfs_file_id] = sample.audio_tensor

        except Exception as e:
            logger.error(f"Errore processando file {audio_file.sample.file_name}: {e}")

    async def _download_from_gridfs(self, gridfs_id: str) -> Optional[np.ndarray]:
        """Scarica file audio da GridFS."""
        try:
            grid_out = await self.fs_bucket.open_download_stream(ObjectId(gridfs_id))
            audio_bytes = await grid_out.read()

            audio, _ = librosa.load(
                io.BytesIO(audio_bytes),
                sr=self.target_sample_rate,
                duration=self.max_duration
            )
            return audio

        except Exception as e:
            logger.error(f"Errore download GridFS {gridfs_id}: {e}")
            return None

    def _preprocess_audio(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Preprocessa l'audio per il training."""
        if audio is None or len(audio) == 0:
            return None

        # Normalizzazione
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Troncamento/Padding
        target_length = int(self.target_sample_rate * self.max_duration)
        if len(audio) < target_length:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        else:
            audio = audio[:target_length]

        return audio

    def _get_text_label(self, gridfs_id: str, original_label: str) -> str:
        """Ottiene il label testuale (clean o originale)."""
        if self.use_clean_labels and gridfs_id in self._clean_labels_cache:
            return self._clean_labels_cache[gridfs_id]
        return original_label

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Restituisce un campione del dataset."""
        sample = self.samples[idx]

        return {
            'audio': sample.audio_tensor,
            'text': sample.text_label,
            'metadata': sample.metadata
        }

    def get_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche del dataset."""
        if not self.samples:
            return {}

        labels = [s.text_label for s in self.samples]
        unique_labels = set(labels)

        return {
            'total_samples': len(self),
            'unique_labels': len(unique_labels),
            'label_distribution': {
                label: labels.count(label) for label in unique_labels
            }
        }


# Funzioni di supporto per audio processing
def float32_to_int16(x: np.ndarray) -> np.ndarray:
    """Converte float32 in int16."""
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def int16_to_float32(x: np.ndarray) -> np.ndarray:
    """Converte int16 in float32."""
    return (x / 32767.0).astype(np.float32)


# Funzioni per split dataset
def split_dataset(
        dataset: CLAPAudioDataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        stratify: bool = True
) -> Tuple[Subset, Subset, Subset]:
    """
    Divide il dataset in train, validation e test.

    Args:
        dataset: Dataset completo
        train_ratio: Percentuale training
        val_ratio: Percentuale validation
        test_ratio: Percentuale test
        random_seed: Seed per riproducibilità
        stratify: Se usare stratificazione per categorie

    Returns:
        Tuple di (train_set, val_set, test_set)
    """
    n_samples = len(dataset)
    if n_samples == 0:
        raise ValueError(f"Dataset is empty! Cannot split 0 samples.")

    # Validazione ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-10:
        raise ValueError(f"Le ratio devono sommare a 1.0, ottenuto: {total_ratio}")

    n_samples = len(dataset)
    indices = np.arange(n_samples)

    # Prepara stratificazione se richiesta
    stratify_labels = None
    if stratify and dataset.samples:
        try:
            stratify_labels = [
                _extract_category(sample.text_label)
                for sample in dataset.samples
            ]
        except Exception:
            stratify_labels = None

    # Prima split: train vs (val + test)
    test_size = val_ratio + test_ratio
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_labels
    )

    # Secondo split: val vs test
    if stratify and stratify_labels:
        temp_labels = [stratify_labels[i] for i in temp_indices]
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=test_ratio / test_size,
            random_state=random_seed,
            stratify=temp_labels
        )
    else:
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=test_ratio / test_size,
            random_state=random_seed
        )

    # Crea subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    logger.info(f"Dataset split completato:")
    logger.info(f"  Training: {len(train_dataset)} samples ({len(train_dataset) / n_samples:.1%})")
    logger.info(f"  Validation: {len(val_dataset)} samples ({len(val_dataset) / n_samples:.1%})")
    logger.info(f"  Test: {len(test_dataset)} samples ({len(test_dataset) / n_samples:.1%})")

    return train_dataset, val_dataset, test_dataset


def _extract_category(label: str) -> str:
    """Estrae categoria da un label testuale.
    stratify should be based on main category in metadati
    """
    label_lower = label.lower()

    category_keywords = {
        'bass': ['bass'],
        'kick': ['kick'],
        'drums': ['drum', 'drums'],
        'synth': ['synth'],
        'pad': ['pad'],
        'vocal': ['vocal', 'voice', 'singing'],
        'guitar': ['guitar'],
        'piano': ['piano', 'keys'],
        'fx': ['fx', 'effect', 'sweep', 'impact'],
        'ambient': ['ambient', 'atmosphere', 'texture']
    }

    for category, keywords in category_keywords.items():
        if any(keyword in label_lower for keyword in keywords):
            return category

    return 'other'


async def create_clap_dataset(
        use_clean_labels: bool = True,
        target_sample_rate: int = None,
        max_duration: float = None
) -> CLAPAudioDataset:
    """
    Factory per creare dataset CLAP.

    Args:
        use_clean_labels: Usa clean labels se disponibili
        target_sample_rate: Sample rate target
        max_duration: Durata massima in secondi

    Returns:
        CLAPAudioDataset configurato
    """

    client = get_mongo_client()
    db = get_mongo_database(client)
    fs_bucket = get_gridfs_bucket(db)
    repository = get_audio_repository()

    logger.info(f"Creating dataset {repository}, bucket {fs_bucket}, db {db}, client {client}")

    dataset = CLAPAudioDataset(
        repository=repository,
        fs_bucket=fs_bucket,
        db=db,
        target_sample_rate=target_sample_rate,
        max_duration=max_duration,
        use_clean_labels=use_clean_labels
    )

    # Carica i dati asincronamente
    await dataset.load_data()

    return dataset


def get_clap_dataset_sync() -> CLAPAudioDataset:
    """Versione sincrona per ottenere il dataset."""
    return asyncio.run(create_clap_dataset())

async def get_clap_dataset() -> CLAPAudioDataset:
    """Versione sincrona per ottenere il dataset."""
    return await create_clap_dataset()

async def create_dataloaders(
        dataset: CLAPAudioDataset = None,
        batch_size: int = None,
        num_workers: int = None,
        train_ratio: float = None,
        val_ratio: float = None,
        test_ratio: float = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea dataloaders per training, validation e test.

    Args:
        dataset: Dataset opzionale (creato se None)
        batch_size: Dimensione batch
        num_workers: Worker per data loading
        train_ratio: Ratio training
        val_ratio: Ratio validation
        test_ratio: Ratio test

    Returns:
        Tuple di (train_loader, val_loader, test_loader)
    """
    # Usa default dalle settings se non specificato
    batch_size = batch_size or settings.training.batch_size
    num_workers = num_workers or settings.training.num_workers
    train_ratio = train_ratio or settings.training.train_ratio
    val_ratio = val_ratio or settings.training.val_ratio
    test_ratio = test_ratio or settings.training.test_ratio

    # Crea dataset se non fornito
    if dataset is None:
        dataset = await get_clap_dataset()

    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify=False
    )

    # Crea dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_fn
    )

    logger.info(f"Dataloaders creati:")
    logger.info(f"  Training batches: {len(train_loader)}")
    logger.info(f"  Validation batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


def _collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Funzione di collate per il dataloader."""
    return {
        'audio': torch.stack([item['audio'] for item in batch]),
        'text': [item['text'] for item in batch],
        'metadata': [item['metadata'] for item in batch]
    }