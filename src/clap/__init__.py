"""
Modulo CLAP - Contrastive Language-Audio Pretraining
Gestione completa di training, testing e dataset per modelli audio-testo.
"""
from .dataset import CLAPAudioDataset, get_clap_dataset, get_clap_dataset_sync, create_dataloaders, create_clap_dataset
from .training import train, validate, compute_embeddings
from .testing import test_model, evaluate_single_example, compute_retrieval_metrics
from .model_handler import CLAPModelHandler
from .config import CLAPConfig, TrainingConfig

__version__ = "1.0.0"
__all__ = [
    "CLAPAudioDataset",
    "get_clap_dataset",
    "get_clap_dataset_sync",
    "create_clap_dataset",
    "create_dataloaders",
    "train",
    "validate",
    "compute_embeddings",
    "test_model",
    "evaluate_single_example",
    "compute_retrieval_metrics",
    "CLAPModelHandler",
    "CLAPConfig",
    "TrainingConfig",
]