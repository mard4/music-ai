"""
Configurazioni specifiche per CLAP.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field
from config.settings import settings

#TODO add stratify

class OptimizerType(str, Enum):
    """Tipi di optimizer supportati."""
    ADAM = "Adam"
    ADAMW = "AdamW"
    SGD = "SGD"
    RMSPROP = "RMSprop"


class SchedulerType(str, Enum):
    """Tipi di scheduler supportati."""
    COSINE = "CosineAnnealingLR"
    PLATEAU = "ReduceLROnPlateau"
    STEP = "StepLR"
    MULTISTEP = "MultiStepLR"


@dataclass
class TrainingConfig:
    """Configurazione per il training CLAP."""

    # Hyperparameters
    learning_rate: float = settings.training.learning_rate
    weight_decay: float = settings.training.weight_decay
    batch_size: int = settings.training.batch_size
    epochs: int = settings.training.epochs
    num_workers: int = settings.training.num_workers

    # Optimizer
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    # Scheduler
    scheduler_type: SchedulerType = SchedulerType.COSINE
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        'T_max': 1000,
        'eta_min': 1e-6
    })

    # Training controls
    patience: int = settings.training.patience
    log_interval: int = settings.training.log_interval
    checkpoint_interval: int = settings.training.checkpoint_interval
    max_grad_norm: float = 1.0

    # CLAP specific
    logit_scale: float = settings.clap.logit_scale
    enable_fusion: bool = settings.clap.enable_fusion

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"

    # Data
    train_ratio: float = settings.training.train_ratio
    val_ratio: float = settings.training.val_ratio
    test_ratio: float = settings.training.test_ratio
    random_seed: int = 42

    def get_optimizer_kwargs(self) -> Dict[str, Any]:
        """Restituisce kwargs per l'optimizer."""
        return {
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay,
            'betas': (self.beta1, self.beta2),
            'eps': self.epsilon
        }

    def to_dict(self) -> Dict[str, Any]:
        """Converte in dizionario."""
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.__dict__.items()
        }


@dataclass
class AudioProcessingConfig:
    """Configurazione per processing audio."""

    target_sample_rate: int = settings.audio.target_sample_rate
    max_duration_seconds: float = settings.audio.max_duration_seconds
    normalize_audio: bool = True
    normalize_db: bool = False
    target_db: float = -20.0

    # Augmentation
    enable_augmentation: bool = False
    augmentation_params: Dict[str, Any] = field(default_factory=lambda: {
        'time_stretch_range': (0.8, 1.2),
        'pitch_shift_range': (-2, 2),
        'noise_level': 0.005,
        'apply_random_gain': True,
        'gain_range': (-6, 6)
    })


@dataclass
class ModelConfig:
    """Configurazione per il modello CLAP."""

    # Model architecture
    model_name: str = "CLAP_Module"
    enable_fusion: bool = settings.clap.enable_fusion
    pretrained: bool = True
    freeze_backbone: bool = False

    # Embedding dimensions
    audio_embedding_dim: int = 512
    text_embedding_dim: int = 512
    projection_dim: int = 512

    # Transformer parameters
    num_heads: int = 8
    num_layers: int = 12
    dropout: float = 0.1

    # Feature extraction
    use_mel_spectrogram: bool = True
    n_mels: int = 64
    n_fft: int = 2048
    hop_length: int = 512


@dataclass
class EvaluationConfig:
    """Configurazione per valutazione."""

    # Retrieval metrics
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    compute_map: bool = True
    compute_mrr: bool = True

    # Similarity thresholds
    similarity_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9])

    # Visualization
    save_embeddings: bool = True
    plot_confusion_matrix: bool = True
    plot_roc_curve: bool = False


class CLAPConfig(BaseModel):
    """Configurazione completa CLAP."""

    training: TrainingConfig = Field(default_factory=TrainingConfig)
    audio_processing: AudioProcessingConfig = Field(default_factory=AudioProcessingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    # Environment
    device: str = settings.clap.device
    num_workers: int = settings.training.num_workers
    mixed_precision: bool = True

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'CLAPConfig':
        """Carica configurazione da file YAML."""
        import yaml

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def save(self, filepath: str):
        """Salva configurazione su file."""
        import yaml

        config_dict = self.dict()
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)