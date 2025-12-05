"""
Training loop per modello CLAP.
Gestisce training, validation e salvataggio checkpoint.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import wandb  # Opzionale per logging

from .dataset import float32_to_int16, int16_to_float32
from .config import TrainingConfig
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metriche di training."""
    train_loss: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    epoch: int
    timestamp: datetime


class CLAPTrainer:
    """Trainer per modello CLAP."""

    def __init__(
            self,
            model,
            optimizer: Optimizer,
            train_loader: DataLoader,
            val_loader: DataLoader,
            config: TrainingConfig = None,
            device: Any = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        self.device = device or settings.clap.device

        # Setup logging
        self.log_dir = settings.checkpoint_dir
        self.log_dir.mkdir(exist_ok=True)

        # Stato training
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Metriche storage
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []

        # Move model to device
        self.model.to(self.device)

    def train_epoch(self) -> Dict[str, float]:
        """Esegue un'epoca di training."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Forward pass
            loss, metrics = self._training_step(batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()

            # Aggiorna metriche
            epoch_loss += loss.item()
            num_batches += 1

            # Log batch progress
            if batch_idx % self.config.log_interval == 0:
                self._log_batch_progress(batch_idx, loss.item(), metrics['accuracy'])

        # Calcola metriche epoca
        avg_loss = epoch_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)

        return {'train_loss': avg_loss}

    def _training_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Esegue un singolo step di training."""
        # Calcola embeddings
        audio_embeddings, text_embeddings = compute_embeddings(
            self.model, batch, self.device
        )

        # Calcola loss
        loss, similarity = compute_contrastive_loss(
            audio_embeddings, text_embeddings, self.config.logit_scale
        )

        # Calcola accuracy
        accuracy = compute_accuracy(similarity)

        return loss, {'accuracy': accuracy}

    def validate(self) -> Dict[str, float]:
        """Esegue validation."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Forward pass
                audio_embeddings, text_embeddings = compute_embeddings(
                    self.model, batch, self.device
                )

                # Calcola loss e accuracy
                loss, similarity = compute_contrastive_loss(
                    audio_embeddings, text_embeddings, self.config.logit_scale
                )
                accuracy = compute_accuracy(similarity)

                # Aggiorna metriche
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1

        # Calcola medie
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_accuracy)

        return {
            'val_loss': avg_loss,
            'val_accuracy': avg_accuracy
        }

    def run_training(self) -> Dict[str, Any]:
        """Esegue il training completo."""
        logger.info(f"Inizio training per {self.config.epochs} epoche")

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch + 1

            logger.info(f"\n{'=' * 50}")
            logger.info(f"Epoca {self.current_epoch}/{self.config.epochs}")
            logger.info(f"{'=' * 50}")

            # Training
            train_metrics = self.train_epoch()

            # Validation
            val_metrics = self.validate()

            # Log risultati epoca
            self._log_epoch_results(train_metrics, val_metrics)

            # Checkpoint e early stopping
            should_stop = self._check_early_stopping(val_metrics['val_loss'])
            if should_stop:
                logger.info("Early stopping attivato")
                break

        logger.info("\nTraining completato")
        return self._get_training_stats()

    def _check_early_stopping(self, val_loss: float) -> bool:
        """Gestisce early stopping e salvataggio checkpoint."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self._save_checkpoint(is_best=True)
            logger.info(f"  ✓ Nuovo best model (loss: {val_loss:.4f})")
        else:
            self.patience_counter += 1
            logger.info(f"  ⚠️  Nessun miglioramento ({self.patience_counter}/{self.config.patience})")

        # Salva checkpoint regolare
        if self.current_epoch % self.config.checkpoint_interval == 0:
            self._save_checkpoint(is_best=False)

        return self.patience_counter >= self.config.patience

    def _save_checkpoint(self, is_best: bool = False):
        """Salva checkpoint del modello."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }

        if is_best:
            filename = self.log_dir / "best_model.pt"
        else:
            filename = self.log_dir / f"model_epoch_{self.current_epoch}.pt"

        torch.save(checkpoint, filename)
        logger.info(f"  Checkpoint salvato: {filename}")

    def _log_batch_progress(self, batch_idx: int, loss: float, accuracy: float):
        """Log progresso batch."""
        logger.info(
            f"  Batch {batch_idx}/{len(self.train_loader)} - "
            f"Loss: {loss:.4f}, Acc: {accuracy:.4f}"
        )

    def _log_epoch_results(self, train_metrics: Dict, val_metrics: Dict):
        """Log risultati epoca."""
        logger.info(f"\nEpoca {self.current_epoch} risultati:")
        logger.info(f"  Training Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"  Validation Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"  Validation Accuracy: {val_metrics['val_accuracy']:.4f}")
        logger.info(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

    def _get_training_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche finali del training."""
        return {
            'final_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'final_val_accuracy': self.val_accuracies[-1] if self.val_accuracies else 0,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'total_training_time': None  # Aggiungere tracking tempo se necessario
        }


# Funzioni di supporto
def compute_embeddings(model, batch: Dict[str, Any], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calcola embeddings audio e testo per un batch."""
    audio_tensors = batch['audio'].to(device)
    texts = batch['text']

    # Preprocess audio
    audio_data = []
    for audio_tensor in audio_tensors:
        audio_np = audio_tensor.cpu().numpy()
        audio_np = int16_to_float32(float32_to_int16(audio_np))
        audio_data.append(audio_np)

    audio_data = torch.from_numpy(np.array(audio_data)).float().to(device)

    # Get embeddings
    audio_embeddings = model.get_audio_embedding_from_data(x=audio_data, use_tensor=True)
    text_embeddings = model.get_text_embedding(texts, use_tensor=True)

    return audio_embeddings, text_embeddings


def compute_contrastive_loss(
        audio_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        logit_scale: float = 100.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calcola contrastive loss per embeddings."""
    # Normalizza embeddings
    audio_embeddings = F.normalize(audio_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

    # Calcola similarità
    similarity = logit_scale * (audio_embeddings @ text_embeddings.T)

    # Contrastive loss
    batch_size = similarity.shape[0]
    labels = torch.arange(batch_size, device=similarity.device)

    loss_audio = F.cross_entropy(similarity, labels)
    loss_text = F.cross_entropy(similarity.T, labels)
    loss = (loss_audio + loss_text) / 2.0

    return loss, similarity


def compute_accuracy(similarity: torch.Tensor) -> float:
    """Calcola accuracy da similarity matrix."""
    batch_size = similarity.shape[0]
    labels = torch.arange(batch_size, device=similarity.device)

    audio_to_text_pred = similarity.argmax(dim=1)
    text_to_audio_pred = similarity.argmax(dim=0)

    audio_acc = (audio_to_text_pred == labels).float().mean().item()
    text_acc = (text_to_audio_pred == labels).float().mean().item()
    avg_acc = (audio_acc + text_acc) / 2.0

    return avg_acc


# Funzioni di interfaccia per backward compatibility
def train(
        training_cfg: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        model,
        optimizer,
        scheduler=None,
        device="cuda"
) -> Tuple[Any, Dict[str, Any]]:
    """
    Funzione di training compatibile con interfaccia originale.

    Args:
        training_cfg: Configurazione training
        train_dataloader: Dataloader training
        val_dataloader: Dataloader validation
        model: Modello CLAP
        optimizer: Optimizer
        scheduler: Scheduler LR (opzionale)
        device: Device

    Returns:
        Modello addestrato e statistiche
    """
    # Converti dict in TrainingConfig
    config = TrainingConfig(**training_cfg)

    # Crea trainer
    trainer = CLAPTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        config=config,
        device=device
    )

    # Esegui training
    stats = trainer.run_training()

    return trainer.model, stats


def validate(model, val_dataloader: DataLoader, device: str = "cuda") -> Tuple[float, float]:
    """Esegue validation."""
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            audio_embeddings, text_embeddings = compute_embeddings(model, batch, device)
            loss, similarity = compute_contrastive_loss(audio_embeddings, text_embeddings)
            accuracy = compute_accuracy(similarity)

            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_accuracy = total_accuracy / max(num_batches, 1)

    return avg_loss, avg_accuracy