import logging
import numpy as np
import torch
from estimation.CLAPAudioDataset import float32_to_int16, int16_to_float32
import os
from pathlib import Path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_batch_embeddings(batch, model, device="cuda"):
    """Calcola embeddings per un batch"""
    audios = batch['audio'].to(device)
    texts = batch['text']

    # Preprocess audio
    audio_data = []
    for audio_tensor in audios:
        audio_np = audio_tensor.cpu().numpy()
        audio_np = int16_to_float32(float32_to_int16(audio_np))
        audio_data.append(audio_np)

    audio_data = torch.from_numpy(np.array(audio_data)).float().to(device)

    # Get embeddings
    audio_embeddings = model.get_audio_embedding_from_data(x=audio_data, use_tensor=True)
    text_embeddings = model.get_text_embedding(texts, use_tensor=True)

    return audio_embeddings, text_embeddings


def compute_loss(audio_embeddings, text_embeddings, logit_scale=100):
    """Calcola la contrastive loss"""
    # Normalizza embeddings
    audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=1)
    text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)

    # Similarità
    similarity = logit_scale * audio_embeddings @ text_embeddings.T

    # Contrastive loss
    batch_size = audio_embeddings.shape[0]
    labels = torch.arange(batch_size).to(audio_embeddings.device)

    loss_audio = torch.nn.functional.cross_entropy(similarity, labels)
    loss_text = torch.nn.functional.cross_entropy(similarity.T, labels)
    loss = (loss_audio + loss_text) / 2

    return loss, similarity


def compute_accuracy(similarity):
    """Calcola accuracy da similarity matrix"""
    batch_size = similarity.shape[0]
    labels = torch.arange(batch_size).to(similarity.device)

    audio_to_text_pred = similarity.argmax(dim=1)
    text_to_audio_pred = similarity.argmax(dim=0)

    audio_acc = (audio_to_text_pred == labels).float().mean().item()
    text_acc = (text_to_audio_pred == labels).float().mean().item()
    avg_acc = (audio_acc + text_acc) / 2

    return avg_acc, audio_acc, text_acc


def validation(model, val_dataloader, device="cuda"):
    """Esegue validation sul dataset"""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            # Calcola embeddings
            audio_embeddings, text_embeddings = compute_batch_embeddings(batch, model, device)

            # Calcola loss e accuracy
            loss, similarity = compute_loss(audio_embeddings, text_embeddings)
            accuracy, audio_acc, text_acc = compute_accuracy(similarity)

            # Aggiorna metriche
            total_loss += loss.item()
            total_accuracy += accuracy
            total_batches += 1

            if batch_idx % 5 == 0:
                logger.info(f"Val batch {batch_idx}: Loss={loss.item():.4f}, Acc={accuracy:.4f}")

    # Media delle metriche
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    avg_accuracy = total_accuracy / total_batches if total_batches > 0 else 0

    return avg_loss, avg_accuracy


def train(training_cfg, train_dataloader, val_dataloader, model, optimizer, scheduler=None, device="cuda"):
    """
    Training loop completo con validation

    Args:
        training_cfg: Configurazione training
        train_dataloader: DataLoader per training
        val_dataloader: DataLoader per validation
        model: Modello da addestrare
        optimizer: Optimizer
        scheduler: Scheduler per learning rate (opzionale)
        device: Device per training
    """

    # Statistiche
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(training_cfg["epochs"]):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Epoca {epoch + 1}/{training_cfg['epochs']}")
        logger.info(f"{'=' * 50}")

        # TRAINING PHASE
        model.train()
        epoch_train_loss = 0
        num_train_batches = 0

        for batch_idx, batch in enumerate(train_dataloader):
            # Calcola embeddings
            audio_embeddings, text_embeddings = compute_batch_embeddings(batch, model, device)

            # Calcola loss
            loss, similarity = compute_loss(audio_embeddings, text_embeddings)
            accuracy, _, _ = compute_accuracy(similarity)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Aggiorna metriche
            epoch_train_loss += loss.item()
            num_train_batches += 1

            # Log ogni N batch
            if batch_idx % training_cfg.get("log_interval", 10) == 0:
                logger.info(f"Train - Epoch {epoch + 1}, Batch {batch_idx}: "
                            f"Loss={loss.item():.4f}, Acc={accuracy:.4f}")

        # Calcola loss media training
        avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0
        train_losses.append(avg_train_loss)

        # VALIDATION PHASE
        avg_val_loss, avg_val_accuracy = validation(model, val_dataloader, device)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}")
        logger.info(f"  Val Accuracy: {avg_val_accuracy:.4f}")

        # Learning rate scheduling
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        # Early stopping e checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Salva checkpoint del modello migliore
            checkpoint_dir = Path(f"checkpoints")
            if checkpoint_dir.exists():
                checkpoint_path = f"{checkpoint_dir}/best_model_epoch_{epoch + 1}.pt"
            else:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = f"{checkpoint_dir}/best_model_epoch_{epoch + 1}.pt"

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': avg_val_accuracy,
            }, checkpoint_path)
            logger.info(f"  ✓ Checkpoint salvato: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= training_cfg.get("patience", 10):
                logger.info(f"  ⚠️  Early stopping dopo {patience_counter} epoche senza miglioramenti")
                break

        # Salva checkpoint regolare
        if (epoch + 1) % training_cfg.get("checkpoint_interval", 5) == 0:
            checkpoint_path = f"checkpoints/model_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)

    logger.info("\n" + "=" * 50)
    logger.info("TRAINING COMPLETATO")
    logger.info("=" * 50)

    # Ritorna statistiche
    stats = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss,
        'final_epoch': epoch + 1
    }

    return model, stats

