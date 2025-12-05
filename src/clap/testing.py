"""
Testing e valutazione per modello CLAP.
Calcola metriche di retrieval e accuracy.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import float32_to_int16, int16_to_float32
from .training import compute_embeddings, compute_contrastive_loss, compute_accuracy
from config.settings import settings

logger = logging.getLogger(__name__)


class CLAPTester:
    """Tester per valutazione modello CLAP."""

    def __init__(self, model, device: str = None):
        self.model = model
        self.device = device or settings.clap.device
        self.model.to(self.device)
        self.model.eval()

    def test_model(
            self,
            test_dataloader: DataLoader,
            compute_retrieval: bool = True,
            k_values: List[int] = None
    ) -> Dict[str, Any]:
        """
        Test completo del modello.

        Args:
            test_dataloader: Dataloader di test
            compute_retrieval: Se calcolare metriche di retrieval
            k_values: Valori k per Recall@k

        Returns:
            Dizionario con metriche e embeddings
        """
        if k_values is None:
            k_values = [1, 5, 10]

        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0

        all_audio_embeddings = []
        all_text_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
                # Calcola embeddings
                audio_embeddings, text_embeddings = compute_embeddings(
                    self.model, batch, self.device
                )

                # Calcola loss e accuracy
                loss, similarity = compute_contrastive_loss(
                    audio_embeddings, text_embeddings, settings.clap.logit_scale
                )
                accuracy = compute_accuracy(similarity)

                # Aggiorna metriche
                batch_size = len(batch['text'])
                total_loss += loss.item() * batch_size
                total_accuracy += accuracy * batch_size
                total_samples += batch_size

                # Salva embeddings per analisi
                all_audio_embeddings.append(audio_embeddings.cpu())
                all_text_embeddings.append(text_embeddings.cpu())
                all_labels.extend(batch['text'])

                # Log progresso
                if batch_idx % 5 == 0:
                    logger.info(
                        f"Test batch {batch_idx}: "
                        f"Loss={loss.item():.4f}, Acc={accuracy:.4f}"
                    )

        # Calcola metriche finali
        avg_loss = total_loss / max(total_samples, 1)
        avg_accuracy = total_accuracy / max(total_samples, 1)

        # Concatena embeddings
        if all_audio_embeddings:
            all_audio_embeddings = torch.cat(all_audio_embeddings, dim=0)
            all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

        # Calcola metriche di retrieval se richiesto
        retrieval_metrics = {}
        if compute_retrieval and len(all_audio_embeddings) > 0:
            retrieval_metrics = compute_retrieval_metrics(
                all_audio_embeddings,
                all_text_embeddings,
                k_values=k_values
            )

        # Prepara risultati
        results = {
            'test_loss': avg_loss,
            'test_accuracy': avg_accuracy,
            'total_samples': total_samples,
            'retrieval_metrics': retrieval_metrics,
            'audio_embeddings': all_audio_embeddings,
            'text_embeddings': all_text_embeddings,
            'labels': all_labels
        }

        # Log risultati
        self._log_results(results)

        return results

    def evaluate_single_example(
            self,
            audio_tensor: torch.Tensor,
            text: str
    ) -> Dict[str, Any]:
        """
        Valuta una singola coppia audio-testo.

        Args:
            audio_tensor: Tensor audio [1, audio_length]
            text: Testo descrittivo

        Returns:
            Dizionario con similarità e informazioni
        """
        self.model.eval()

        with torch.no_grad():
            # Preprocess audio
            audio_np = audio_tensor.cpu().numpy()
            audio_np = int16_to_float32(float32_to_int16(audio_np))
            audio_data = torch.from_numpy(audio_np).float().unsqueeze(0).to(self.device)

            # Get embeddings
            audio_embedding = self.model.get_audio_embedding_from_data(
                x=audio_data,
                use_tensor=True
            )
            text_embedding = self.model.get_text_embedding([text], use_tensor=True)

            # Normalizza
            audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
            text_embedding = F.normalize(text_embedding, p=2, dim=1)

            # Calcola similarità
            similarity = settings.clap.logit_scale * (audio_embedding @ text_embedding.T).item()

            # Converti in probabilità
            probability = 1 / (1 + np.exp(-similarity))

        return {
            'similarity_score': similarity,
            'probability': probability,
            'audio_embedding_shape': audio_embedding.shape,
            'text_embedding_shape': text_embedding.shape,
            'audio_embedding_norm': audio_embedding.norm().item(),
            'text_embedding_norm': text_embedding.norm().item()
        }

    def _log_results(self, results: Dict[str, Any]):
        """Log dei risultati del test."""
        logger.info(f"\n{'=' * 50}")
        logger.info("TEST RESULTS:")
        logger.info(f"{'=' * 50}")
        logger.info(f"Average Loss: {results['test_loss']:.4f}")
        logger.info(f"Average Accuracy: {results['test_accuracy']:.4f}")
        logger.info(f"Total Samples: {results['total_samples']}")

        if results['retrieval_metrics']:
            logger.info("\nRetrieval Metrics:")
            for key, value in results['retrieval_metrics'].items():
                logger.info(f"  {key}: {value:.4f}")


def compute_retrieval_metrics(
        audio_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        k_values: List[int] = None
) -> Dict[str, float]:
    """
    Calcola metriche di retrieval (Recall@k, MRR).

    Args:
        audio_embeddings: Embeddings audio
        text_embeddings: Embeddings testo
        k_values: Valori k per Recall@k

    Returns:
        Dizionario con metriche di retrieval
    """
    if k_values is None:
        k_values = [1, 5, 10]

    # Normalizza embeddings
    audio_embeddings = F.normalize(audio_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

    # Calcola matrice di similarità
    similarity = audio_embeddings @ text_embeddings.T
    n = similarity.shape[0]

    metrics = {}

    # Audio -> Text retrieval
    audio_sorted_indices = similarity.argsort(dim=1, descending=True)
    labels = torch.arange(n, device=similarity.device)

    for k in k_values:
        recall_k = (audio_sorted_indices[:, :k] == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[f'audio_recall@{k}'] = recall_k

    # Text -> Audio retrieval
    text_sorted_indices = similarity.argsort(dim=0, descending=True).T

    for k in k_values:
        recall_k = (text_sorted_indices[:, :k] == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[f'text_recall@{k}'] = recall_k

    # Mean Reciprocal Rank (MRR)
    audio_ranks = (audio_sorted_indices == labels.unsqueeze(1)).nonzero()[:, 1] + 1
    text_ranks = (text_sorted_indices == labels.unsqueeze(1)).nonzero()[:, 1] + 1

    metrics['audio_mrr'] = (1 / audio_ranks.float()).mean().item()
    metrics['text_mrr'] = (1 / text_ranks.float()).mean().item()

    # Mean Average Precision (MAP)
    metrics['audio_map'] = _compute_map(audio_sorted_indices, labels)
    metrics['text_map'] = _compute_map(text_sorted_indices, labels)

    return metrics


def _compute_map(sorted_indices: torch.Tensor, labels: torch.Tensor) -> float:
    """Calcola Mean Average Precision."""
    n = sorted_indices.shape[0]
    average_precisions = []

    for i in range(n):
        relevant = (sorted_indices[i] == labels[i]).float()
        precision_at_k = relevant.cumsum(dim=0) / torch.arange(1, n + 1, device=relevant.device)
        ap = (precision_at_k * relevant).sum() / relevant.sum()
        average_precisions.append(ap.item())

    return np.mean(average_precisions)


# Funzioni di interfaccia per backward compatibility
def test_model(
        model,
        test_dataloader: DataLoader,
        device: str = "cuda"
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, List[str]]:
    """
    Funzione di test compatibile con interfaccia originale.

    Args:
        model: Modello CLAP
        test_dataloader: Dataloader di test
        device: Device

    Returns:
        Tuple di (metrics, audio_embeddings, text_embeddings, labels)
    """
    tester = CLAPTester(model, device)
    results = tester.test_model(test_dataloader)

    return (
        {
            'test_loss': results['test_loss'],
            'test_accuracy': results['test_accuracy'],
            'total_samples': results['total_samples'],
            'retrieval_metrics': results['retrieval_metrics']
        },
        results['audio_embeddings'],
        results['text_embeddings'],
        results['labels']
    )


def evaluate_single_example(
        model,
        audio_tensor: torch.Tensor,
        text: str,
        device: str = "cuda"
) -> Dict[str, Any]:
    """
    Valuta una singola coppia audio-testo.

    Args:
        model: Modello CLAP
        audio_tensor: Tensor audio
        text: Testo
        device: Device

    Returns:
        Dizionario con risultati
    """
    tester = CLAPTester(model, device)
    return tester.evaluate_single_example(audio_tensor, text)


def create_test_dataloader(
        dataset,
        batch_size: int = 16,
        shuffle: bool = False,
        num_workers: int = 2
) -> DataLoader:
    """
    Crea DataLoader per testing.

    Args:
        dataset: Dataset
        batch_size: Dimensione batch
        shuffle: Se mischiare i dati
        num_workers: Worker per data loading

    Returns:
        DataLoader configurato
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_fn
    )


def _collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Funzione di collate per dataloader."""
    return {
        'audio': torch.stack([item['audio'] for item in batch]),
        'text': [item['text'] for item in batch]
    }