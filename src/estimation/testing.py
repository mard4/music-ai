import logging

import laion_clap
import numpy as np
import torch
from torch.utils.data import DataLoader
from estimation.CLAPAudioDataset import float32_to_int16, int16_to_float32, get_clap_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model(model, test_dataloader, device="cuda"):
    """
    Testa il modello CLAP su un dataset di test

    Args:
        model: Modello CLAP addestrato
        test_dataloader: DataLoader con dati di test
        device: Device su cui eseguire il test

    Returns:
        dict: Metriche di test
    """
    model.eval()
    model.to(device)

    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_audio_embeddings = []
    all_text_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            audios = batch['audio'].to(device)
            texts = batch['text']

            # 1. Preprocess audio
            audio_data = []
            for audio_tensor in audios:
                audio_np = audio_tensor.cpu().numpy()
                audio_np = int16_to_float32(float32_to_int16(audio_np))
                audio_data.append(audio_np)

            audio_data = torch.from_numpy(np.array(audio_data)).float().to(device)

            # 2. Get embeddings
            audio_embeddings = model.get_audio_embedding_from_data(x=audio_data, use_tensor=True)
            text_embeddings = model.get_text_embedding(texts, use_tensor=True)

            # 3. Normalize embeddings
            audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=1)
            text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)

            # 4. Compute similarity and metrics
            logit_scale = 100
            similarity = logit_scale * audio_embeddings @ text_embeddings.T

            # Ground truth: diagonal matrix (audio_i matches text_i)
            batch_size = len(audios)
            labels = torch.arange(batch_size).to(device)

            # Cross-entropy loss
            loss_audio = torch.nn.functional.cross_entropy(similarity, labels)
            loss_text = torch.nn.functional.cross_entropy(similarity.T, labels)
            loss = (loss_audio + loss_text) / 2

            # Accuracy
            audio_to_text_pred = similarity.argmax(dim=1)
            text_to_audio_pred = similarity.argmax(dim=0)

            audio_acc = (audio_to_text_pred == labels).float().mean().item()
            text_acc = (text_to_audio_pred == labels).float().mean().item()
            avg_acc = (audio_acc + text_acc) / 2

            # Update totals
            total_loss += loss.item() * batch_size
            total_correct += avg_acc * batch_size
            total_samples += batch_size

            # Save embeddings for further analysis
            all_audio_embeddings.append(audio_embeddings.cpu())
            all_text_embeddings.append(text_embeddings.cpu())
            all_labels.extend(texts)

            if batch_idx % 5 == 0:
                logger.info(f"Test batch {batch_idx}: Loss={loss.item():.4f}, Acc={avg_acc:.4f}")

    # Compute final metrics
    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples

    # Concatenate all embeddings
    all_audio_embeddings = torch.cat(all_audio_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

    # Compute retrieval metrics
    retrieval_metrics = compute_retrieval_metrics(
        all_audio_embeddings,
        all_text_embeddings
    )

    metrics = {
        'test_loss': avg_loss,
        'test_accuracy': avg_accuracy,
        'total_samples': total_samples,
        'retrieval_metrics': retrieval_metrics
    }

    logger.info(f"\n{'=' * 50}")
    logger.info("TEST RESULTS:")
    logger.info(f"{'=' * 50}")
    logger.info(f"Average Loss: {avg_loss:.4f}")
    logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
    logger.info(f"Total Samples: {total_samples}")

    for key, value in retrieval_metrics.items():
        logger.info(f"{key}: {value:.4f}")

    return metrics, all_audio_embeddings, all_text_embeddings, all_labels


def compute_retrieval_metrics(audio_embeddings, text_embeddings, k_values=[1, 5, 10]):
    """
    Calcola metriche di retrieval (Recall@k, MRR)
    """
    # Normalize embeddings
    audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=1)
    text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)

    # Compute similarity matrix
    similarity = audio_embeddings @ text_embeddings.T

    n = similarity.shape[0]
    labels = torch.arange(n)

    metrics = {}

    # Audio -> Text retrieval
    audio_sorted_indices = similarity.argsort(dim=1, descending=True)
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

    return metrics


def evaluate_single_example(model, audio_tensor, text, device="cuda"):
    """
    Valuta una singola coppia audio-text

    Args:
        model: Modello CLAP
        audio_tensor: Tensor audio [1, audio_length]
        text: Stringa del testo
        device: Device

    Returns:
        dict: Similarità e risultati
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        # Process audio
        audio_np = audio_tensor.cpu().numpy()
        audio_np = int16_to_float32(float32_to_int16(audio_np))
        audio_data = torch.from_numpy(audio_np).float().unsqueeze(0).to(device)

        # Get embeddings
        audio_embedding = model.get_audio_embedding_from_data(x=audio_data, use_tensor=True)
        text_embedding = model.get_text_embedding([text], use_tensor=True)

        # Normalize
        audio_embedding = torch.nn.functional.normalize(audio_embedding, p=2, dim=1)
        text_embedding = torch.nn.functional.normalize(text_embedding, p=2, dim=1)

        # Compute similarity
        logit_scale = 100
        similarity = logit_scale * (audio_embedding @ text_embedding.T).item()

        # Convert to probability
        probability = 1 / (1 + np.exp(-similarity))

    return {
        'similarity_score': similarity,
        'probability': probability,
        'audio_embedding_shape': audio_embedding.shape,
        'text_embedding_shape': text_embedding.shape
    }


def create_test_dataloader(dataset, batch_size=16, shuffle=False):
    """
    Crea un DataLoader per testing
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )


if __name__ == "__main__":

    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    #model.load_state_dict(torch.load("path/to/checkpoint.pth"))

    test_dataset = get_clap_dataset()
    #test_dataset = YourAudioDataset(split='test')
    test_dataloader = create_test_dataloader(test_dataset, batch_size=8)

    metrics, audio_embs, text_embs, labels = test_model(
        model,
        test_dataloader,
        device="cuda"
    )

    sample_audio, sample_text = test_dataset[0]['audio'], test_dataset[0]['text']
    single_result = evaluate_single_example(
        model,
        sample_audio,
        sample_text,
        device="cuda"
    )

    print(f"\nSingle example result:")
    print(f"Similarity: {single_result['similarity_score']:.4f}")
    print(f"Probability: {single_result['probability']:.4f}")