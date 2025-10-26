import librosa
import numpy as np
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json



def prepare_clip_training_data(clip_dataset_csv: Path, output_dir: Path):
    """Prepara i dati per il training CLIP"""
    
    df = pd.read_csv(clip_dataset_csv)
    
    # Separa in train/val
    train_df = df[df['split'] == 'dev'].copy()
    val_df = df[df['split'] == 'eval'].copy()
    
    # Crea coppie positive (queste verranno usate per il contrastive learning)
    positive_pairs = []
    
    for _, row in df.iterrows():
        positive_pairs.append({
            'audio_path': row['audio_path'],
            'text': row['text_description'],
            'label': 1  # Coppia positiva
        })
    
    # Crea dataset finale
    training_data = {
        'train': train_df.to_dict('records'),
        'val': val_df.to_dict('records'),
        'positive_pairs': positive_pairs
    }
    
    # Salva i dati
    with open(output_dir / "clip_training_data.json", 'w') as f:
        json.dump(training_data, f, indent=2)
    
    # Salva anche in formato CSV per ispezione
    train_df.to_csv(output_dir / "clip_train.csv", index=False)
    val_df.to_csv(output_dir / "clip_val.csv", index=False)
    
    print(f"📊 Dati training preparati:")
    print(f"   - Train samples: {len(train_df)}")
    print(f"   - Val samples: {len(val_df)}")
    print(f"   - Coppie positive: {len(positive_pairs)}")
    
    return training_data

def create_config_file(output_dir: Path, clip_df: pd.DataFrame):
    """Crea file di configurazione per il training"""
    
    config = {
        "dataset_info": {
            "total_samples": len(clip_df),
            "unique_audio_files": clip_df['file_id'].nunique(),
            "unique_descriptions": clip_df['text_description'].nunique(),
            "audio_format": "wav",
            "sample_rate": 16000,
            "duration_seconds": "variable"
        },
        "training_config": {
            "positive_pairs": len(clip_df),
            "batch_size": 32,
            "audio_encoder": "CLAP_or_Custom",
            "text_encoder": "BERT_or_CLIP",
            "projection_dim": 512
        },
        "paths": {
            "audio_dir": "data/data_processed/fsd50k/fsd50k_selected_samples/audios",
            "clip_dataset": str(output_dir / "enriched_clip_dataset.csv"),
            "train_split": str(output_dir / "clip_train.csv"),
            "val_split": str(output_dir / "clip_val.csv")
        }
    }
    
    with open(output_dir / "dataset_config.json", 'w') as f:
        json.dump(config, f, indent=2)
        
def prepare_all_data():
    """Script completo per preparare tutti i dati per CLIP arricchiti."""
    
    # Paths
    samples_mapping_csv = Path("data/data_processed/fsd50k/fsd50k_selected_samples/samples_mapping.csv")
    audio_dir = Path("data/data_processed/fsd50k/fsd50k_selected_samples/audios")
    output_dir = Path("data/data_processed/clip_dataset")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Crea dataset CLIP arricchito
    print("🎵 Creando dataset CLIP arricchito...")
    clip_df = create_enriched_clip_dataset(samples_mapping_csv, audio_dir, output_dir)
    
    # 2. Prepara dati training
    print("🔧 Preparando dati training...")
    clip_dataset_csv = output_dir / "enriched_clip_dataset.csv"
    training_data = prepare_clip_training_data(clip_dataset_csv, output_dir)
    
    # 3. Crea file di configurazione
    create_config_file(output_dir, clip_df)
    
    print("✅ Preparazione dati CLIP arricchiti completata!")
    return clip_df, training_data

if __name__ == "__main__":
    # Installa dipendenze se necessario: pip install librosa scipy pandas tqdm
    prepare_all_data()