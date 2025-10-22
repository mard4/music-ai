import librosa
import numpy as np
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json


# Descrizioni tecniche (basate su features audio)

# Descrizioni semantiche (suono cupo, brillante, etc.)

# Metadati completi per il training CLIP/CLAP
# Configura il logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

def extract_audio_features(audio_path: Path, sr=16000):
    """Estrae caratteristiche audio per generare descrizioni tecniche."""
    
    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        logging.error(f"Errore nel caricare {audio_path}: {e}")
        return None
    
    features = {}
    
    # Features nel dominio del tempo
    features['rms'] = np.mean(librosa.feature.rms(y=y))
    features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # Features spettrali
    stft = np.abs(librosa.stft(y))
    spectral_centroids = librosa.feature.spectral_centroid(S=stft, sr=sr)
    features['spectral_centroid'] = np.mean(spectral_centroids)
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(S=stft, sr=sr))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(S=stft, sr=sr))
    
    # MFCCs (prendiamo le medie dei primi 5 coefficienti)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    for i in range(5):
        features[f'mfcc_{i+1}'] = np.mean(mfccs[i])
    
    # Altri features
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    features['chroma'] = np.mean(chroma)
    
    return features

def generate_technical_description(features):
    """Genera una descrizione tecnica basata sulle features audio."""
    
    desc = []
    
    # EQ: basato su spectral centroid e bandwidth
    centroid = features['spectral_centroid']
    if centroid < 1000:
        desc.append("low-pass below 1kHz")
    elif centroid > 4000:
        desc.append("high-pass above 4kHz")
    else:
        desc.append(f"peak at {int(centroid)}Hz")
    
    # Compressione: basato su RMS e dinamica
    rms = features['rms']
    if rms < 0.01:
        desc.append("low gain")
    elif rms > 0.1:
        desc.append("high gain")
    
    # Altri effetti basati su altre features
    zcr = features['zcr']
    if zcr > 0.1:
        desc.append("high noise content")
    
    return ", ".join(desc)

def generate_semantic_description(features):
    """Genera una descrizione semantica basata sulle features audio."""
    
    desc = []
    
    # Bright/Dark: basato su spectral centroid
    centroid = features['spectral_centroid']
    if centroid < 1000:
        desc.append("dark")
    elif centroid > 3000:
        desc.append("bright")
    
    # Full/Thin: basato su spectral bandwidth
    bandwidth = features['spectral_bandwidth']
    if bandwidth > 2000:
        desc.append("full")
    else:
        desc.append("thin")
    
    # Punchy/Soft: basato su RMS e ZCR
    rms = features['rms']
    if rms > 0.05:
        desc.append("punchy")
    else:
        desc.append("soft")
    
    # Clean/Noisy: basato su ZCR
    zcr = features['zcr']
    if zcr > 0.05:
        desc.append("noisy")
    else:
        desc.append("clean")
    
    return ", ".join(desc)

def create_text_descriptions(labels):
    """Crea descrizioni testuali dalle labels"""
    labels_clean = labels.strip('"')
    
    descriptions = [
        f"Sound of {labels_clean}",
        f"Audio recording of {labels_clean}",
        f"{labels_clean} sound effect",
        f"Professional {labels_clean} audio sample",
        f"High quality {labels_clean} sound",
        f"Digital recording of {labels_clean}",
        f"Clear audio of {labels_clean}",
        f"{labels_clean} audio clip"
    ]
    
    return list(set(descriptions))

def create_enriched_clip_dataset(samples_mapping_csv: Path, audio_dir: Path, output_dir: Path):
    """Crea un dataset arricchito per CLIP audio-text con metadati tecnici e semantici."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(samples_mapping_csv)
    
    clip_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio files"):
        audio_file = row['audio_file']
        labels = row['labels']
        split = row['split']
        file_id = row['file_id']
        
        # Verifica che il file audio esista
        audio_path = audio_dir / audio_file
        if not audio_path.exists():
            logging.warning(f"Audio file non trovato: {audio_file}")
            continue
        
        # Estrai features audio
        features = extract_audio_features(audio_path)
        if features is None:
            continue
        
        # Genera descrizioni tecniche e semantiche
        technical_desc = generate_technical_description(features)
        semantic_desc = generate_semantic_description(features)
        
        # Crea descrizioni testuali dalle labels (come prima)
        text_descriptions = create_text_descriptions(labels)
        
        # Aggiungi le descrizioni tecniche e semantiche come opzioni di testo
        text_descriptions.append(technical_desc)
        text_descriptions.append(semantic_desc)
        
        for desc in text_descriptions:
            clip_data.append({
                'file_id': file_id,
                'audio_path': str(audio_path),
                'text_description': desc,
                'split': split,
                'original_labels': labels,
                'technical_description': technical_desc,
                'semantic_description': semantic_desc,
                'features': str(features)  # Possiamo salvare le features come stringa per debug
            })
    
    # Salva il dataset CLIP arricchito
    clip_df = pd.DataFrame(clip_data)
    clip_df.to_csv(output_dir / "enriched_clip_dataset.csv", index=False)
    
    # Salva le statistiche
    print(f"🎯 Dataset CLIP arricchito creato:")
    print(f"   - Campioni totali: {len(clip_df)}")
    print(f"   - File audio unici: {clip_df['file_id'].nunique()}")
    print(f"   - Descrizioni uniche: {clip_df['text_description'].nunique()}")
    print(f"   - Split: {clip_df['split'].value_counts().to_dict()}")
    
    return clip_df

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