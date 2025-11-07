import pandas as pd
import os
import json
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
from pathlib import Path

class CLAPAudioDataset(Dataset):
    def __init__(self, csv_path, audio_dir, target_sr=48000, max_duration=10.0):
        """
        Dataset per preparare dati audio-testo per CLAP
        
        Args:
            csv_path (str): Percorso al file CSV con metadati
            audio_dir (str): Directory contenente i file audio
            target_sr (int): Sample rate target per l'audio (default: 48000 come in CLAP)
            max_duration (float): Durata massima in secondi (i file più lunghi vengono troncati)
        """
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.max_duration = max_duration
        
        # Carica il CSV
        self.df = pd.read_csv(csv_path)
        
        # Verifica che le colonne necessarie esistano
        required_columns = ['audio_file', 'labels']  # Modifica questi nomi in base al tuo CSV
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Colonna '{col}' non trovata nel CSV")
        
        # Filtra solo i file audio che esistono
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            audio_path = os.path.join(audio_dir, row['audio_file'])
            if os.path.exists(audio_path):
                self.valid_indices.append(idx)
            else:
                print(f"Avviso: File audio non trovato: {audio_path}")
        
        print(f"Caricati {len(self.valid_indices)} esempi validi su {len(self.df)} totali")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        
        # Carica l'audio
        audio_path = os.path.join(self.audio_dir, row['audio_file'])
        audio, sr = librosa.load(audio_path, sr=self.target_sr, duration=self.max_duration)
        
        # Se l'audio è più corto della durata massima, aggiungi padding
        target_length = int(self.target_sr * self.max_duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Prepara il testo
        text = str(row['labels'])
        
        return {
            'audio': torch.FloatTensor(audio),
            'labels': text,
            'audio_path': audio_path
        }
    
    def get_sample_rate(self):
        return self.target_sr

def prepare_clap_data(csv_path, audio_dir, output_dir=None):
    """
    Prepara i dati per CLAP e opzionalmente salva i metadati
    
    Args:
        csv_path (str): Percorso al CSV
        audio_dir (str): Directory audio
        output_dir (str): Directory di output opzionale per salvare i metadati
    
    Returns:
        CLAPAudioDataset: Dataset pronto per l'uso
    """
    dataset = CLAPAudioDataset(csv_path, audio_dir)
    
    if output_dir:
        # Crea la directory di output se non esiste
        os.makedirs(output_dir, exist_ok=True)
        
        # Salva i metadati in formato JSON
        metadata = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            metadata.append({
                'audio_path': sample['audio_path'],
                'text': sample['labels'],
                'audio_length': len(sample['audio']),
                'sample_rate': dataset.get_sample_rate()
            })
        
        metadata_path = os.path.join(output_dir, 'clap_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Metadati salvati in: {metadata_path}")
    
    return dataset

# Esempio di utilizzo
if __name__ == "__main__":
    # Configura i percorsi
    CSV_PATH = "H:/music-ai/data/data_processed/fsd50k/fsd50k_selected_samples/samples_mapping.csv"
    AUDIO_DIR = "H:/music-ai/data/data_processed/fsd50k/fsd50k_selected_samples/audios"
    OUTPUT_DIR = "prepared_data"
    
    # Prepara i dati
    dataset = prepare_clap_data(CSV_PATH, AUDIO_DIR, OUTPUT_DIR)
    
    # Esempio di accesso ai dati
    sample = dataset[0]
    print(f"Audio shape: {sample['audio'].shape}")
    print(f"Testo: {sample['labels']}")
    print(f"Sample rate: {dataset.get_sample_rate()} Hz")
    
    # Per utilizzare con un DataLoader di PyTorch
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=4
    )
    
    # Esempio di batch
    for batch in dataloader:
        audios = batch['audio']  # Shape: [batch_size, audio_length]
        texts = batch['labels']    # Lista di testi
        print(f"Batch audio shape: {audios.shape}")
        break