import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import logging

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLAPAudioDataset(Dataset):
    def __init__(self, csv_path, audio_dir, target_sr=48000, max_duration=10.0):
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.max_duration = max_duration
        
        # Carica e valida CSV
        self.df = pd.read_csv(csv_path)
        if 'audio_file' not in self.df.columns or 'labels' not in self.df.columns:
            raise ValueError("CSV deve contenere colonne 'audio_file' e 'labels'")
        
        # Filtra file esistenti
        self.valid_data = []
        for idx, row in self.df.iterrows():
            audio_path = os.path.join(audio_dir, row['audio_file'])
            if os.path.exists(audio_path):
                self.valid_data.append({
                    'audio_path': audio_path,
                    'labels': row['labels']
                })
            else:
                logger.warning(f"File non trovato: {audio_path}")
        
        logger.info(f"Caricati {len(self.valid_data)} esempi validi")
    
    def __len__(self):
        return len(self.valid_data)
    
    def __getitem__(self, idx):
        item = self.valid_data[idx]
        
        # Carica audio
        audio, sr = librosa.load(item['audio_path'], sr=self.target_sr, duration=self.max_duration)
        
        # Padding/troncamento
        target_length = int(self.target_sr * self.max_duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        return {
            'audio': torch.FloatTensor(audio),
            'text': item['labels'],
            'audio_path': item['audio_path']
        }
        
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

# INIZIO DIRETTO DEL TRAINING
if __name__ == "__main__":
    # Configurazione
    CSV_PATH = "H:/music-ai/data/data_processed/fsd50k/fsd50k_selected_samples/samples_mapping.csv"
    AUDIO_DIR = "H:/music-ai/data/data_processed/fsd50k/fsd50k_selected_samples/audios"
    
    # 1. Crea dataset direttamente
    logger.info("Creazione dataset CLAP...")
    dataset = CLAPAudioDataset(CSV_PATH, AUDIO_DIR)
    
    # 2. Crea dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=16,
        shuffle=True,
        num_workers=2
    )
    
    # 3. Training loop semplice - sostituisci con il training CLAP reale
    logger.info("Inizio training loop...")
    
    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    #model.cuda()
    
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    scaler = torch.amp.GradScaler()  # Se usi mixed precision
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(3):  # 3 epoche di esempio
        logger.info(f"Epoca {epoch + 1}")


    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(70):
        for batch_idx, batch in enumerate(dataloader):
            audios = batch['audio']  # Tensor PyTorch [batch_size, audio_length]
            texts = batch['text']    # Lista di stringhe
            
            # PREPARAZIONE AUDIO come nell'unit test
            audio_data = []
            for audio_tensor in audios:
                # Converti tensor -> numpy e applica quantizzazione come nell'unit test
                audio_np = audio_tensor.cpu().numpy()
                # Applica la stessa quantizzazione dell'unit test
                audio_np = int16_to_float32(float32_to_int16(audio_np))
                audio_data.append(audio_np)
            
            # Converti in tensor come nell'unit test
            audio_data = torch.from_numpy(np.array(audio_data)).float().cuda()
            
            # 3. Get embeddings come nell'unit test
            audio_embeddings = model.get_audio_embedding_from_data(x=audio_data, use_tensor=True)
            text_embeddings = model.get_text_embedding(texts, use_tensor=True)
            
            # 4. Calcola similarity e loss
            # Normalizza gli embeddings
            audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=1)
            text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)
            
            # Compute similarity matrix
            logit_scale = 100  # scaling factor
            similarity = logit_scale * audio_embeddings @ text_embeddings.T
            
            # Contrastive loss
            batch_size = len(audios)
            labels = torch.arange(batch_size).cuda()
            
            loss_audio = torch.nn.functional.cross_entropy(similarity, labels)
            loss_text = torch.nn.functional.cross_entropy(similarity.T, labels)
            loss = (loss_audio + loss_text) / 2
            
            # 5. Optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    logger.info("Training completato!")