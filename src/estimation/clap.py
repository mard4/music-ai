import logging
import pandas as pd
import librosa
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from commons.data_models.models import MongoDBConfig
from estimation.CLAPAudioDataset import CLAPAudioDataset, create_processor_extractor, float32_to_int16, int16_to_float32
import asyncio
import yaml
import laion_clap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: split train / val / test
# TODO: augmentations???
#TODO: 


def get_clap_dataset() -> CLAPAudioDataset:
    logger.info("Creazione dataset CLAP...")
    mongo_config = MongoDBConfig(
        connection_string=os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017"),
        database_name=os.getenv("MONGODB_DATABASE_NAME", "audio_db"),
        audio_collection=os.getenv("MONGODB_AUDIO_COLLECTION", "audio_samples"),
        fs_collection=os.getenv("MONGODB_FS_COLLECTION", "fs_audio_files")
    )
    
    return asyncio.run(create_processor_extractor(mongo_config))
    

if __name__ == "__main__":
    dataset = get_clap_dataset()
    logger.info(f"Elementi nel dataset: {len(dataset)}")    
    
    CONFIG_PATH = os.getenv("CONFIG_PATH", "./src/estimation/config.yaml")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        print(config)
    training_cfg = config["training"]
    
    dataloader = DataLoader(
        dataset, 
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    
    logger.info("Inizio training loop...")
    
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    #model.cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(training_cfg["optimizer"]["lr"]),
                                  weight_decay=training_cfg["optimizer"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg["scheduler"]["T_max"])
    scaler = torch.amp.GradScaler()  # Se usi mixed precision
    criterion = torch.nn.CrossEntropyLoss()

    def train(training_cfg=training_cfg, dataloader=dataloader, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, criterion=criterion, logger=logger):
        for epoch in range(training_cfg["epochs"]): 
            logger.info(f"Epoca {epoch + 1}")
            model.train()
            for batch_idx, batch in enumerate(dataloader):
                audios = batch['audio']  # Tensor PyTorch [batch_size, audio_length]
                texts = batch['text']    # Lista di stringhe
                
                audio_data = []
                for audio_tensor in audios:
                    # Converti tensor -> numpy e applica quantizzazione
                    audio_np = audio_tensor.cpu().numpy()
                    # stessa quantizzazione dell'unit test
                    audio_np = int16_to_float32(float32_to_int16(audio_np))
                    audio_data.append(audio_np)
                
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
        
    train()