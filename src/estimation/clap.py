import logging
import pandas as pd
import librosa
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from commons.data_models.models import MongoDBConfig
from estimation.CLAPAudioDataset import CLAPAudioDataset, create_processor_extractor, float32_to_int16, \
    int16_to_float32, get_clap_dataset, create_dataloaders
from estimation.training import train
import asyncio
import yaml
import laion_clap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    CONFIG_PATH = os.getenv("CONFIG_PATH", "./config.yaml")
    print(f"CONFIG_PATH: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        print(config)
    training_cfg = config["training"]
    
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=training_cfg.get("batch_size", 16),
        num_workers=training_cfg.get("num_workers", 0),
        train_ratio=training_cfg.get("train_ratio", 0.7),
        val_ratio=training_cfg.get("val_ratio", 0.15),
        test_ratio=training_cfg.get("test_ratio", 0.15)
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

    trained_model, stats = train(
        training_cfg=training_cfg,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cuda"
    )