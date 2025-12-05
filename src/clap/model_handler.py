"""
Gestione del modello CLAP - Caricamento, salvataggio e utilità.
"""
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import laion_clap

from .config import ModelConfig, CLAPConfig
from config.settings import settings

logger = logging.getLogger(__name__)


class CLAPModelHandler:
    """Handler per gestione modello CLAP."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.device = settings.clap.device

    def load_pretrained(self, model_path: Optional[str] = None) -> laion_clap.CLAP_Module:
        """
        Carica modello CLAP pre-addestrato.

        Args:
            model_path: Percorso al checkpoint (None per default)

        Returns:
            Modello CLAP caricato
        """
        try:
            # Inizializza modello
            self.model = laion_clap.CLAP_Module(
                enable_fusion=self.config.enable_fusion
            )

            # Carica weights
            if model_path and Path(model_path).exists():
                logger.info(f"Caricamento modello da checkpoint: {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)

                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                logger.info("Caricamento modello pre-addestrato default")
                self.model.load_ckpt()

            # Move to device
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Modello caricato su dispositivo: {self.device}")
            return self.model

        except Exception as e:
            logger.error(f"Errore caricamento modello: {e}")
            raise

    def save_model(
            self,
            filepath: str,
            include_optimizer: bool = False,
            optimizer: Optional[torch.optim.Optimizer] = None,
            epoch: Optional[int] = None,
            metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Salva modello su file.

        Args:
            filepath: Percorso file
            include_optimizer: Se salvare optimizer state
            optimizer: Optimizer da salvare
            epoch: Epoca corrente
            metrics: Metriche da salvare
        """
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'config': self.config.__dict__,
                'epoch': epoch or 0
            }

            if include_optimizer and optimizer:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

            if metrics:
                checkpoint['metrics'] = metrics

            # Crea directory se non esiste
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Salva checkpoint
            torch.save(checkpoint, filepath)
            logger.info(f"Modello salvato: {filepath}")

        except Exception as e:
            logger.error(f"Errore salvataggio modello: {e}")
            raise

    def load_checkpoint(
            self,
            filepath: str,
            load_optimizer: bool = False,
            optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Carica checkpoint completo.

        Args:
            filepath: Percorso checkpoint
            load_optimizer: Se caricare optimizer state
            optimizer: Optimizer da aggiornare

        Returns:
            Dizionario con checkpoint info
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)

            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state if requested
            if load_optimizer and optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            logger.info(f"Checkpoint caricato da: {filepath}")
            logger.info(f"Epoca: {checkpoint.get('epoch', 'N/A')}")

            return checkpoint

        except Exception as e:
            logger.error(f"Errore caricamento checkpoint: {e}")
            raise

    def freeze_layers(self, layer_names: Optional[list] = None):
        """
        Congela layers specifici del modello.

        Args:
            layer_names: Nomi layers da congelare (None per tutti tranne classificatore)
        """
        if layer_names:
            for name, param in self.model.named_parameters():
                if any(layer in name for layer in layer_names):
                    param.requires_grad = False
        else:
            # Congela tutto tranne il classificatore/fusion layer
            for name, param in self.model.named_parameters():
                if 'classifier' not in name and 'fusion' not in name:
                    param.requires_grad = False

        # Log dei parametri congelati/trainable
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        logger.info(f"Parametri trainable: {trainable_params:,} / {total_params:,} "
                    f"({trainable_params / total_params:.1%})")

    def get_model_info(self) -> Dict[str, Any]:
        """Restituisce informazioni sul modello."""
        if not self.model:
            return {}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Layer info
        layers_info = []
        for name, module in self.model.named_modules():
            if name:  # Skip empty name (root)
                layers_info.append({
                    'name': name,
                    'type': module.__class__.__name__,
                    'trainable': any(p.requires_grad for p in module.parameters()),
                    'num_parameters': sum(p.numel() for p in module.parameters())
                })

        return {
            'model_name': self.model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'device': str(next(self.model.parameters()).device),
            'layers': layers_info[:10]  # Primi 10 layers
        }

    def switch_to_train_mode(self):
        """Passa in modalità training."""
        if self.model:
            self.model.train()

    def switch_to_eval_mode(self):
        """Passa in modalità evaluation."""
        if self.model:
            self.model.eval()


# Factory function
def create_clap_model(
        config: Optional[ModelConfig] = None,
        pretrained: bool = True,
        model_path: Optional[str] = None
) -> CLAPModelHandler:
    """
    Factory per creare handler modello CLAP.

    Args:
        config: Configurazione modello
        pretrained: Se caricare weights pre-addestrati
        model_path: Percorso a checkpoint specifico

    Returns:
        CLAPModelHandler configurato
    """
    handler = CLAPModelHandler(config)

    if pretrained:
        handler.load_pretrained(model_path)

    return handler