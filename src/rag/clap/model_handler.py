import logging
import torch
import laion_clap
import numpy as np
from typing import List
from config.settings import settings

logger = logging.getLogger(__name__)

class CLAPModelHandler:
    """
    Gestisce il caricamento e l'inferenza del modello CLAP (Laion).
    Ottimizzato per RAG (solo inferenza)
    """
    def __init__(self):
        self.device = "cuda" if settings.clap.use_cuda and torch.cuda.is_available() else "cpu"
        self.sample_rate = settings.clap.audio_sample_rate
        self.enable_fusion = settings.clap.enable_fusion
        
        logger.info(f"Inizializzazione CLAP su {self.device}...")
        
        try:
            self.model = laion_clap.CLAP_Module(
                enable_fusion=self.enable_fusion,
                amodel=settings.clap.model_name,
                device=self.device
            )
            
            # Caricamento pesi (se specificato un path locale, altrimenti scarica default)
            ckpt_id = 3 if self.enable_fusion else 1

            logger.info(f"Caricamento checkpoint ID {ckpt_id}...")
            self.model.load_ckpt(model_id=ckpt_id, verbose=False)

            self.model.to(self.device)
            self.model.eval()

            logger.info("Modello CLAP caricato e pronto per inferenza.")
            
        except Exception as e:
            logger.error(f"Errore critico avvio CLAP: {e}")
            raise

    def get_text_embedding(self, texts: List[str]) -> np.ndarray:
        """
        Genera embedding per una lista di testi.
        """
        if not self.model:
            raise RuntimeError("Modello CLAP non inizializzato")

        try:
            with torch.no_grad():
                text_embed = self.model.get_text_embedding(texts, use_tensor=False)
                return text_embed

        except Exception as e:
            logger.error(f"Errore embedding testo: {e}")
            return np.array([])

    def get_audio_embedding(self, audio_paths: List[str]) -> np.ndarray:
        """
        Genera embedding per una lista di file audio.
        """
        if not self.model:
            raise RuntimeError("Modello CLAP non inizializzato")

        try:
            with torch.no_grad():
                # use_tensor=False restituisce numpy array
                audio_embed = self.model.get_audio_embedding_from_filelist(
                    x=audio_paths,
                    use_tensor=False
                )
                return audio_embed

        except Exception as e:
            logger.error(f"Errore embedding audio: {e}")
            return np.array([])

# Singleton Factory
_clap_instance = None

def create_clap_model(pretrained: bool = True) -> CLAPModelHandler:
    """Restituisce l'istanza Singleton del modello."""
    global _clap_instance
    if _clap_instance is None:
        _clap_instance = CLAPModelHandler()
    return _clap_instance