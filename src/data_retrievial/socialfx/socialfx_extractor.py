import numpy as np
from datasets import load_dataset
from typing import Dict, Any
import logging
from core.infrastructure.database.dependecies import get_socialfx_repository
from core.interfaces.repositories import SocialFxAudioRepository
from core.domain.audio import EffectParams, SocialFXEntry
from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class SocialFXKnowledgeBase:
    """
    Classe per gestire i dati SocialFX-Original.
    """

    def __init__(self, repository: SocialFxAudioRepository):
        """
        Inizializza con il repository iniettato.
        """
        self.repository = repository
        self.dataset_name = settings.socialfx_dataset_name

        self.raw_datasets = {}
        self.indices = {}

        self._load_dataset()
        self._build_indices()

    def _load_dataset(self):
        """Carica il dataset da HuggingFace Hub."""
        try:
            logging.info(f"Caricamento dataset: {self.dataset_name}...")
            ds = load_dataset(self.dataset_name)

            self.raw_datasets['eq'] = ds['eq']
            self.raw_datasets['comp'] = ds['comp']
            self.raw_datasets['reverb'] = ds['reverb']

            logging.info("Dataset caricato con successo.")
        except Exception as e:
            logging.error(f"Errore nel caricamento del dataset: {e}")
            raise

    def _build_indices(self):
        """Costruisce gli indici aggregati."""
        if not self.raw_datasets:
            raise ValueError("Dataset non caricato.")

        for effect_type, dataset in self.raw_datasets.items():
            self.indices[effect_type] = self._process_effect_data(dataset)

    def _process_effect_data(self, dataset) -> Dict[str, Any]:
        """Processa un singolo subset aggregando per descrittore (Mean Pooling)."""
        descriptor_groups = {}

        for sample in dataset:
            descriptor = sample['text'].lower().strip()
            param_values = sample['param_values']
            param_keys = sample['param_keys']

            if descriptor not in descriptor_groups:
                descriptor_groups[descriptor] = {
                    'param_values_list': [],
                    'param_keys': param_keys
                }
            descriptor_groups[descriptor]['param_values_list'].append(param_values)

        processed_index = {}
        for descriptor, data in descriptor_groups.items():
            param_values_list = data['param_values_list']
            avg_param_values = np.mean(param_values_list, axis=0).tolist()

            processed_index[descriptor] = {
                'param_values': avg_param_values,
                'param_keys': data['param_keys'],
                'sample_count': len(param_values_list)
            }

        return processed_index

    async def save_kb_to_mongo(self) -> bool:
        """
        Salva la Knowledge Base usando il Repository.
        """
        if not self.repository:
            logging.error("Repository non inizializzato.")
            return False

        try:
            total_saved = 0

            for effect_type, index_data in self.indices.items():
                logging.info(f"Salvataggio {effect_type.upper()} su MongoDB ({len(index_data)} voci)...")

                for descriptor, data in index_data.items():
                    entry = SocialFXEntry(
                        descriptor=descriptor,
                        effect_type=effect_type,
                        parameters=EffectParams(
                            param_values=data['param_values'],
                            param_keys=data['param_keys']
                        ),
                        sample_count=data['sample_count'],
                        source="socialfx-original"
                    )

                    # Delega al repository la logica di inserimento/upsert
                    await self.repository.insert_social_fx_audio(entry)
                    total_saved += 1

            logging.info(f"Salvataggio completato. Totale voci: {total_saved}")
            return True

        except Exception as e:
            logging.error(f"Errore durante il salvataggio: {e}")
            return False


async def create_socialfx_extractor(mongo_config: dict = None) -> SocialFXKnowledgeBase:
    """
    Factory che usa le dependencies del core.
    """
    # Usa il getter singleton che gestisce connessione e collection corretta
    repository = get_socialfx_repository()
    return SocialFXKnowledgeBase(repository=repository)


