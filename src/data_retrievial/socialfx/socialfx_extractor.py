import numpy as np
from datasets import load_dataset
from typing import Dict, List, Optional, Any
import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from core.domain.audio import EQParams, SocialFXEntry
from config.settings import settings, DatabaseSettings, mongo_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger('pymongo').setLevel(logging.WARNING)

class SocialFXKnowledgeBase:
    """
    Classe per gestire i dati SocialFX-Original contenenti parametri di eq,compressor
    Carica i dati, crea indici per ricerca rapida e salva su MongoDB per RAG.
    """

    def __init__(self, collection: Optional[AsyncIOMotorCollection] = None):
        """
        Inizializza la knowledge base caricando il dataset SocialFX.
        """
        self.dataset_name = settings.socialfx_dataset_name
        if collection is not None:
            self.collection = collection
        else:
            try:
                client = AsyncIOMotorClient(mongo_config["connection_string"])
                db = client[mongo_config["database_name"]]
                col_name = mongo_config.get("socialfx_collection")
                self.collection = db[col_name]
            except Exception as e:
                logging.error(f"Impossibile inizializzare connessione Mongo di default: {e}")
                self.collection = None

        self.eq_data = None
        self.comp_data = None
        self.reverb_data = None

        self.eq_index = {}  # {descriptor: {param_values: [], param_keys: [], sample_count: int}}
        self.eq_descriptors = []  # Lista di tutti i descrittori unici

        self._load_dataset()
        self._build_indices()

    def _load_dataset(self):
        """Carica il dataset da HuggingFace Hub."""
        try:
            ds = load_dataset(self.dataset_name)
            logging.info("Dataset caricato con successo.")

            self.eq_data = ds['eq']
            self.comp_data = ds['comp']
            self.reverb_data = ds['reverb']

            logging.info(f"Statistiche dataset:")
            logging.info(f"  - EQ samples: {len(self.eq_data)}")
            logging.info(f"  - Compressor samples: {len(self.comp_data)}")
            logging.info(f"  - Reverb samples: {len(self.reverb_data)}")

        except Exception as e:
            logging.error(f"Errore nel caricamento del dataset: {e}")
            raise

    def _build_indices(self):
        """Costruisce gli indici per ricerca rapida dei parametri EQ."""
        if self.eq_data is None:
            raise ValueError("Dataset non caricato. Chiamare _load_dataset prima.")

        descriptor_groups = {}

        for sample in self.eq_data:
            descriptor = sample['text'].lower().strip()
            param_values = sample['param_values']
            param_keys = sample['param_keys']

            if descriptor not in descriptor_groups:
                descriptor_groups[descriptor] = {
                    'param_values_list': [],
                    'param_keys': param_keys
                }

            descriptor_groups[descriptor]['param_values_list'].append(param_values)

        # Calcola la media dei parametri per ogni descrittore
        for descriptor, data in descriptor_groups.items():
            param_values_list = data['param_values_list']

            # Calcola la curva media (mean pooling)
            avg_param_values = np.mean(param_values_list, axis=0).tolist()

            # Salva nell'indice in memoria
            self.eq_index[descriptor] = {
                'param_values': avg_param_values,
                'param_keys': data['param_keys'],
                'sample_count': len(param_values_list)
            }

        self.eq_descriptors = list(self.eq_index.keys())
        logging.info(f"Indici costruiti. {len(self.eq_descriptors)} descrittori unici trovati.")

    async def save_kb_to_mongo(self) -> bool:
        """
        Salva la Knowledge Base (mappa descrittori -> parametri) nella collezione MongoDB iniettata.
        Questo abilita il sistema RAG a interrogare i parametri tramite query semantiche.
        """
        if self.collection is None:
            logging.error(
                "Collezione MongoDB non inizializzata. Passare una collezione valida nel costruttore o factory.")
            return False

        try:
            await self.collection.create_index("descriptor", unique=True)

            count = 0
            for descriptor, data in self.eq_index.items():

                entry = SocialFXEntry(
                    descriptor=descriptor,
                    effect_type="eq",
                    parameters=EQParams(
                        param_values=data['param_values'],
                        param_keys=data['param_keys']
                    ),
                    sample_count=data['sample_count'],
                    source="socialfx-original"
                )

                await self.collection.replace_one(
                    {"descriptor": descriptor},
                    entry.model_dump(),
                    upsert=True
                )
                count += 1

                if count % 100 == 0:
                    logging.info(f"Processati {count} descrittori...")

            logging.info(f"Salvataggio completato. {count} voci salvate.")
            return True

        except Exception as e:
            logging.error(f"Errore durante il salvataggio su MongoDB: {e}")
            return False


async def create_socialfx_extractor(mongo_config: dict = None) -> SocialFXKnowledgeBase:
    """
    Factory per creare SocialFXKnowledgeBase con dipendenze MongoDB configurate.
    Segue il pattern di Dependency Injection usato negli altri estrattori.
    """

    client = AsyncIOMotorClient(mongo_config["connection_string"])
    db = client[mongo_config["database_name"]]
    collection_name = mongo_config.get("socialfx_collection")
    print("collection name",collection_name)
    collection = db[collection_name]

    return SocialFXKnowledgeBase(collection=collection)

