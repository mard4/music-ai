"""
Parameter Estimation Layer
Utilizza il dataset 'seungheondoh/socialfx-original' come knowledge base
per mappare aggettivi semantici a curve di equalizzazione.
"""
import numpy as np
from datasets import load_dataset
from typing import Dict, List, Optional, Any
import logging
import asyncio
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

# Configurazione di base del logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger('pymongo').setLevel(logging.WARNING)


# --- Pydantic Models per la Knowledge Base ---
class EQParams(BaseModel):
    """Modello per i parametri dell'equalizzatore."""
    param_values: List[float] = Field(..., description="Valori di guadagno per le bande")
    param_keys: List[str] = Field(..., description="Frequenze delle bande (es. 'Low', 'Mid')")


class SocialFXEntry(BaseModel):
    """
    Modello per una voce della Knowledge Base SocialFX.
    Rappresenta il mapping tra un descrittore semantico e i parametri tecnici.
    """
    descriptor: str = Field(..., description="Aggettivo semantico normalizzato (es. 'warm')")
    effect_type: str = Field("eq", description="Tipo di effetto (es. 'eq', 'comp')")
    parameters: EQParams = Field(..., description="I parametri tecnici associati")
    sample_count: int = Field(..., description="Numero di campioni originali aggregati per calcolare la media")
    source: str = Field("socialfx-original", description="Dataset di origine")


class SocialFXKnowledgeBase:
    """
    Classe per gestire la knowledge base derivata da SocialFX-Original.
    Carica i dati, crea indici per ricerca rapida e salva su MongoDB per RAG.
    """

    def __init__(self, dataset_name: str = "seungheondoh/socialfx-original",
                 collection: Optional[AsyncIOMotorCollection] = None):
        """
        Inizializza la knowledge base caricando il dataset SocialFX.

        Args:
            dataset_name: Nome del dataset su HuggingFace Hub
            collection: Collezione MongoDB per il salvataggio (Dependency Injection)
        """
        self.dataset_name = dataset_name
        self.collection = collection
        self.eq_data = None
        self.comp_data = None
        self.reverb_data = None

        # Indici per ricerca rapida
        self.eq_index = {}  # {descriptor: {param_values: [], param_keys: [], sample_count: int}}
        self.eq_descriptors = []  # Lista di tutti i descrittori unici

        self._load_dataset()
        self._build_indices()

    def _load_dataset(self):
        """Carica il dataset da HuggingFace Hub."""
        try:
            logging.info(f"Caricamento dataset {self.dataset_name}...")
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

        # Raggruppa i campioni per descrittore
        descriptor_groups = {}

        for sample in self.eq_data:
            descriptor = sample['text'].lower().strip()  # Normalizza
            param_values = sample['param_values']
            param_keys = sample['param_keys']

            if descriptor not in descriptor_groups:
                descriptor_groups[descriptor] = {
                    'param_values_list': [],
                    'param_keys': param_keys  # Le chiavi sono costanti per l'EQ
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
            logging.info(f"Inizio salvataggio KB su MongoDB ({self.collection.name})...")

            # 1. Creazione Indice
            # Fondamentale per il RAG: indicizziamo il campo 'descriptor' per lookup rapidi
            await self.collection.create_index("descriptor", unique=True)
            logging.info("Indice su 'descriptor' verificato/creato.")

            logging.info(f"Inizio upsert di {len(self.eq_index)} voci...")

            count = 0
            for descriptor, data in self.eq_index.items():
                # Crea il documento strutturato
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

                # Upsert: se il descrittore esiste, aggiorna i valori
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
    # Se non viene fornita una configurazione, prova a caricare quella di default
    if mongo_config is None:
        try:
            from config.settings import settings
            mongo_config = {
                "connection_string": settings.database.mongodb_connection_string,
                "database_name": settings.database.mongodb_database_name,
                "collection_name": "socialfx_kb"  # Collection specifica per la KB
            }
        except ImportError:
            logging.warning("Impossibile importare settings. Utilizzare config esplicita.")
            raise

    client = AsyncIOMotorClient(mongo_config["connection_string"])
    db = client[mongo_config["database_name"]]

    # Usiamo una collection specifica per la Knowledge Base, non quella degli audio files
    collection_name = mongo_config.get("collection_name", "socialfx_kb")
    collection = db[collection_name]

    logging.info(f"Factory: Creata connessione a MongoDB -> {collection_name}")

    return SocialFXKnowledgeBase(collection=collection)


