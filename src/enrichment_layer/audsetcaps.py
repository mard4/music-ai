import os
import torch
import logging
import asyncio
import tempfile
import librosa
from typing import Optional
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import laion_clap
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket

# Import dai tuoi file di configurazione
# Assicurati che i path siano corretti rispetto alla struttura del tuo progetto
try:
    from config.settings import settings
    from infrastructure.database.repositories import MongoAudioFilesRepository, MongoEnrichedAudioRepository
except ImportError:
    # Fallback per test o esecuzione diretta
    from src.config.settings import settings
    from src.infrastructure.database.repositories import MongoAudioFilesRepository, MongoEnrichedAudioRepository

# Configurazione Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AudioSetCapsPipeline:
    """
    Implementazione della pipeline AudioSetCaps [Bai et al., 2024] AGGIORNATA a Qwen2-Audio.
    1. Ascolto (Qwen2-Audio): Estrazione feature grezze
    2. Scrittura (Mistral/Llama): Generazione caption
    3. Verifica (CLAP): Controllo qualità e filtraggio
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.qwen_model = None
        self.qwen_processor = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.clap_model = None

        # Configurazione per caricare i modelli in 4-bit (risparmio VRAM)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    def load_listening_model(self):
        """Stage 1: Carica Qwen2-Audio-7B-Instruct (LALM)"""
        logger.info("Caricamento Qwen2-Audio-7B-Instruct...")
        try:
            model_id = "Qwen/Qwen2-Audio-7B-Instruct"
            self.qwen_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=self.bnb_config,
                trust_remote_code=True
            ).eval()
        except Exception as e:
            logger.error(f"Errore caricamento Qwen2: {e}")
            raise

    def load_writing_model(self):
        """Stage 2: Carica Mistral-7B-Instruct (LLM)"""
        logger.info("Caricamento Mistral-7B...")
        try:
            model_id = "mistralai/Mistral-7B-Instruct-v0.2"
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=self.bnb_config
            ).eval()
        except Exception as e:
            logger.error(f"Errore caricamento Mistral: {e}")
            raise

    def load_verification_model(self):
        """Stage 3: Carica LAION-CLAP"""
        logger.info("Caricamento CLAP (LAION-Audio-630K)...")
        try:
            # Reinstallare laion-clap se necessario: pip install laion-clap
            self.clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
            self.clap_model.load_ckpt()  # Scarica checkpoint di default
            self.clap_model.to(self.device)
        except Exception as e:
            logger.error(f"Errore caricamento CLAP: {e}")
            raise

    async def step_1_listen(self, audio_path: str) -> str:
        """Estrae descrizioni grezze dall'audio usando Qwen2."""
        try:
            # 1. Template della conversazione per Qwen2
            conversation = [
                {'role': 'system', 'content': 'You are a helpful audio assistant.'},
                {'role': 'user', 'content': [
                    {'type': 'audio', 'audio_url': audio_path},
                    {'type': 'text', 'text': 'Describe this sound in detail, specifying instruments, timbre, and mood.'}
                ]}
            ]

            # 2. Preprocessing del testo
            text = self.qwen_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

            # 3. Caricamento e resampling audio
            # Qwen2 richiede il sample rate nativo del feature extractor (di solito 32k o 16k)
            # Usiamo librosa per caricare e ricampionare al volo
            target_sr = self.qwen_processor.feature_extractor.sampling_rate
            audio, _ = librosa.load(audio_path, sr=target_sr)

            # 4. Creazione tensori input
            inputs = self.qwen_processor(text=text, audios=audio, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Sposta su GPU

            # 5. Generazione
            with torch.no_grad():
                generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=256)

            # 6. Decodifica (rimuovendo i token di input per avere solo la risposta)
            # Qwen2 output include l'input, dobbiamo tagliarlo
            input_len = inputs['input_ids'].shape[1]
            generated_ids = generated_ids[:, input_len:]

            raw_description = self.qwen_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            logger.debug(f"Qwen2 Raw Output: {raw_description}")
            return raw_description

        except Exception as e:
            logger.error(f"Errore in step_1_listen: {e}")
            return "Audio analysis failed."

    async def step_2_write(self, raw_description: str, metadata_tags: list) -> str:
        """Riscrive la descrizione in una caption coerente."""

        tags_str = ", ".join(metadata_tags) if metadata_tags else "No tags"

        prompt = f"""[INST] You are an expert audio captioner. 
        Raw Analysis: {raw_description}
        User Tags: {tags_str}

        Task: Combine the raw analysis and user tags into a single, concise, and natural English caption. 
        Do not output any explanation, just the caption. [/INST]"""

        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.llm_model.generate(**inputs, max_new_tokens=128)

        caption = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Pulizia output
        if "[/INST]" in caption:
            caption = caption.split("[/INST]")[-1].strip()

        return caption

    async def step_3_verify(self, audio_path: str, caption: str) -> float:
        """Calcola score di similarità (Cosine Similarity) con CLAP."""
        try:
            # CLAP richiede liste
            audio_embeddings = self.clap_model.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=True)
            text_embeddings = self.clap_model.get_text_embedding([caption], use_tensor=True)

            # Calcolo Cosine Similarity
            similarity = torch.nn.functional.cosine_similarity(audio_embeddings, text_embeddings)
            return similarity.item()
        except Exception as e:
            logger.error(f"Errore calcolo CLAP score: {e}")
            return 0.0

    async def run_pipeline_on_file(self, audio_bytes: bytes, metadata: dict) -> Optional[dict]:
        """Esegue l'intera pipeline su un singolo file."""

        # Scrivi byte su file temporaneo (le librerie audio richiedono path)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name

        try:
            # 1. Listen
            raw_desc = await self.step_1_listen(temp_path)
            if not raw_desc or raw_desc == "Audio analysis failed.":
                return None

            # 2. Write
            tags = metadata.get('categories', [])
            caption = await self.step_2_write(raw_desc, tags)

            # 3. Verify
            score = await self.step_3_verify(temp_path, caption)

            logger.info(f"Pipeline Result -> Score: {score:.3f} | Caption: {caption}")

            result = {
                "generated_caption": caption,
                "raw_analysis": raw_desc,
                "clap_score": score,
                "model_version": "v2-qwen2-mistral-clap"
            }

            return result

        except Exception as e:
            logger.error(f"Errore pipeline generico: {e}")
            return None
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass


async def main_processor():
    """Loop principale per processare i file da MongoDB."""

    # 1. Connessione DB
    # Nota: Assicurati che settings sia caricato correttamente o usa valori hardcoded per test
    try:
        connection_string = settings.database.mongodb_connection_string
        db_name = settings.database.mongodb_database_name
    except:
        connection_string = "mongodb://localhost:27017/"
        db_name = "audio_db"

    client = AsyncIOMotorClient(connection_string)
    db = client[db_name]

    # Repository (Adatta i nomi delle collection se diverso)
    audio_collection_name = getattr(settings.database, "mongodb_audio_collection", "audio_samples")
    enriched_collection_name = "enriched_audio"  # O usa settings
    fs_collection_name = getattr(settings.database, "mongodb_fs_collection", "audio_files")

    audio_repo = MongoAudioFilesRepository(db[audio_collection_name])
    enriched_repo = MongoEnrichedAudioRepository(db[enriched_collection_name])
    fs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name=fs_collection_name)

    # 2. Inizializza Pipeline (Carica Modelli)
    logger.info("Inizializzazione modelli AI...")
    pipeline = AudioSetCapsPipeline()
    pipeline.load_listening_model()
    # pipeline.load_writing_model() # Decommenta se hai abbastanza VRAM (>16GB) o carica sequenzialmente
    # pipeline.load_verification_model() # Decommenta per stage 3

    # NOTA: Per GPU < 24GB, conviene caricare/scaricare i modelli uno alla volta
    # o fare batch processing per step (prima tutti step 1, poi tutti step 2, etc.)

    # 3. Recupera file da processare
    logger.info("Ricerca file da processare...")
    # Esempio: prendiamo file SampleFocus
    all_files = await audio_repo.find_audio_by_filter(source="samplefocus")
    logger.info(f"Trovati {len(all_files)} file totali.")

    processed_count = 0

    for audio_file in all_files:
        try:
            # Controllo se già processato
            # existing = await enriched_repo.get_enriched_by_gridfs_id(audio_file.gridfs_file_id)
            # if existing:
            #     continue

            logger.info(f"Processing: {audio_file.sample.file_name}")

            # Scarica Audio
            grid_out = await fs_bucket.open_download_stream_by_name(audio_file.sample.file_name)
            audio_bytes = await grid_out.read()

            # Esegui Pipeline
            # Per testare solo Qwen2 senza caricare altri modelli, chiamiamo step_1 direttamente
            # result = await pipeline.run_pipeline_on_file(audio_bytes, audio_file.metadata.model_dump())

            # TEST DEBUG QWEN2:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as t:
                t.write(audio_bytes)
                t_path = t.name

            desc = await pipeline.step_1_listen(t_path)
            print(f"DESCRIZIONE GENERATA: {desc}")
            os.remove(t_path)

            processed_count += 1
            if processed_count >= 5: break  # Limitiamo a 5 per test

        except Exception as e:
            logger.error(f"Errore file {audio_file.sample.file_name}: {e}")

    logger.info("Processamento terminato.")


if __name__ == "__main__":
    asyncio.run(main_processor())