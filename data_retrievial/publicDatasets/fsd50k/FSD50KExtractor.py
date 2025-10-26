import zipfile
import os
import pandas as pd
from pathlib import Path
import logging 
import io
from commons.utils import _checkOutputDir, _check_alternative_paths
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")

class FSD50KExtractor():
    def __init__(self, main_zip_path: Path, output_dir: Path):
        self.main_zip_path = main_zip_path
        self.output_dir = output_dir
        
    def extract_ground_truth(self, output_dir: Path = None):
        """
        Extract ground truth in csv files containing info about labels
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        if _checkOutputDir(output_dir):
            return

        print(f"Estraggo ground truth da {self.main_zip_path} in {output_dir}")
        with zipfile.ZipFile(self.main_zip_path, 'r') as main_zip:
            if 'FSD50K.ground_truth.zip' in main_zip.namelist():
                logging.info("Trovato FSD50K.ground_truth.zip, estraendo...")

                with main_zip.open('FSD50K.ground_truth.zip') as inner:
                    gt_bytes = inner.read()

                with zipfile.ZipFile(io.BytesIO(gt_bytes)) as gt_zip:
                    for member in gt_zip.infolist():
                        parts = member.filename.split('/')
                        sanitized = '/'.join(parts[1:]) if len(parts) > 1 else parts[0]
                        if not sanitized:
                            continue

                        target_path = os.path.join(output_dir, sanitized)
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)

                        if member.is_dir():
                            os.makedirs(target_path, exist_ok=True)
                            continue

                        with gt_zip.open(member) as src, open(target_path, 'wb') as dst:
                            dst.write(src.read())
                logging.info("Ground truth estratto con successo")
            else:
                logging.error("FSD50K.ground_truth.zip non trovato nell'archivio principale")
                
    def select_samples(self, ground_truth_dir: Path, categories: list, max_samples: int) -> pd.DataFrame:
        """
        Analizza la struttura di FSD50K
        e seleziona i campioni che corrispondono alle categorie specificate.
        Restituisce un DataFrame con i campioni selezionati.
        """
        
        try:
            dev_df = pd.read_csv(ground_truth_dir / "dev.csv")
            eval_df = pd.read_csv(ground_truth_dir / "eval.csv")
            vocabulary_df = pd.read_csv(ground_truth_dir / "vocabulary.csv")
        except Exception as e:
            logging.error(f"Errore nel caricamento dei file CSV: {e}")
            return pd.DataFrame()
        
        # Verifica che le categorie esistano nel vocabolario
        valid_categories = set(vocabulary_df['mids'])
        categories_filtered = [cat for cat in categories if cat in valid_categories]
        
        if len(categories_filtered) < len(categories):
            logging.warning(f"Alcune categorie non trovate nel vocabolario: {set(categories) - set(categories_filtered)}")
        
        if not categories_filtered:
            logging.error("Nessuna categoria valida trovata")
            return pd.DataFrame()
  
        def filter_relevant_samples(df):
            mask = df['labels'].apply(
                lambda x: any(category in str(x) for category in categories_filtered)
            )
            return df[mask]
    
        relevant_dev = filter_relevant_samples(dev_df)
        relevant_eval = filter_relevant_samples(eval_df)
        logging.info(f"Campioni rilevanti in dev set: {len(relevant_dev)}")
        logging.info(f"Campioni rilevanti in eval set: {len(relevant_eval)}")
    
        all_relevant = pd.concat([relevant_dev, relevant_eval], ignore_index=True)
        
        if len(all_relevant) > max_samples:
            selected_samples = all_relevant.sample(max_samples, random_state=42)
            logging.info(f"Selezionati {max_samples} campioni random su {len(all_relevant)} disponibili")
        else:
            selected_samples = all_relevant
            logging.info(f"Selezionati tutti i {len(selected_samples)} campioni disponibili")
    
        return selected_samples

    def create_sample_mapping(self, 
                            ground_truth_dir: Path = None,
                            categories: list = None, 
                            max_samples: int = 100) -> pd.DataFrame:
        """
        Crea un mapping dei campioni selezionati con le loro etichette, e salva in un CSV
        """
        if ground_truth_dir is None:
            ground_truth_dir = self.output_dir
            
        if categories is None:
            logging.error("Specificare almeno una categoria")
            return pd.DataFrame()

        if _checkOutputDir(self.output_dir):
            return pd.DataFrame()
    
        selected_samples = self.select_samples(ground_truth_dir, categories, max_samples)
        
        if selected_samples.empty:
            logging.error("Nessun campione selezionato")
            return pd.DataFrame()
            
        print(f"Campioni selezionati:\n{selected_samples.head()}")
        sample_mapping = []
    
        for idx, row in selected_samples.iterrows():
            fname = row['fname']
            labels = row['labels']
            
            # Determina lo split originale
            if 'split' in row:
                original_split = row['split']
            else:
                # Se non c'è colonna split, controlla da quale dataframe proviene
                original_split = 'eval' if idx >= len(pd.read_csv(ground_truth_dir / "dev.csv")) else 'dev'
            
            # Mappa lo split al percorso audio corretto
            if original_split in ['train', 'val']:
                audio_split = 'dev'
            else:
                audio_split = 'eval'
        
            sample_info = {
                'file_id': fname,
                'audio_file': f"{fname}.wav",
                'labels': labels,
                'split': audio_split,
                'original_split': original_split
            }
            sample_mapping.append(sample_info)
    
        mapping_df = pd.DataFrame(sample_mapping)
        mapping_path = self.output_dir / "samples_mapping.csv"
        mapping_df.to_csv(mapping_path, index=False)
    
        logging.info(f"Mapping salvato: {mapping_path}")
        logging.info(f"Creato mapping per {len(mapping_df)} campioni")
    
        return mapping_df
    
    def extract_selected_samples(input_dir: Path, output_dir: Path, selected_samples_dir_csv: Path):
        """Estrae i campioni selezionati dal fsd50k.zip con la struttura corretta"""
        
        if _checkOutputDir(output_dir):
            return
        
        df = pd.read_csv(selected_samples_dir_csv)
        
        logging.info(f"Estraendo {len(df)} file audio da fsd50k.zip")

        main_zip_path = input_dir / "fsd50k.zip"
        
        if not main_zip_path.exists():
            logging.error(f"File ZIP non trovato: {main_zip_path}")
            return False

        extracted_count = 0
        
        with tqdm(total=len(df), desc="Estraendo da ZIP") as pbar:
            for _, row in df.iterrows():
                audio_file = row['audio_file']  # es. "10000.wav"
                split = row.get('split', 'eval')  # Default a 'eval' se non specificato
                
                internal_path = f"fsd50k/FSD50K.{split}_audio_16k/{audio_file}"
                
                try:
                    with zipfile.ZipFile(main_zip_path, 'r') as zip_ref:
                        if internal_path in zip_ref.namelist():
                            with zip_ref.open(internal_path) as source_file:
                                output_file_path = output_dir / audio_file
                                with open(output_file_path, 'wb') as target_file:
                                    target_file.write(source_file.read())
                            extracted_count += 1
                            pbar.set_postfix(file=audio_file[:20])
                            print(f"✅ Estratto: {audio_file}")
                        else:
                            if _check_alternative_paths(zip_ref, audio_file, split, output_dir):
                                extracted_count += 1
                                pbar.set_postfix(file=audio_file[:20])
                            else:
                                logging.warning(f"❌ File non trovato: {internal_path}")
                                
                except Exception as e:
                    logging.error(f"Errore con {audio_file}: {e}")
                
                pbar.update(1)
        
        logging.info(f"Estrazione completata: {extracted_count}/{len(df)} file estratti")
        return extracted_count > 0

    def extract_audio_samples(self, selected_samples_csv: Path = None, output_dir: Path = None) -> bool:
        """
        Estrae i campioni audio selezionati dal file ZIP principale
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        if selected_samples_csv is None:
            selected_samples_csv = self.output_dir / "samples_mapping.csv"
        
        if not selected_samples_csv.exists():
            logging.error(f"File CSV dei campioni selezionati non trovato: {selected_samples_csv}")
            return False
            
        return self.extract_selected_samples(
            input_dir=self.main_zip_path.parent,
            output_dir=output_dir,
            selected_samples_dir_csv=selected_samples_csv
        )

    def extract_audio_samples_optimized(self, selected_samples_csv: Path = None, output_dir: Path = None) -> bool:
        """
        Versione ottimizzata che apre lo ZIP una volta sola
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        if selected_samples_csv is None:
            selected_samples_csv = self.output_dir / "samples_mapping.csv"
        
        if not selected_samples_csv.exists():
            logging.error(f"File CSV dei campioni selezionati non trovato: {selected_samples_csv}")
            return False
        
        df = pd.read_csv(selected_samples_csv)
        
        logging.info(f"Estraendo {len(df)} file audio da {self.main_zip_path}")

        if not self.main_zip_path.exists():
            logging.error(f"File ZIP non trovato: {self.main_zip_path}")
            return False

        extracted_count = 0
        not_found_files = []
        
        with zipfile.ZipFile(self.main_zip_path, 'r') as zip_ref:
            zip_contents = set(zip_ref.namelist())  # Cache per ricerca veloce
            
            with tqdm(total=len(df), desc="Estraendo audio") as pbar:
                for _, row in df.iterrows():
                    audio_file = row['audio_file']
                    split = row.get('split', 'eval')
                    
                    # Lista di path possibili
                    possible_paths = [
                        f"fsd50k/FSD50K.{split}_audio_16k/{audio_file}",
                        f"FSD50K.{split}_audio_16k/{audio_file}",
                        f"fsd50k/FSD50K.{split}_audio/{audio_file}",
                        f"FSD50K.{split}_audio/{audio_file}",
                        audio_file
                    ]
                    
                    file_extracted = False
                    
                    for internal_path in possible_paths:
                        if internal_path in zip_contents:
                            try:
                                with zip_ref.open(internal_path) as source_file:
                                    output_file_path = output_dir / audio_file
                                    with open(output_file_path, 'wb') as target_file:
                                        target_file.write(source_file.read())
                                extracted_count += 1
                                file_extracted = True
                                pbar.set_postfix(file=audio_file[:20], status="✅")
                                break
                            except Exception as e:
                                logging.error(f"Errore nell'estrarre {audio_file}: {e}")
                                continue
                    
                    if not file_extracted:
                        not_found_files.append(audio_file)
                        pbar.set_postfix(file=audio_file[:20], status="❌")
                        logging.warning(f"File non trovato: {audio_file}")
                    
                    pbar.update(1)
        
        # Report finale
        if not_found_files:
            logging.warning(f"File non trovati ({len(not_found_files)}): {not_found_files[:10]}{'...' if len(not_found_files) > 10 else ''}")
        
        success_rate = (extracted_count / len(df)) * 100
        logging.info(f"Estrazione completata: {extracted_count}/{len(df)} file estratti ({success_rate:.1f}%)")
        
        return extracted_count > 0

    def full_pipeline(self, categories: list, max_samples: int = 100) -> bool:
        """
        Esegue l'intera pipeline: estrazione ground truth, creazione mapping ed estrazione audio
        """
        try:
            logging.info("Step 1: Estrazione ground truth...")
            self.extract_ground_truth()
            
            logging.info("Step 2: Creazione mapping campioni...")
            mapping_df = self.create_sample_mapping(categories=categories, max_samples=max_samples)
            
            if mapping_df.empty:
                logging.error("Nessun campione selezionato per l'estrazione")
                return False
            
            logging.info("Step 3: Estrazione file audio...")
            success = self.extract_audio_samples_optimized()
            
            if success:
                logging.info("Pipeline completata con successo!")
            else:
                logging.error("Pipeline completata con errori")
                
            return success
            
        except Exception as e:
            logging.error(f"Errore durante la pipeline: {e}")
            return False

def extract_ground_truth(main_zip_path: Path, output_dir: Path):
    extractor = FSD50KExtractor(main_zip_path, output_dir)
    return extractor.extract_ground_truth()

def selectSamples(ground_truth_dir: Path, categories: list, max_samples: int) -> pd.DataFrame:
    extractor = FSD50KExtractor(Path("dummy.zip"), ground_truth_dir)
    return extractor.select_samples(ground_truth_dir, categories, max_samples)

def create_sample_mapping(output_dir: Path, ground_truth_dir: Path, categories: list, max_samples: int) -> pd.DataFrame:
    extractor = FSD50KExtractor(Path("dummy.zip"), output_dir)
    return extractor.create_sample_mapping(ground_truth_dir, categories, max_samples)

