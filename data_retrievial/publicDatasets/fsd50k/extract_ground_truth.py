import zipfile
import os
import pandas as pd
from pathlib import Path
import logging 
import io

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")

def extract_ground_truth(main_zip_path: Path, output_dir: Path):
    """Extract ground truth in csv files containing info about labels
    """
    
    if output_dir.exists():
        if any(output_dir.iterdir()):
            logging.info(f"Directory già esistente e non vuota, salto l'estrazione: {output_dir}")
            return
        else:
            logging.info("Directory vuota, procedo con l'estrazione.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with zipfile.ZipFile(main_zip_path, 'r') as main_zip:
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
                            
                            
def selectSamples(ground_truth_dir: Path, categories: list, max_samples: int) -> pd.DataFrame:
    """Analizza la struttura di FSD50K"""
    
    try:
        dev_df = pd.read_csv(os.path.join(ground_truth_dir, "dev.csv"))
        eval_df = pd.read_csv(os.path.join(ground_truth_dir, "eval.csv"))
        vocabulary_df = pd.read_csv(os.path.join(ground_truth_dir, "vocabulary.csv"))
    except Exception as e:
        logging.debug(f"Errore nel caricamento dei file CSV: {e}")
        return None, None, None
  
    def filter_relevant_samples(df):
        mask = df['labels'].apply(
            lambda x: any(category in x for category in categories)
        )
        return df[mask]
    
    relevant_dev = filter_relevant_samples(dev_df)
    relevant_eval = filter_relevant_samples(eval_df)
    logging.info(f"Campioni rilevanti in dev set: {len(relevant_dev)} \n Campioni rilevanti in eval set: {len(relevant_eval)}")
    
    all_relevant = pd.concat([relevant_dev, relevant_eval])
    if len(all_relevant) > max_samples:
        selected_samples = all_relevant.sample(max_samples, random_state=42)
    else:
        selected_samples = all_relevant
    
    logging.info(f"Campioni selezionati: {len(selected_samples)}")
    return selected_samples

def create_sample_mapping(output_dir: Path,
                          ground_truth_dir: Path,
                          categories: list,
                          max_samples: int) -> pd.DataFrame:
    """Crea un mapping dei campioni selezionati con le loro etichette, e salva in un CSV"""

    if output_dir.exists():
        if any(output_dir.iterdir()):
            logging.info(f"Directory già esistente e non vuota, salto l'estrazione: {output_dir}")
            return
        else:
            logging.info("Directory vuota, procedo con l'estrazione.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    selected_samples = selectSamples(ground_truth_dir, categories, max_samples)
    print(f"Campioni selezionati:\n{selected_samples}")
    sample_mapping = []
    
    for idx, row in selected_samples.iterrows():
        fname = row['fname']
        labels = row['labels']
        
        original_split = row.get('split', 'eval')
        if original_split in ['train', 'val']:
            audio_split = 'dev'  # sia train che val sono nel file FSD50K.dev_audio.zip
        else:
            audio_split = 'eval'  # per eval.csv non c'è colonna split
        
        sample_info = {
            'file_id': fname,
            'audio_file': f"{fname}.wav",
            'labels': labels,
            'split': audio_split
        }
        sample_mapping.append(sample_info)
    
    mapping_df = pd.DataFrame(sample_mapping)
    mapping_path = Path(output_dir) / "samples_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    
    logging.info(f"Mapping salvato: {output_dir}")
    
    return mapping_df


