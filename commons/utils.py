import os
from pathlib import Path
import logging 
import zipfile

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")

def _checkOutputDir(output_dir: Path):
    """Controlla se la directory di output esiste e non è vuota"""
    if output_dir.exists():
        if any(output_dir.iterdir()):
            logging.info(f"Directory già esistente e non vuota, salto l'estrazione: {output_dir}")
            return True
        else:
            logging.info("Directory vuota, procedo con l'estrazione.")
            os.makedirs(output_dir, exist_ok=True)
    return False

def _check_and_create_output_dir(output_dir: Path, force_extract: bool = False) -> bool:
    """Helper function to check output directory and create if needed."""
    if output_dir.exists() and any(output_dir.iterdir()) and not force_extract:
        logging.info(f"Directory già esistente e non vuota, salto l'estrazione: {output_dir}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return True

def _check_alternative_paths(zip_ref: zipfile.ZipFile, audio_file: str, split: str, output_dir: Path) -> bool:
    """
    Controlla se uno dei path alternativi esiste nello zip
    """
    alternative_paths = [
        f"fsd50k/FSD50K.{split}_audio/{audio_file}",
        f"FSD50K.{split}_audio_16k/{audio_file}",
        f"FSD50K.{split}_audio/{audio_file}",
        f"fsd50k/FSD50K.{split}_audio_16k/{audio_file}",  
        f"FSD50K.{split}_audio/{audio_file}",
        audio_file  
    ]
    
    for alt_path in alternative_paths:
        if alt_path in zip_ref.namelist():
            try:
                with zip_ref.open(alt_path) as source_file:
                    output_file_path = output_dir / audio_file
                    with open(output_file_path, 'wb') as target_file:
                        target_file.write(source_file.read())
                print(f"✅ Estratto (path alternativo): {audio_file}")
                return True
            except Exception as e:
                logging.error(f"Errore nell'estrarre {audio_file} da {alt_path}: {e}")
                continue
    return False