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