import logging
from pathlib import Path

def _check_and_create_output_dir(output_dir: Path, force_extract: bool = False) -> bool:
    """Helper function to check output directory and create if needed."""
    if output_dir.exists() and any(output_dir.iterdir()) and not force_extract:
        logging.info(f"Directory già esistente e non vuota, salto l'estrazione: {output_dir}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return True