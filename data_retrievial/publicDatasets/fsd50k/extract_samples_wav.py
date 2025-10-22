import subprocess
import os
import pandas as pd
from pathlib import Path
import logging 
import io
from tqdm import tqdm
import zipfile

logging.basicConfig(level=logging.INFO)

def extract_selected_samples(input_dir: Path, output_dir: Path, selected_samples_dir_csv: Path):
    """Estrae i campioni selezionati dal fsd50k.zip con la struttura corretta"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_files = list(output_dir.glob("*.wav"))
    if existing_files:
        logging.info(f"Directory già contiene {len(existing_files)} file audio, salto l'estrazione: {output_dir}")
        return True
    
    df = pd.read_csv(selected_samples_dir_csv)
    
    logging.info(f"Estraendo {len(df)} file audio da fsd50k.zip")

    main_zip_path = input_dir / "fsd50k.zip"
    
    if not main_zip_path.exists():
        logging.error(f"File ZIP non trovato: {main_zip_path}")
        return False

    extracted_count = 0
    
    with tqdm(total=len(df), desc="🎵 Estraendo da ZIP") as pbar:
        for _, row in df.iterrows():
            audio_file = row['audio_file']  # es. "10000.wav"
            split = row.get('split', 'eval')
            
            # Path interno CORRETTO nel ZIP
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
                        # Cerca versioni alternative del path
                        alternative_paths = [
                            f"fsd50k/FSD50K.{split}_audio/{audio_file}",
                            f"FSD50K.{split}_audio_16k/{audio_file}",
                            f"FSD50K.{split}_audio/{audio_file}",
                        ]
                        
                        for alt_path in alternative_paths:
                            if alt_path in zip_ref.namelist():
                                with zip_ref.open(alt_path) as source_file:
                                    output_file_path = output_dir / audio_file
                                    with open(output_file_path, 'wb') as target_file:
                                        target_file.write(source_file.read())
                                extracted_count += 1
                                print(f"✅ Estratto (path alternativo): {audio_file}")
                                break
                        else:
                            logging.warning(f"❌ File non trovato: {internal_path}")
                            
            except Exception as e:
                logging.error(f"Errore con {audio_file}: {e}")
            
            pbar.update(1)
    
    logging.info(f"Estrazione completata: {extracted_count}/{len(df)} file estratti")
    return extracted_count > 0