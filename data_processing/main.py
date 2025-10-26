from data_processing.AudioFeatureExtractor import AudioFeatureExtractor


def create_enriched_dataset(audio_dir: Path, output_dir: Path) -> pd.DataFrame:
    """
    Da folder audio con .wav o mp3.
    L'id del file è il nome senza estensione.
    Crea un dataset arricchito peraudio-text con metadati tecnici e semantici.
    """
    
    __checkOutputDir(output_dir)
    
    data = []
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
    
    if not audio_files:
        logging.warning(f"Nessun file audio trovato in {audio_dir}")
        return pd.DataFrame()

    extractor = AudioFeatureExtractor()

    for file in tqdm(audio_files, desc="Processing audio files"):

        file_id = file.stem
        audio_file = file.name

        try:
            features = extractor.extract_audio_features(file)
            if features is None:
                logging.warning(f"Impossibile estrarre features per: {audio_file}")
                continue
            
            technical_desc = extractor.generate_technical_description(features)
            semantic_desc = extractor.generate_semantic_description(features)
            file_descriptions = extractor.create_descriptions(features)
            
            for desc in file_descriptions:
                data.append({
                    'file_id': file_id,
                    'text_description': desc,
                    'technical_description': technical_desc,
                    'semantic_description': semantic_desc,
                    'duration': features.get('duration', 0),  
                    'rms': features.get('rms', 0),
                    'spectral_centroid': features.get('spectral_centroid', 0)
                })
                
        except Exception as e:
            logging.error(f"Errore processando {audio_file}: {e}")
            continue
    
    final_df = pd.DataFrame(data)
    final_df.to_csv(output_dir / "metadata.csv", index=False)
    
    print(f"File audio unici: {final_df['file_id'].nunique()}")
    
    return final_df


if __name__ == "__main__":
    samples_mapping_csv = Path("data/data_processed/fsd50k/fsd50k_selected_samples/samples_mapping.csv")
    audio_dir = Path("data/data_processed/fsd50k/fsd50k_selected_samples/audios")
    output_dir = Path("data/data_processed/fsd50k/clip_dataset")
    prepare_all_data(samples_mapping_csv,audio_dir,output_dir)
    
    
    
    
