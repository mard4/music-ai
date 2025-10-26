
from FSD50KExtractor import FSD50KExtractor
from pathlib import Path

DATA_DIR = Path("data").resolve()
AUDIO_DIR = DATA_DIR / "audios"
FSD50K_ZIP = AUDIO_DIR / "fsd50k.zip"
FSD50K_PROC_DIR = DATA_DIR / "data_processed" / "fsd50k"
GROUND_TRUTH_DIR = FSD50K_PROC_DIR / "ground_truth"

categories = [
        'Music', 'Musical', 'Instrument', 'Drum', 'Guitar', 'Bass', 'Violin', 
        'Piano', 'Synthesizer', 'String', 'Percussion', 'Wind', 'Brass',
        'Vehicle', 'Engine', 'Motor', 'Car', 'Train', 'Airplane', 'Horn',
        'Tools', 'Hammer', 'Saw', 'Drill', 'Machine', 'Power_tools',
        'Domestic', 'Clock', 'Door', 'Bell', 'Alarm', 'Telephone',
        'Water', 'Fire', 'Wind', 'Rain', 'Thunder', 'Nature',
        'Human', 'Speech', 'Laughter', 'Cough', 'Footsteps'
        ]      


if __name__ == "__main__":
        
    extractor = FSD50KExtractor(
        main_zip_path=FSD50K_ZIP,
        output_dir=GROUND_TRUTH_DIR)
    

    success= extractor.full_pipeline(
                categories=categories,
                max_samples=50
                )