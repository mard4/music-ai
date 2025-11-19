
from FSD50KExtractor import FSD50KExtractor, MongoDBConfig, full_pipeline_mongo
from pathlib import Path

DATA_DIR = Path("data").resolve()
FSD50K_ZIP =  DATA_DIR / "audios" / "fsd50k.zip"

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
    mongo_config = MongoDBConfig(
        connection_string="mongodb://localhost:27017/",
        database_name="fsd50k_db",
        audio_collection="audio_samples",
        fs_collection="audio_files"
    )
    
    full_pipeline_mongo(
        data_dir=FSD50K_ZIP,
        mongo_config=mongo_config,
        categories=categories,
        max_samples=100
    )
    
  