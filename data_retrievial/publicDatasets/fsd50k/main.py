
from extract_ground_truth import extract_ground_truth, create_sample_mapping
from extract_samples_wav import extract_selected_samples
from pathlib import Path

fsd50k_path = Path("data/audios/")
fsd50k_zip_path = Path("data/audios/fsd50k.zip")
ground_truth_dir = Path("data/data_processed/fsd50k/fsd50k_ground_truth")
selected_samples_dir = Path("data/data_processed/fsd50k/fsd50k_selected_samples")
selected_samples_files_dir = Path("data/data_processed/fsd50k/fsd50k_selected_samples/audios")
    
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
    
    extract_ground_truth(main_zip_path=fsd50k_zip_path, output_dir=ground_truth_dir)
    create_sample_mapping(output_dir=selected_samples_dir,
                          ground_truth_dir=ground_truth_dir,
                          categories=categories,
                          max_samples=100)
    extract_selected_samples(input_dir=fsd50k_path,
                 output_dir=selected_samples_files_dir,
                  selected_samples_dir_csv=selected_samples_dir / "samples_mapping.csv")
