import librosa
import numpy as np
import logging
from pathlib import Path
from commons.utils import _checkOutputDir

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

class AudioFeatureExtractor:
    """
    Classe per l'estrazione delle caratteristiche audio.
    Extende le funzionalità per generare descrizioni tecniche e semantiche.
    """
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def extract_audio_features(self, audio_path: Path, sr: int = 16000) -> dict:
        """Estrae caratteristiche audio per generare descrizioni tecniche."""
        
        if sr is None:
            sr = self.sr
        try:
            y, sr = librosa.load(str(audio_path), sr=sr)
        except Exception as e:
            logging.error(f"Errore nel caricare {audio_path}: {e}")
            return None
        
        features = {}
        
        # Features nel dominio del tempo
        features['rms'] = np.mean(librosa.feature.rms(y=y))
        features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Features spettrali
        stft = np.abs(librosa.stft(y))
        spectral_centroids = librosa.feature.spectral_centroid(S=stft, sr=sr)
        features['spectral_centroid'] = np.mean(spectral_centroids)
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(S=stft, sr=sr))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(S=stft, sr=sr))
        
        # MFCCs (prendiamo le medie dei primi 5 coefficienti)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
        for i in range(5):
            features[f'mfcc_{i+1}'] = np.mean(mfccs[i])
        
        # Altri features
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        features['chroma'] = np.mean(chroma)
        
        return features

    def generate_technical_description(self, features: dict) -> str:
        """Genera una descrizione tecnica basata sulle features audio."""
        
        desc = []
        
        # EQ: basato su spectral centroid e bandwidth
        centroid = features['spectral_centroid']
        if centroid < 1000:
            desc.append("low-pass below 1kHz")
        elif centroid > 4000:
            desc.append("high-pass above 4kHz")
        else:
            desc.append(f"peak at {int(centroid)}Hz")
        
        # Compressione: basato su RMS e dinamica
        rms = features['rms']
        if rms < 0.01:
            desc.append("low gain")
        elif rms > 0.1:
            desc.append("high gain")
        
        # Altri effetti basati su altre features
        zcr = features['zcr']
        if zcr > 0.1:
            desc.append("high noise content")
        
        return ", ".join(desc)

    def generate_semantic_description(self, features: dict) -> str:
        """Genera una descrizione semantica basata sulle features audio."""
        
        desc = []
        
        # Bright/Dark: basato su spectral centroid
        centroid = features['spectral_centroid']
        if centroid < 1000:
            desc.append("dark")
        elif centroid > 3000:
            desc.append("bright")
        
        # Full/Thin: basato su spectral bandwidth
        bandwidth = features['spectral_bandwidth']
        if bandwidth > 2000:
            desc.append("full")
        else:
            desc.append("thin")
        
        # Punchy/Soft: basato su RMS e ZCR
        rms = features['rms']
        if rms > 0.05:
            desc.append("punchy")
        else:
            desc.append("soft")
        
        # Clean/Noisy: basato su ZCR
        zcr = features['zcr']
        if zcr > 0.05:
            desc.append("noisy")
        else:
            desc.append("clean")
        
        return ", ".join(desc)
    
    def create_descriptions(self, audio_path: Path) -> list:
        """Estrae features e genera descrizioni tecniche e semantiche."""
        
        features = self.extract_audio_features(audio_path)
        if features is None:
            return None
        
        technical_desc = self.generate_technical_description(features)
        semantic_desc = self.generate_semantic_description(features)
        
        file_descriptions = [
                f"Sound with {technical_desc}",
                f"Audio featuring {technical_desc} and {semantic_desc} characteristics", 
                f"Recording with {semantic_desc} qualities, {technical_desc}",
                f"{semantic_desc} sound with {technical_desc}",
                f"Audio sample: {technical_desc}, {semantic_desc}"
            ]
        
        return file_descriptions
    
    
    
    
