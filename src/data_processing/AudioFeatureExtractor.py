import librosa
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from core.domain.audio import AudioFile, AudioMetadata, EnrichedAudioFile, Sample


class AudioFeatureExtractor:
    """Extracts audio features and generates technical/semantic descriptions."""

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def extract_audio_features(self, audio_path: Path, sr: Optional[int] = None) -> Optional[Dict]:
        """Extract comprehensive audio features from file."""

        sampling_rate = sr or self.sr
        try:
            audio_data, actual_sr = librosa.load(str(audio_path), sr=sampling_rate)
        except Exception:
            return None

        features = self._compute_time_domain_features(audio_data)
        features.update(self._compute_spectral_features(audio_data, actual_sr))
        features.update(self._compute_mfcc_features(audio_data, actual_sr))
        features.update(self._compute_chroma_features(audio_data, actual_sr))

        return features

    def _compute_time_domain_features(self, audio_data: np.ndarray) -> Dict:
        """Compute time-domain audio features."""
        return {
            'rms': np.mean(librosa.feature.rms(y=audio_data)),
            'zcr': np.mean(librosa.feature.zero_crossing_rate(audio_data))
        }

    def _compute_spectral_features(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Compute spectral audio features."""
        stft = np.abs(librosa.stft(audio_data))

        return {
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(S=stft, sr=sr)),
            'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(S=stft, sr=sr)),
            'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(S=stft, sr=sr))
        }

    def _compute_mfcc_features(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Compute MFCC features."""
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=5)
        features = {}
        for i in range(5):
            features[f'mfcc_{i + 1}'] = np.mean(mfccs[i])
        return features

    def _compute_chroma_features(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Compute chroma features."""
        stft = np.abs(librosa.stft(audio_data))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        return {'chroma': np.mean(chroma)}

    def generate_technical_description(self, features: Dict) -> List[str]:
        """Generate technical description based on audio features."""

        descriptions = []
        centroid = features.get('spectral_centroid', 0)
        rms = features.get('rms', 0)
        zcr = features.get('zcr', 0)

        if centroid < 1000:
            descriptions.append("low-pass below 1kHz")
        elif centroid > 4000:
            descriptions.append("high-pass above 4kHz")
        else:
            descriptions.append(f"peak at {int(centroid)}Hz")

        if rms < 0.01:
            descriptions.append("low gain")
        elif rms > 0.1:
            descriptions.append("high gain")

        if zcr > 0.1:
            descriptions.append("high noise content")

        return descriptions

    def generate_semantic_description(self, features: Dict) -> List[str]:
        """Generate semantic description based on audio features."""

        descriptions = []
        centroid = features.get('spectral_centroid', 0)
        bandwidth = features.get('spectral_bandwidth', 0)
        rms = features.get('rms', 0)
        zcr = features.get('zcr', 0)

        if centroid < 1000:
            descriptions.append("dark")
        elif centroid > 3000:
            descriptions.append("bright")

        descriptions.append("full" if bandwidth > 2000 else "thin")
        descriptions.append("punchy" if rms > 0.05 else "soft")
        descriptions.append("noisy" if zcr > 0.05 else "clean")

        return descriptions

    def create_descriptions(self, audio_path: Path) -> Optional[List[str]]:
        """Extract features and generate complete descriptions."""

        features = self.extract_audio_features(audio_path)
        if not features:
            return None

        technical_desc = self.generate_technical_description(features)
        semantic_desc = self.generate_semantic_description(features)

        technical_str = ", ".join(technical_desc)
        semantic_str = ", ".join(semantic_desc)

        return [
            f"Sound with {technical_str}",
            f"Audio featuring {technical_str} and {semantic_str} characteristics",
            f"Recording with {semantic_str} qualities, {technical_str}",
            f"{semantic_str} sound with {technical_str}",
            f"Audio sample: {technical_str}, {semantic_str}"
        ]