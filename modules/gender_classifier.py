import librosa
import numpy as np
import logging

logger = logging.getLogger(__name__)

class GenderClassifier:
    def __init__(self):
        # Thresholds (Hz)
        # Typically: Male 85-180 Hz, Female 165-255 Hz
        self.fem_threshold = 165.0
        self.current_audio_path = None
        self.y = None
        self.sr = None
        
    def load_audio(self, audio_path):
        """Loads audio into memory if not already loaded."""
        if self.current_audio_path != audio_path:
            logger.info(f"Loading full audio for gender analysis: {audio_path}")
            # Load with lower sample rate for faster pitch processing (16kHz is enough for pitch)
            self.y, self.sr = librosa.load(audio_path, sr=16000) 
            self.current_audio_path = audio_path
            
    def detect_gender(self, audio_path, start_time, end_time):
        """
        Detects gender (male/female) from an audio segment.
        Returns: 'male', 'female', or 'unknown'
        """
        try:
            self.load_audio(audio_path)
            
            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)
            
            if end_sample - start_sample < 512: # Too short
                return 'unknown'
                
            y_seg = self.y[start_sample:end_sample]
            
            # Use YIN (faster than pYIN)
            f0 = librosa.yin(y_seg, fmin=80, fmax=400, sr=self.sr)
            
            # Filter valid pitch (not NaN and within reasonable range)
            f0_clean = f0[~np.isnan(f0)]
            f0_clean = f0_clean[(f0_clean > 80) & (f0_clean < 400)]
            
            if len(f0_clean) == 0:
                return 'unknown'
            
            avg_pitch = np.mean(f0_clean)
            
            # logger.debug(f"Segment {start_time:.2f}-{end_time:.2f}s Pitch: {avg_pitch:.2f} Hz")
            
            if avg_pitch < self.fem_threshold:
                return 'male'
            else:
                return 'female'
                
        except Exception as e:
            logger.error(f"Error detecting gender: {e}")
            return 'unknown'

# Singleton
classifier = GenderClassifier()
