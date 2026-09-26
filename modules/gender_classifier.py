import librosa
import numpy as np
import logging
import torch
from config import Config

logger = logging.getLogger(__name__)

class GenderClassifier:
    def __init__(self):
        # Pitch-based settings
        self.fem_threshold = 165.0  # Hz
        self.current_audio_path = None
        self.y = None
        self.sr = None
        
        # ML model settings
        self.ml_model = None
        self.ml_processor = None
        self.ml_initialized = False
        
    def initialize_ml_model(self):
        """Initialize wav2vec2 gender classification model."""
        if self.ml_initialized:
            return True
            
        try:
            # Check if transformers is available
            try:
                from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
            except ImportError:
                logger.error("transformers library not installed. Install with: uv pip install transformers")
                return False
            
            model_name = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
            logger.info(f"Loading ML gender model: {model_name}...")
            
            # Use AutoFeatureExtractor instead of Processor to avoid tokenizer issues
            # Using FeatureExtractor is sufficient for audio classification tasks
            self.ml_processor = AutoFeatureExtractor.from_pretrained(model_name)
            self.ml_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() and Config.USE_GPU else "cpu"
            self.ml_model = self.ml_model.to(device)
            self.ml_model.eval()
            
            self.ml_initialized = True
            logger.info(f"✅ ML gender model loaded on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ML gender model: {e}")
            logger.info("Falling back to pitch-based detection")
            import traceback
            traceback.print_exc()
            return False
        
    def load_audio(self, audio_path):
        """Loads audio into memory if not already loaded."""
        if self.current_audio_path != audio_path:
            logger.info(f"Loading full audio for gender analysis: {audio_path}")
            self.y, self.sr = librosa.load(audio_path, sr=16000)
            self.current_audio_path = audio_path
    
    def detect_gender_ml(self, audio_segment):
        """
        ML-based gender detection using wav2vec2.
        Returns: 'male', 'female', or 'unknown'
        """
        try:
            if not self.initialize_ml_model():
                return 'unknown'
            
            # Prepare input
            inputs = self.ml_processor(
                audio_segment,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Move to same device as model
            device = next(self.ml_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                logits = self.ml_model(**inputs).logits
                predicted_id = torch.argmax(logits, dim=-1).item()
            
            # Model outputs: 0 = female, 1 = male
            if predicted_id == 0:
                return 'female'
            elif predicted_id == 1:
                return 'male'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.warning(f"ML gender detection failed: {e}")
            return 'unknown'
    
    def detect_gender_pitch(self, y_seg):
        """
        Pitch-based gender detection (legacy method).
        Returns: 'male', 'female', or 'unknown'
        """
        try:
            if len(y_seg) < 512:
                return 'unknown'
            
            # Use YIN (faster than pYIN)
            f0 = librosa.yin(y_seg, fmin=80, fmax=400, sr=self.sr)
            
            # Filter valid pitch
            f0_clean = f0[~np.isnan(f0)]
            f0_clean = f0_clean[(f0_clean > 80) & (f0_clean < 400)]
            
            if len(f0_clean) == 0:
                return 'unknown'
            
            avg_pitch = np.mean(f0_clean)
            
            if avg_pitch < self.fem_threshold:
                return 'male'
            else:
                return 'female'
                
        except Exception as e:
            logger.error(f"Pitch detection error: {e}")
            return 'unknown'
            
    def detect_gender(self, audio_path, start_time, end_time):
        """
        Detects gender (male/female) from an audio segment.
        Uses ML model or pitch-based detection based on config.
        Returns: 'male', 'female', or 'unknown'
        """
        try:
            self.load_audio(audio_path)
            
            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)
            
            if end_sample - start_sample < 512:
                return 'unknown'
                
            y_seg = self.y[start_sample:end_sample]
            
            # Choose detection method
            if Config.AUDIO_GENDER_MODEL == "ml":
                gender = self.detect_gender_ml(y_seg)
                # Fallback to pitch if ML fails
                if gender == 'unknown':
                    gender = self.detect_gender_pitch(y_seg)
            else:
                gender = self.detect_gender_pitch(y_seg)
            
            return gender

        except Exception as e:
            logger.error(f"Error detecting gender: {e}")
            return 'unknown'

    def unload(self):
        """Unloads ML model to free VRAM for downstream translation and TTS."""
        if self.ml_model is not None:
            del self.ml_model
            del self.ml_processor
            self.ml_model = None
            self.ml_processor = None
            self.ml_initialized = False
        self.y = None
        self.current_audio_path = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("✅ ML gender model unloaded from VRAM")

# Singleton
classifier = GenderClassifier()
