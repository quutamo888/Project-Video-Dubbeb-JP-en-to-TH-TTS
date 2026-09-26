import re
import librosa
import numpy as np
import logging
import torch
from config import Config

logger = logging.getLogger(__name__)

class GenderClassifier:
    """
    Advanced Multi-Model Voice Gender Classifier & Speaker Diarization / Clustering.
    Supports:
    1. Robust SOTA ML models (audeering/wav2vec2-large-robust-12-ft-age-gender)
    2. Legacy LibriSpeech ML (alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech)
    3. Fast pitch-based (YIN algorithm)
    4. Speaker Acoustic Embedding Clustering + Duration-Weighted Majority Voting + Linguistic Pronoun Analysis
    """

    def __init__(self):
        # Pitch-based settings
        self.fem_threshold = 165.0  # Hz
        self.current_audio_path = None
        self.y = None
        self.sr = 16000

        # ML model settings
        self.ml_model = None
        self.ml_processor = None
        self.ml_model_name = None
        self.ml_initialized = False

    def initialize_ml_model(self, model_type=None):
        """
        Initializes audio ML model based on Config or parameter.
        Model options:
        - 'ml-robust': audeering/wav2vec2-large-robust-12-ft-age-gender
        - 'ml-librispeech': alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech
        - 'pitch': No model needed
        """
        selected_type = model_type or getattr(Config, "AUDIO_GENDER_MODEL", "ml-robust")
        if selected_type == "pitch":
            return True

        if self.ml_initialized and self.ml_model_name == selected_type:
            return True

        try:
            from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

            # Determine target model
            if selected_type in ["ml-robust", "audeering", "ml"]:
                target_model = getattr(Config, "ROBUST_GENDER_MODEL", "audeering/wav2vec2-large-robust-12-ft-age-gender")
            else:
                target_model = getattr(Config, "LEGACY_GENDER_MODEL", "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech")

            logger.info(f"Loading ML gender model: {target_model}...")

            token_arg = getattr(Config, "HF_TOKEN", None)

            try:
                self.ml_processor = AutoFeatureExtractor.from_pretrained(target_model, token=token_arg)
                self.ml_model = AutoModelForAudioClassification.from_pretrained(target_model, token=token_arg)
            except Exception as e_auto:
                logger.warning(f"AutoModelForAudioClassification fallback: {e_auto}")
                self.ml_processor = AutoFeatureExtractor.from_pretrained(target_model, token=token_arg)
                self.ml_model = Wav2Vec2ForSequenceClassification.from_pretrained(target_model, token=token_arg)

            device = "cuda" if torch.cuda.is_available() and Config.USE_GPU else "cpu"
            self.ml_model = self.ml_model.to(device)
            self.ml_model.eval()

            self.ml_model_name = selected_type
            self.ml_initialized = True
            logger.info(f"✅ ML gender model loaded: {target_model} on {device}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize primary ML gender model ({selected_type}): {e}")
            # Secondary fallback if robust model failed to download
            if selected_type != "ml-librispeech":
                logger.info("Attempting fallback to Legacy LibriSpeech ML model...")
                try:
                    fallback_model = getattr(Config, "LEGACY_GENDER_MODEL", "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech")
                    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
                    self.ml_processor = AutoFeatureExtractor.from_pretrained(fallback_model)
                    self.ml_model = Wav2Vec2ForSequenceClassification.from_pretrained(fallback_model)
                    device = "cuda" if torch.cuda.is_available() and Config.USE_GPU else "cpu"
                    self.ml_model = self.ml_model.to(device)
                    self.ml_model.eval()
                    self.ml_model_name = "ml-librispeech"
                    self.ml_initialized = True
                    logger.info("✅ Fallback ML model loaded successfully")
                    return True
                except Exception as e_fb:
                    logger.error(f"Fallback ML model failed: {e_fb}")

            logger.info("Falling back to pitch-based YIN frequency detection")
            return False

    def load_audio(self, audio_path):
        """Loads audio into memory if not already loaded."""
        if self.current_audio_path != audio_path:
            logger.info(f"Loading full audio for gender analysis: {audio_path}")
            self.y, self.sr = librosa.load(audio_path, sr=16000)
            self.current_audio_path = audio_path

    def detect_gender_ml(self, audio_segment):
        """
        ML-based gender detection using Wav2Vec2.
        Returns: 'male', 'female', or 'unknown'
        """
        try:
            if not self.initialize_ml_model():
                return 'unknown'

            if len(audio_segment) < 800:  # < 50ms is too short for reliable ML
                return 'unknown'

            inputs = self.ml_processor(
                audio_segment,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            device = next(self.ml_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.ml_model(**inputs)
                logits = outputs.logits

            # Check label mappings if available in model config
            id2label = getattr(self.ml_model.config, "id2label", None)
            if id2label and isinstance(id2label, dict):
                # If model has specific label names like female/male/child
                predicted_id = torch.argmax(logits, dim=-1).item()
                lbl = str(id2label.get(predicted_id, "")).lower()
                if "female" in lbl or "woman" in lbl:
                    return 'female'
                elif "male" in lbl or "man" in lbl:
                    return 'male'
                elif "child" in lbl:
                    # Children generally have higher F0 pitch similar to female register
                    return 'female'

            # Audeering model outputs 4 dims: [age, female, male, child]
            if logits.shape[-1] >= 4:
                gender_logits = logits[0, 1:4]
                pred_idx = torch.argmax(gender_logits).item()
                if pred_idx == 0:
                    return 'female'
                elif pred_idx == 1:
                    return 'male'
                else:
                    return 'female'

            # Standard binary output: 0 = female, 1 = male
            predicted_id = torch.argmax(logits, dim=-1).item()
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
        Pitch-based gender detection using YIN algorithm.
        Returns: 'male', 'female', or 'unknown'
        """
        try:
            if len(y_seg) < 512:
                return 'unknown'

            f0 = librosa.yin(y_seg, fmin=80, fmax=400, sr=self.sr)
            f0_clean = f0[~np.isnan(f0)]
            f0_clean = f0_clean[(f0_clean > 80) & (f0_clean < 400)]

            if len(f0_clean) == 0:
                return 'unknown'

            avg_pitch = float(np.mean(f0_clean))
            return 'male' if avg_pitch < self.fem_threshold else 'female'

        except Exception as e:
            logger.error(f"Pitch detection error: {e}")
            return 'unknown'

    def detect_gender(self, audio_path, start_time, end_time):
        """
        Detects gender (male/female) from an audio segment using configured audio model.
        Returns: 'male', 'female', or 'unknown'
        """
        try:
            self.load_audio(audio_path)

            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)

            if end_sample - start_sample < 512:
                return 'unknown'

            y_seg = self.y[start_sample:end_sample]

            model_mode = getattr(Config, "AUDIO_GENDER_MODEL", "ml-robust")
            if model_mode != "pitch":
                gender = self.detect_gender_ml(y_seg)
                if gender == 'unknown':
                    gender = self.detect_gender_pitch(y_seg)
            else:
                gender = self.detect_gender_pitch(y_seg)

            return gender

        except Exception as e:
            logger.error(f"Error detecting gender: {e}")
            return 'unknown'

    # =========================================================================
    # SPEAKER ACOUSTIC EMBEDDING & CLUSTERING
    # =========================================================================

    def extract_speaker_embedding(self, y_seg):
        """
        Extracts a normalized acoustic timbre embedding vector (MFCCs, Spectral, Pitch).
        Returns a 1D unit-normalized feature vector.
        """
        try:
            if len(y_seg) < 1024:
                return np.zeros(64, dtype=np.float32)

            # 1. 20 MFCCs (mean & std across time) -> 40 dims
            mfcc = librosa.feature.mfcc(y=y_seg, sr=self.sr, n_mfcc=20)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            # 2. Spectral Contrast (mean across 7 bands) -> 7 dims
            contrast = librosa.feature.spectral_contrast(y=y_seg, sr=self.sr)
            contrast_mean = np.mean(contrast, axis=1)

            # 3. Spectral Centroid & Rolloff (mean & std) -> 4 dims
            centroid = librosa.feature.spectral_centroid(y=y_seg, sr=self.sr)
            rolloff = librosa.feature.spectral_rolloff(y=y_seg, sr=self.sr)
            spec_stats = np.array([
                np.mean(centroid), np.std(centroid),
                np.mean(rolloff), np.std(rolloff)
            ], dtype=np.float32)

            # 4. Fundamental pitch statistics (F0) -> 2 dims
            f0 = librosa.yin(y_seg, fmin=75, fmax=400, sr=self.sr)
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) > 0:
                f0_stats = np.array([np.mean(f0_clean), np.std(f0_clean)], dtype=np.float32)
            else:
                f0_stats = np.array([150.0, 0.0], dtype=np.float32)

            # Concatenate all features
            features = np.concatenate([mfcc_mean, mfcc_std, contrast_mean, spec_stats, f0_stats])

            # Unit normalize
            norm = np.linalg.norm(features)
            if norm > 1e-8:
                features = features / norm
            return features

        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return np.zeros(64, dtype=np.float32)

    def cluster_speakers(self, segments, audio_path, distance_threshold=0.32, max_speakers=8):
        """
        Clusters audio segments into speaker groups using cosine distance on acoustic embeddings.
        Returns a dict mapping segment_index -> speaker_id ('SPEAKER_00', 'SPEAKER_01', ...)
        """
        self.load_audio(audio_path)

        clusters = []  # list of {'id': str, 'centroid': np.array, 'count': int, 'indices': []}
        seg_speaker_map = {}

        for idx, seg in enumerate(segments):
            start_s = int(seg['start'] * self.sr)
            end_s = int(seg['end'] * self.sr)
            y_seg = self.y[start_s:end_s] if end_s > start_s else np.array([])

            emb = self.extract_speaker_embedding(y_seg)

            if len(clusters) == 0:
                cluster_id = "SPEAKER_00"
                clusters.append({'id': cluster_id, 'centroid': emb, 'count': 1, 'indices': [idx]})
                seg_speaker_map[idx] = cluster_id
                continue

            # Compute cosine distance: 1 - dot(u, v) (vectors are unit normalized)
            best_dist = 999.0
            best_cluster_idx = -1

            for c_idx, c in enumerate(clusters):
                norm_c = np.linalg.norm(c['centroid'])
                norm_e = np.linalg.norm(emb)
                if norm_c > 1e-6 and norm_e > 1e-6:
                    sim = float(np.dot(c['centroid'], emb) / (norm_c * norm_e))
                    dist = 1.0 - sim
                else:
                    dist = 1.0

                if dist < best_dist:
                    best_dist = dist
                    best_cluster_idx = c_idx

            # Assign to existing cluster or create new one
            if best_dist <= distance_threshold or len(clusters) >= max_speakers:
                c = clusters[best_cluster_idx]
                c['indices'].append(idx)
                # Update centroid with running average
                c['centroid'] = (c['centroid'] * c['count'] + emb) / (c['count'] + 1)
                c['count'] += 1
                seg_speaker_map[idx] = c['id']
            else:
                cluster_id = f"SPEAKER_{len(clusters):02d}"
                clusters.append({'id': cluster_id, 'centroid': emb, 'count': 1, 'indices': [idx]})
                seg_speaker_map[idx] = cluster_id

        return seg_speaker_map, clusters

    # =========================================================================
    # LINGUISTIC PRONOUN & PARTICLE GENDER HINTS
    # =========================================================================

    @staticmethod
    def detect_linguistic_gender_hints(text, lang="ja"):
        """
        Analyzes Japanese or English text for gender-specific pronouns and particles.
        Returns: (male_score, female_score)
        """
        if not text:
            return 0.0, 0.0

        male_score = 0.0
        female_score = 0.0

        # Japanese Gender Markers
        # Strong first-person male pronouns: 僕 (boku), 俺 (ore), わし (washi), おいら
        if re.search(r'(僕|ボク|俺|オレ|わし|おいら|俺たち|僕たち)', text):
            male_score += 3.0

        # Masculine sentence-ending particles: ぜ, ぞ, だろ
        if re.search(r'([ぜぞ]\s*$|だろ\s*$|ぜ[！!]|ぞ[！!])', text):
            male_score += 1.5

        # Strong first-person female pronouns: あたし (atashi), アタシ, あたい, わたくし
        if re.search(r'(あたし|アタシ|あたい|わたくし|あたしたち)', text):
            female_score += 3.0

        # Feminine sentence-ending particles: かしら, わよ, のよ, わね
        if re.search(r'(かしら|わよ|のよ|わね|〜わ\s*$|わ[！!])', text):
            female_score += 1.5

        # English Gender Markers
        text_lower = text.lower()
        if re.search(r"\b(i'm a (boy|man|guy)|as a (man|guy|boy))\b", text_lower):
            male_score += 3.0
        if re.search(r"\b(i'm a (girl|woman|lady)|as a (girl|woman))\b", text_lower):
            female_score += 3.0

        return male_score, female_score

    # =========================================================================
    # COMPLETE BATCH PIPELINE WITH CONSENSUS MAJORITY VOTING
    # =========================================================================

    def analyze_all_segments(self, audio_path, segments, use_clustering=True, lang="ja", log_callback=None):
        """
        Executes complete gender analysis across all segments.
        If use_clustering is True:
          - Clusters segments by speaker
          - Aggregates duration-weighted audio votes + linguistic text hints
          - Resolves consistent gender per speaker (zero mid-dialogue flipping)
        """
        def log(msg):
            logger.info(msg)
            if log_callback:
                log_callback(msg)

        self.load_audio(audio_path)

        # 1. Collect segment-level predictions
        seg_predictions = []
        for i, seg in enumerate(segments):
            dur = max(0.5, min(5.0, seg['end'] - seg['start']))
            raw_gender = self.detect_gender(audio_path, seg['start'], seg['end'])
            m_hint, f_hint = self.detect_linguistic_gender_hints(seg.get('text', ''), lang)

            seg_predictions.append({
                'index': i,
                'raw_gender': raw_gender,
                'duration': dur,
                'm_hint': m_hint,
                'f_hint': f_hint
            })

        if not use_clustering:
            # Direct assignment without speaker clustering
            for i, p in enumerate(seg_predictions):
                gender = p['raw_gender'] if p['raw_gender'] != 'unknown' else 'female'
                segments[i]['gender'] = gender
                segments[i]['speaker'] = f"SPEAKER_{i:02d}"
            return segments

        # 2. Speaker Diarization / Clustering
        log("   👥 Performing Speaker Clustering & Majority Voting...")
        seg_speaker_map, clusters = self.cluster_speakers(segments, audio_path)

        # 3. Aggregate votes per speaker
        speaker_stats = {}
        for c in clusters:
            spk_id = c['id']
            speaker_stats[spk_id] = {
                'male_score': 0.0,
                'female_score': 0.0,
                'segment_indices': c['indices'],
                'male_count': 0,
                'female_count': 0
            }

        for p in seg_predictions:
            spk_id = seg_speaker_map.get(p['index'], "SPEAKER_00")
            stats = speaker_stats[spk_id]

            weight = p['duration']
            if p['raw_gender'] == 'male':
                stats['male_score'] += weight * 1.2
                stats['male_count'] += 1
            elif p['raw_gender'] == 'female':
                stats['female_score'] += weight * 1.2
                stats['female_count'] += 1

            # Incorporate linguistic hints
            stats['male_score'] += p['m_hint']
            stats['female_score'] += p['f_hint']

        # 4. Resolve winner gender per speaker cluster
        speaker_gender_resolved = {}
        for spk_id, stats in speaker_stats.items():
            tot = stats['male_score'] + stats['female_score']
            if stats['male_score'] >= stats['female_score']:
                win_gender = "male"
                conf = (stats['male_score'] / tot * 100) if tot > 0 else 50.0
            else:
                win_gender = "female"
                conf = (stats['female_score'] / tot * 100) if tot > 0 else 50.0

            speaker_gender_resolved[spk_id] = win_gender
            seg_len = len(stats['segment_indices'])
            log(f"   👤 [{spk_id}]: {win_gender.upper()} ({seg_len} segs, confidence: {conf:.1f}%)")

        # 5. Apply consistent speaker and gender tags to all segments
        for i, seg in enumerate(segments):
            spk_id = seg_speaker_map.get(i, "SPEAKER_00")
            seg['speaker'] = spk_id
            seg['gender'] = speaker_gender_resolved.get(spk_id, "female")

        return segments

    def unload(self):
        """Unloads ML model to free VRAM for downstream translation and TTS."""
        if self.ml_model is not None:
            del self.ml_model
            del self.ml_processor
            self.ml_model = None
            self.ml_processor = None
            self.ml_model_name = None
            self.ml_initialized = False

        self.y = None
        self.current_audio_path = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("✅ ML gender model unloaded from VRAM")

# Singleton
classifier = GenderClassifier()
