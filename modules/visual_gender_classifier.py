import cv2
import numpy as np
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class VisualGenderClassifier:
    """
    Detects gender from video frames using face detection + gender classification.
    Uses DeepFace for accurate gender detection from facial features.
    """
    
    def __init__(self):
        self.detector_backend = 'opencv'  # Fast detection
        self.model_name = 'Gender'  # DeepFace gender model
        self.face_cache = {}  # Cache detected faces per timestamp
        self.current_video_path = None
        self.video_capture = None
        
    def load_video(self, video_path):
        """Load video for frame extraction."""
        if self.current_video_path != video_path:
            if self.video_capture is not None:
                self.video_capture.release()
            
            self.video_capture = cv2.VideoCapture(video_path)
            self.current_video_path = video_path
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            logger.info(f"Loaded video for visual gender detection: {video_path} (FPS: {self.fps})")
    
    def get_frame_at_time(self, video_path, timestamp):
        """Extract a frame at specific timestamp (in seconds)."""
        try:
            self.load_video(video_path)
            
            # Seek to timestamp
            frame_number = int(timestamp * self.fps)
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = self.video_capture.read()
            if ret:
                return frame
            return None
        except Exception as e:
            logger.error(f"Error extracting frame at {timestamp}s: {e}")
            return None
    
    def detect_gender_from_frame(self, frame):
        """
        Detect gender from a video frame using DeepFace.
        Returns: 'male', 'female', or 'unknown'
        """
        try:
            from deepface import DeepFace
            
            # Analyze frame
            result = DeepFace.analyze(
                frame, 
                actions=['gender'],
                detector_backend=self.detector_backend,
                enforce_detection=False,  # Don't fail if no face
                silent=True
            )
            
            # Handle both single result and list of results
            if isinstance(result, list):
                if len(result) == 0:
                    return 'unknown'
                result = result[0]
            
            # Get dominant gender
            gender_pred = result.get('dominant_gender', 'unknown')
            confidence = result.get('gender', {})
            
            # Map to our format
            if gender_pred.lower() in ['man', 'male']:
                return 'male'
            elif gender_pred.lower() in ['woman', 'female']:
                return 'female'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.debug(f"DeepFace gender detection failed: {e}")
            return 'unknown'
    
    def detect_gender(self, video_path, start_time, end_time):
        """
        Detect gender for a video segment.
        Samples multiple frames across the segment for robustness.
        """
        try:
            # Sample frames at start, middle, and end
            sample_times = [
                start_time,
                (start_time + end_time) / 2,
                end_time - 0.5  # Slightly before end
            ]
            
            gender_votes = []
            
            for timestamp in sample_times:
                # Check cache first
                cache_key = f"{video_path}_{timestamp:.2f}"
                if cache_key in self.face_cache:
                    gender = self.face_cache[cache_key]
                else:
                    frame = self.get_frame_at_time(video_path, timestamp)
                    if frame is None:
                        continue
                    
                    gender = self.detect_gender_from_frame(frame)
                    self.face_cache[cache_key] = gender
                
                if gender != 'unknown':
                    gender_votes.append(gender)
            
            # Majority vote
            if len(gender_votes) == 0:
                return 'unknown'
            
            # Count votes
            male_votes = gender_votes.count('male')
            female_votes = gender_votes.count('female')
            
            if male_votes > female_votes:
                logger.debug(f"Visual detection {start_time:.1f}-{end_time:.1f}s: Male ({male_votes}/{len(gender_votes)} votes)")
                return 'male'
            elif female_votes > male_votes:
                logger.debug(f"Visual detection {start_time:.1f}-{end_time:.1f}s: Female ({female_votes}/{len(gender_votes)} votes)")
                return 'female'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.error(f"Visual gender detection error: {e}")
            return 'unknown'
    
    def cleanup(self):
        """Release video resources."""
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.face_cache.clear()


# Singleton instance
visual_classifier = VisualGenderClassifier()
