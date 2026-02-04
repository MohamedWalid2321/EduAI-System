"""
Face Recognition Module
=======================
Face recognition using InsightFace library:
- RetinaFace for face detection
- ArcFace for face recognition/embedding extraction

Usage:
    from face_recognition import FaceRecognition
    
    # Initialize
    fr = FaceRecognition()
    
    # Compare frame with reference image
    result = fr.compare_faces(frame, reference_image)
    # Returns: {
    #     'id': 3,
    #     'timestamp': '2026-02-03T23:38:20.483018',
    #     'flag': True/False,
    #     'probability': 0.8678,
    #     'evidence': '...'
    # }
    
    # Or with base64 encoded images
    result = fr.compare_faces_base64(frame_base64, reference_base64)
"""
"""
HOW TO USE:

from face_recognition import FaceRecognition
import cv2

# Initialize
fr = FaceRecognition()

# Load images
reference = cv2.imread("reference.jpg")  # Authorized person's photo
frame = cv2.imread("frame.jpg")          # Current frame to check

# Compare
result = fr.compare_faces(frame, reference)  # for file inputs
# or result = fr.compare_faces_base64(frame_base64, reference_base64)  # for base64 inputs (ex: from web)
print(result)


Output: 
{
    'id': 3,
    'timestamp': '2026-02-03T23:38:20.483018',
    'flag': False,
    'probability': 0.9123,
    'evidence': 'Authorized person verified (similarity: 0.9123)'
}
"""

import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import base64

try:
    from insightface.app import FaceAnalysis
except ImportError:
    raise ImportError(
        "InsightFace is required. Install with: pip install insightface onnxruntime"
    )


# Constants
FACE_RECOGNITION_MODULE_ID = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_DETECTION_THRESHOLD = 0.5


class FaceRecognition:
    """
    Face Recognition for AI Proctoring.
    
    Uses RetinaFace for detection and ArcFace for recognition.
    
    Cheating is flagged when:
    - Multiple faces are detected (unauthorized person in frame)
    - No face is detected (person left the frame)
    - Face doesn't match registered identity (different person)
    """
    
    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
        model_name: str = "buffalo_l",
        ctx_id: int = 0,  # 0 for GPU, -1 for CPU
        det_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize the Face Recognition Pipeline.
        
        Args:
            similarity_threshold: Minimum cosine similarity to consider a match (0.0 to 1.0)
            detection_threshold: Minimum confidence for face detection
            model_name: InsightFace model pack name ('buffalo_l', 'buffalo_s', 'buffalo_sc')
            ctx_id: Context ID (0 for GPU, -1 for CPU)
            det_size: Detection input size (width, height)
        """
        self.similarity_threshold = similarity_threshold
        self.detection_threshold = detection_threshold
        
        # Initialize InsightFace FaceAnalysis
        # This automatically loads RetinaFace for detection and ArcFace for recognition
        self.face_analyzer = FaceAnalysis(
            name=model_name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_analyzer.prepare(ctx_id=ctx_id, det_size=det_size)
    
    def compare_faces(
        self, 
        frame: np.ndarray, 
        reference_image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare face in current frame with reference image.
        
        This is the main method - stateless comparison of two images.
        
        Args:
            frame: BGR image from video (current frame to check)
            reference_image: BGR image of the authorized person (reference)
            
        Returns:
            dict: {
                'id': int,              # Module ID (3 for face recognition)
                'timestamp': str,       # ISO format timestamp
                'flag': bool,           # True = cheating detected
                'probability': float,   # Confidence score (0.0 to 1.0)
                'evidence': str         # Description of detection result
            }
        """
        timestamp = datetime.now().isoformat()
        
        # Detect face in reference image first
        ref_faces = self._detect_faces(reference_image)
        
        if len(ref_faces) == 0:
            return {
                'id': FACE_RECOGNITION_MODULE_ID,
                'timestamp': timestamp,
                'flag': True,
                'probability': 0.0,
                'evidence': 'No face detected in reference image'
            }
        
        if len(ref_faces) > 1:
            return {
                'id': FACE_RECOGNITION_MODULE_ID,
                'timestamp': timestamp,
                'flag': True,
                'probability': 0.0,
                'evidence': f'Multiple faces ({len(ref_faces)}) in reference image. Please provide single face.'
            }
        
        # Get reference embedding
        ref_embedding = ref_faces[0].embedding / np.linalg.norm(ref_faces[0].embedding)
        
        # Detect faces in current frame
        frame_faces = self._detect_faces(frame)
        num_faces = len(frame_faces)
        
        # Case 1: Multiple faces detected - CHEATING
        if num_faces > 1:
            return {
                'id': FACE_RECOGNITION_MODULE_ID,
                'timestamp': timestamp,
                'flag': True,
                'probability': 0.95,
                'evidence': f'Multiple faces detected: {num_faces} faces found in frame'
            }
        
        # Case 2: No face detected - CHEATING
        if num_faces == 0:
            return {
                'id': FACE_RECOGNITION_MODULE_ID,
                'timestamp': timestamp,
                'flag': True,
                'probability': 0.85,
                'evidence': 'No face detected in frame'
            }
        
        # Case 3: Single face detected - verify identity
        frame_embedding = frame_faces[0].embedding / np.linalg.norm(frame_faces[0].embedding)
        similarity = self._compute_similarity(frame_embedding, ref_embedding)
        
        # Check if face matches reference
        if similarity < self.similarity_threshold:
            return {
                'id': FACE_RECOGNITION_MODULE_ID,
                'timestamp': timestamp,
                'flag': True,
                'probability': round(1.0 - similarity, 4), # Higher means more likely cheating, 4: decimal places
                'evidence': f'Face does not match reference identity (similarity: {similarity:.4f})'
            }
        
        # All checks passed - NO CHEATING
        return {
            'id': FACE_RECOGNITION_MODULE_ID,
            'timestamp': timestamp,
            'flag': False,
            'probability': round(similarity, 4), # Higher means less likely cheating, 4: decimal places
            'evidence': f'Authorized person verified (similarity: {similarity:.4f})'
        }
    
    def compare_faces_base64(
        self, 
        frame_base64: str, 
        reference_base64: str
    ) -> Dict[str, Any]:
        """
        Compare faces using base64 encoded images.
        
        Args:
            frame_base64: Base64 encoded current frame
            reference_base64: Base64 encoded reference image
            
        Returns:
            Same as compare_faces()
        """
        frame = self._decode_base64_image(frame_base64)
        reference = self._decode_base64_image(reference_base64)
        
        if frame is None:
            return {
                'id': FACE_RECOGNITION_MODULE_ID,
                'timestamp': datetime.now().isoformat(),
                'flag': True,
                'probability': 0.0,
                'evidence': 'Failed to decode frame image from base64'
            }
        
        if reference is None:
            return {
                'id': FACE_RECOGNITION_MODULE_ID,
                'timestamp': datetime.now().isoformat(),
                'flag': True,
                'probability': 0.0,
                'evidence': 'Failed to decode reference image from base64'
            }
        
        return self.compare_faces(frame, reference)
    
    def get_face_embeddings(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Extract face embeddings from a frame.
        
        Args:
            frame: BGR image
            
        Returns:
            List of 512-D embedding vectors for each detected face
        """
        faces = self._detect_faces(frame)
        return [face.embedding for face in faces]
    
    def _detect_faces(self, frame: np.ndarray) -> List:
        """
        Detect faces in a frame using RetinaFace.
        
        Args:
            frame: BGR image
            
        Returns:
            List of face objects with bbox, landmarks, embedding, etc.
        """
        faces = self.face_analyzer.get(frame)
        # Filter by detection threshold
        faces = [f for f in faces if f.det_score >= self.detection_threshold]
        return faces
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        # Normalize embeddings
        e1 = emb1.flatten() / (np.linalg.norm(emb1) + 1e-10)
        e2 = emb2.flatten() / (np.linalg.norm(emb2) + 1e-10)
        
        # Compute cosine similarity
        similarity = np.dot(e1, e2)
        
        # Clip to [0, 1] range (cosine similarity can be negative)
        return float(np.clip(similarity, 0.0, 1.0))
    
    @staticmethod
    def _decode_base64_image(base64_string: str) -> Optional[np.ndarray]:
        """
        Decode a base64 string to an OpenCV image.
        
        Args:
            base64_string: Base64 encoded image
            
        Returns:
            BGR image as numpy array, or None if decoding fails
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(base64_string)
            
            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return image
        
        except Exception:
            return None
        
        except Exception:
            return None
