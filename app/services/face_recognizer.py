"""Face recognition service using DeepFace (more reliable than InsightFace GPU)"""
import os
import uuid
from pathlib import Path
from typing import List, Optional
import numpy as np

from deepface import DeepFace

from app.config import MODEL_NAME, DETECTOR_BACKEND
from app.models.schemas import FaceResult


class FaceRecognizerService:
    """Service for extracting face embeddings and detecting faces using DeepFace"""
    
    def __init__(self):
        self.model_name = MODEL_NAME
        self.detector_backend = DETECTOR_BACKEND
        print("[FaceRecognizer] Using DeepFace with ArcFace model")
    
    def extract_faces(self, image_path: str) -> List[FaceResult]:
        """Extract all faces from an image using DeepFace."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        faces = []
        
        try:
            # Use DeepFace for detection + embedding
            # This is more reliable than InsightFace with GPU
            results = DeepFace.represent(
                img_path=image_path,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False  # Don't fail if no face found
            )
            
            for idx, result in enumerate(results):
                embedding = result.get("embedding")
                facial_area = result.get("facial_area", {})
                
                if embedding is None:
                    continue
                
                x = facial_area.get("x", 0)
                y = facial_area.get("y", 0)
                w = facial_area.get("w", 0)
                h = facial_area.get("h", 0)
                
                face = FaceResult(
                    face_id=str(uuid.uuid4()),
                    image_path=image_path,
                    bbox={"x": x, "y": y, "w": w, "h": h},
                    embedding=embedding if isinstance(embedding, list) else embedding.tolist(),
                    is_known=False,
                    name=None,
                    confidence=float(result.get("face_confidence", 1.0))
                )
                faces.append(face)
                
        except Exception as e:
            print(f"[FaceRecognizer] Error processing {image_path}: {e}")
            
        return faces
    
    def get_embedding(self, image_path: str) -> Optional[List[float]]:
        """Get embedding for a single face in an image."""
        faces = self.extract_faces(image_path)
        if faces:
            return faces[0].embedding
        return None
    
    def extract_faces_from_directory(self, directory_path: str, recursive: bool = True) -> List[FaceResult]:
        """Extract all faces from all images in a directory."""
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        all_faces = []
        
        directory = Path(directory_path)
        
        if recursive:
            image_files = [
                f for f in directory.rglob("*") 
                if f.is_file() and f.suffix.lower() in supported_extensions
            ]
        else:
            image_files = [
                f for f in directory.iterdir() 
                if f.is_file() and f.suffix.lower() in supported_extensions
            ]
        
        print(f"[FaceRecognizer] Found {len(image_files)} images in {directory_path}")
        print(f"[FaceRecognizer] Using DeepFace + ArcFace + RetinaFace")
        
        total = len(image_files)
        for idx, image_file in enumerate(image_files):
            if idx % 5 == 0 or idx == total - 1:
                pct = ((idx + 1) / total) * 100
                print(f"[{idx+1}/{total}] {pct:.0f}% - {image_file.name}")
            try:
                faces = self.extract_faces(str(image_file))
                if faces:
                    print(f"  -> Found {len(faces)} face(s)")
                all_faces.extend(faces)
            except Exception as e:
                print(f"  -> ERROR: {e}")
        
        print(f"[FaceRecognizer] Total: {len(all_faces)} faces from {len(image_files)} images")
                
        return all_faces
    
    def extract_face_embedding_only(self, image_path: str) -> Optional[np.ndarray]:
        """Extract embedding for first detected face."""
        try:
            results = DeepFace.represent(
                img_path=image_path,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False
            )
            if results and len(results) > 0:
                embedding = results[0].get("embedding")
                if embedding is not None:
                    if isinstance(embedding, list):
                        return np.array(embedding)
                    return embedding
        except Exception as e:
            print(f"[FaceRecognizer] Error extracting embedding: {e}")
            
        return None


face_recognizer = FaceRecognizerService()