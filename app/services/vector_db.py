"""Qdrant vector database service for face storage and search"""
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue

from app.config import QDRANT_HOST, QDRANT_PORT, VECTOR_SIZE, KNOWN_THRESHOLD


@dataclass
class MatchResult:
    """Result from searching for similar face"""
    face_id: str
    name: str
    image_path: str
    score: float


class VectorDBService:
    """Service for storing and searching face embeddings in Qdrant"""
    
    COLLECTION_NAME = "known_faces"
    
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self._connected = False
    
    def connect(self) -> bool:
        """Connect to Qdrant and ensure collection exists"""
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            
            # Create collection if not exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.COLLECTION_NAME not in collection_names:
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
            
            self._connected = True
            return True
            
        except Exception as e:
            print(f"Failed to connect to Qdrant: {e}")
            self._connected = False
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to Qdrant"""
        return self._connected and self.client is not None
    
    def add_known_face(
        self, 
        name: str, 
        embedding: List[float], 
        image_path: str
    ) -> str:
        """Add a known face to the database."""
        if not self.is_connected():
            raise RuntimeError("Not connected to Qdrant")
        
        face_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        point = PointStruct(
            id=face_id,
            vector=embedding,
            payload={
                "name": name,
                "image_path": image_path,
                "created_at": created_at
            }
        )
        
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point]
        )
        
        return face_id
    
    def search_known(
        self, 
        embedding: List[float], 
        threshold: float = KNOWN_THRESHOLD,
        limit: int = 1
    ) -> Optional[MatchResult]:
        """Search for most similar known face."""
        if not self.is_connected():
            raise RuntimeError("Not connected to Qdrant")
        
        search_result = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=embedding,
            limit=limit,
            score_threshold=threshold
        )
        
        if search_result and search_result.points:
            result = search_result.points[0]
            return MatchResult(
                face_id=str(result.id),
                name=result.payload["name"],
                image_path=result.payload["image_path"],
                score=result.score
            )
        
        return None
    
    def get_all_known(self) -> List[Dict[str, Any]]:
        """Get all known faces from database."""
        if not self.is_connected():
            raise RuntimeError("Not connected to Qdrant")
        
        results = self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            limit=1000
        )
        
        faces = []
        for point in results[0]:
            faces.append({
                "face_id": str(point.id),
                "name": point.payload["name"],
                "image_path": point.payload["image_path"],
                "created_at": point.payload["created_at"]
            })
        
        return faces
    
    def get_count(self) -> int:
        """Get count of known faces."""
        if not self.is_connected():
            return 0
        
        return self.client.count(
            collection_name=self.COLLECTION_NAME
        ).count
    
    def delete_face(self, face_id: str) -> bool:
        """Delete a face by ID."""
        if not self.is_connected():
            raise RuntimeError("Not connected to Qdrant")
        
        try:
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=[face_id]
            )
            return True
        except Exception as e:
            print(f"Failed to delete face: {e}")
            return False
    
    def clear_all(self) -> bool:
        """Clear all faces from collection."""
        if not self.is_connected():
            raise RuntimeError("Not connected to Qdrant")
        
        try:
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="id", match=MatchValue(value="*"))]
                )
            )
            return True
        except Exception:
            # Alternative approach - recreate collection
            try:
                self.client.delete_collection(self.COLLECTION_NAME)
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                return True
            except Exception as e:
                print(f"Failed to clear collection: {e}")
                return False


# Singleton instance
vector_db = VectorDBService()