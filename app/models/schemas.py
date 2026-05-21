"""Pydantic models for Face Recognition API"""
from typing import List, Optional
from pydantic import BaseModel


class FaceResult(BaseModel):
    """Result of face detection and recognition"""
    face_id: str
    image_path: str
    bbox: dict
    embedding: List[float]
    is_known: bool = False
    name: Optional[str] = None
    confidence: Optional[float] = None


class PersonCluster(BaseModel):
    """Cluster of unknown faces (same person)"""
    cluster_id: int
    face_count: int
    image_paths: List[str]
    sample_embedding: List[float]


class ScanResult(BaseModel):
    """Result of directory scan"""
    total_images: int
    total_faces: int
    known_faces: List[FaceResult]
    unknown_clusters: List[PersonCluster]


class KnownFaceInfo(BaseModel):
    """Information about a known face in database"""
    face_id: str
    name: str
    image_path: str
    created_at: str


class AddKnownFacesRequest(BaseModel):
    """Request to add known faces"""
    person_name: str


class AddKnownFacesResponse(BaseModel):
    """Response after adding known faces"""
    status: str
    faces_count: int
    person_name: str


class ScanDirectoryRequest(BaseModel):
    """Request to scan directory"""
    directory_path: str


class KnownFacesListResponse(BaseModel):
    """List of all known faces"""
    faces: List[KnownFaceInfo]
    total: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    qdrant_connected: bool
    known_faces_count: int