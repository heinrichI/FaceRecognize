"""Clustering service for grouping unknown faces using DBSCAN"""
from typing import List
import numpy as np
from sklearn.cluster import DBSCAN

from app.config import CLUSTERING_EPS, CLUSTERING_MIN_SAMPLES
from app.models.schemas import PersonCluster


class ClusteringService:
    """Service for clustering unknown faces using DBSCAN"""
    
    def __init__(self):
        self.eps = CLUSTERING_EPS
        self.min_samples = CLUSTERING_MIN_SAMPLES
    
    def cluster_unknown_faces(
        self,
        embeddings: List[List[float]],
        image_paths: List[str]
    ) -> List[PersonCluster]:
        """Cluster unknown faces using DBSCAN."""
        if not embeddings or not image_paths:
            return []
        
        X = np.array(embeddings)
        
        if len(X) == 1:
            return [PersonCluster(
                cluster_id=0,
                face_count=1,
                image_paths=[image_paths[0]],
                sample_embedding=embeddings[0]
            )]
        
        try:
            clustering = DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                metric='cosine'
            ).fit(X)
        except Exception as e:
            print(f"DBSCAN clustering failed: {e}")
            return []
        
        labels = clustering.labels_
        unique_labels = set(labels)
        
        clusters = []
        cluster_idx = 0
        
        for label in sorted(unique_labels):
            if label == -1:
                continue
            
            indices = [i for i, l in enumerate(labels) if l == label]
            
            if len(indices) < self.min_samples:
                continue
            
            cluster_images = [image_paths[i] for i in indices]
            cluster_embeddings = [embeddings[i] for i in indices]
            
            centroid = np.mean(cluster_embeddings, axis=0).tolist()
            
            clusters.append(PersonCluster(
                cluster_id=cluster_idx,
                face_count=len(indices),
                image_paths=cluster_images,
                sample_embedding=centroid
            ))
            
            cluster_idx += 1
        
        return clusters
    
    def get_noise_faces(
        self,
        embeddings: List[List[float]],
        image_paths: List[str]
    ) -> List[str]:
        """Get faces marked as noise by DBSCAN."""
        if not embeddings or not image_paths:
            return []
        
        X = np.array(embeddings)
        
        if len(X) == 1:
            return image_paths
        
        try:
            clustering = DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                metric='cosine'
            ).fit(X)
        except Exception:
            return image_paths
        
        labels = clustering.labels_
        noise_paths = [image_paths[i] for i, l in enumerate(labels) if l == -1]
        
        return noise_paths


clustering_service = ClusteringService()