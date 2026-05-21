"""Configuration for Face Recognition Application"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWN_FACES_DIR = DATA_DIR / "known_faces"

# Qdrant settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))

# DeepFace settings
MODEL_NAME = os.getenv("MODEL_NAME", "ArcFace")
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "retinaface")

# Search and clustering thresholds
KNOWN_THRESHOLD = float(os.getenv("KNOWN_THRESHOLD", "0.6"))
CLUSTERING_EPS = float(os.getenv("CLUSTERING_EPS", "0.4"))
CLUSTERING_MIN_SAMPLES = int(os.getenv("CLUSTERING_MIN_SAMPLES", "3"))

# Vector size for ArcFace
VECTOR_SIZE = 512

# Create directories if not exist
KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)