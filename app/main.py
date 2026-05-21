"""FastAPI application for Face Recognition"""
import os
import sys

# Fix Unicode for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from app.config import KNOWN_FACES_DIR
from app.models.schemas import (
    ScanResult, 
    AddKnownFacesResponse, 
    KnownFacesListResponse,
    KnownFaceInfo,
    HealthResponse,
    FaceResult,
    PersonCluster
)
from app.services.face_recognizer import face_recognizer
from app.services.vector_db import vector_db
from app.services.clustering import clustering_service

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="Web application for face recognition and clustering",
    version="1.0.0"
)

# Setup templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Store last scan result
last_scan_result: Optional[ScanResult] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Ensure directories exist
    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Connect to Qdrant
    vector_db.connect()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page"""
    return templates.TemplateResponse("index.html", {
        "request": request
    })


@app.get("/results", response_class=HTMLResponse)
async def results(request: Request):
    """Results page"""
    return templates.TemplateResponse("results.html", {
        "request": request,
        "result": last_scan_result
    })


@app.get("/api/debug/paths")
async def debug_paths():
    """Debug path info"""
    import os
    return {
        "cwd": os.getcwd(),
        "test_data_exists": os.path.exists("test_data"),
        "test_data_abs": os.path.abspath("test_data"),
        "data_exists": os.path.exists("data"),
    }

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check"""
    qdrant_connected = vector_db.is_connected()
    known_count = vector_db.get_count() if qdrant_connected else 0
    
    return HealthResponse(
        status="ok" if qdrant_connected else "warning",
        qdrant_connected=qdrant_connected,
        known_faces_count=known_count
    )


@app.post("/api/known/add", response_model=AddKnownFacesResponse)
async def add_known_faces(
    files: list[UploadFile] = File(...),
    person_name: str = Form(...)
):
    """Add known faces to database"""
    if not person_name:
        raise HTTPException(status_code=400, detail="Person name is required")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    faces_count = 0
    person_dir = KNOWN_FACES_DIR / person_name.replace(" ", "_")
    person_dir.mkdir(parents=True, exist_ok=True)
    
    for upload_file in files:
        # Save uploaded file
        file_path = person_dir / upload_file.filename
        content = await upload_file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Extract face embedding
        embedding = face_recognizer.extract_face_embedding_only(str(file_path))
        
        if embedding is not None:
            # Add to Qdrant
            vector_db.add_known_face(
                name=person_name,
                embedding=embedding.tolist(),
                image_path=str(file_path)
            )
            faces_count += 1
    
    return AddKnownFacesResponse(
        status="ok",
        faces_count=faces_count,
        person_name=person_name
    )


@app.get("/api/known/list", response_model=KnownFacesListResponse)
async def list_known_faces():
    """List all known faces"""
    faces = vector_db.get_all_known()
    
    known_faces = [
        KnownFaceInfo(
            face_id=f["face_id"],
            name=f["name"],
            image_path=f["image_path"],
            created_at=f["created_at"]
        )
        for f in faces
    ]
    
    return KnownFacesListResponse(
        faces=known_faces,
        total=len(known_faces)
    )


@app.post("/api/scan", response_model=ScanResult)
async def scan_directory(directory_path: str = Form(...)):
    """Scan directory for faces"""
    global last_scan_result
    
    print(f"[API] Starting scan of directory: {directory_path}")
    
    if not os.path.exists(directory_path):
        raise HTTPException(status_code=400, detail="Directory not found")
    
    if not os.path.isdir(directory_path):
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    # Count only image files (recursive)
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    directory = Path(directory_path)
    image_files = [
        f for f in directory.rglob("*") 
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]
    total_images = len(image_files)
    
    print(f"[API] Found {total_images} image files (recursive)")
    
    # Extract all faces from directory (recursive)
    print(f"[API] Starting face extraction...")
    faces = face_recognizer.extract_faces_from_directory(directory_path, recursive=True)
    print(f"[API] Extracted {len(faces)} faces")
    
    # Separate known and unknown
    known_faces = []
    unknown_embeddings = []
    unknown_image_paths = []
    
    print(f"[API] Processing {len(faces)} faces...")
    for face in faces:
        # Search in database
        match = vector_db.search_known(face.embedding)
        
        if match:
            face.is_known = True
            face.name = match.name
            face.confidence = match.score
            known_faces.append(face)
            print(f"[API] Known: {match.name} ({match.score:.2f})")
        else:
            unknown_embeddings.append(face.embedding)
            unknown_image_paths.append(face.image_path)
    
    print(f"[API] Known faces: {len(known_faces)}, Unknown: {len(unknown_embeddings)}")
    
    # Cluster unknown faces
    unknown_clusters = clustering_service.cluster_unknown_faces(
        unknown_embeddings,
        unknown_image_paths
    )
    
    print(f"[API] Found {len(unknown_clusters)} unknown clusters")
    
    # Build result
    result = ScanResult(
        total_images=total_images,
        total_faces=len(faces),
        known_faces=known_faces,
        unknown_clusters=unknown_clusters
    )
    
    last_scan_result = result
    print(f"[API] Scan complete!")
    
    return result


@app.get("/api/results", response_model=Optional[ScanResult])
async def get_last_results():
    """Get last scan result"""
    return last_scan_result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)