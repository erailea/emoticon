from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import json
import os
import base64
from collections import defaultdict
from datetime import datetime
import shutil
import tempfile
from deepface import DeepFace
import cv2
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Emotion Analysis API", version="1.0.0")

# Storage directory
STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

# In-memory storage for active sessions
active_sessions = {}

# Pydantic models
class InputData(BaseModel):
    sessionId: str
    inputs: List[Dict[str, Any]]  # [{"timestamp": float, "file": str}]

class EmotionResult(BaseModel):
    timestamp: float
    value: float

class SessionResult(BaseModel):
    sessionId: str
    aspects: Dict[str, List[EmotionResult]]

def save_session(session_id: str, data: dict):
    """Save session data to file"""
    file_path = os.path.join(STORAGE_DIR, f"{session_id}.json")
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_session(session_id: str) -> dict:
    """Load session data from file"""
    file_path = os.path.join(STORAGE_DIR, f"{session_id}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def analyze_emotions(inputs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, float]]]:
    """Analyze emotions from input files"""
    results = defaultdict(list)
    
    for item in inputs:
        try:
            timestamp = item["timestamp"]
            file_data = item["file"]
            
            # Handle base64 encoded image
            if isinstance(file_data, str) and file_data.startswith('data:image'):
                # Extract base64 data
                base64_data = file_data.split(',')[1]
                image_bytes = base64.b64decode(base64_data)
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(image_bytes)
                    tmp_file_path = tmp_file.name
            else:
                # Assume it's a file path
                tmp_file_path = file_data
            
            # Analyze emotions using DeepFace
            try:
                analysis = DeepFace.analyze(
                    img_path=tmp_file_path,
                    actions=['emotion'],
                    enforce_detection=False
                )
                
                # Handle both single result and list of results
                if isinstance(analysis, list):
                    emotion_data = analysis[0]['emotion']
                else:
                    emotion_data = analysis['emotion']
                
                print(emotion_data)

                # Store results for each emotion
                for emotion, value in emotion_data.items():
                    results[emotion].append({
                        "timestamp": timestamp,
                        "value": value
                    })
                    
            except Exception as e:
                print(f"Error analyzing image: {e}")
                # If analysis fails, add default values
                default_emotions = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']
                for emotion in default_emotions:
                    results[emotion].append({
                        "timestamp": timestamp,
                        "value": 0.0
                    })
            
            # Clean up temporary file if created
            if 'tmp_file_path' in locals() and tmp_file_path != file_data:
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"Error processing input: {e}")
    
    return dict(results)

@app.get("/start")
async def start_session():
    """Start a new emotion analysis session"""
    session_id = str(uuid.uuid4())
    
    # Initialize session data
    session_data = {
        "sessionId": session_id,
        "inputs": [],
        "results": None,
        "created_at": datetime.now().isoformat(),
        "status": "active"
    }
    
    # Store in memory and persist to file
    active_sessions[session_id] = session_data
    save_session(session_id, session_data)
    
    return {"sessionId": session_id}

@app.post("/")
async def upload_inputs(data: InputData):
    """Upload timestamp and file pairs for analysis"""
    session_id = data.sessionId
    
    # Check if session exists
    if session_id not in active_sessions:
        # Try to load from file
        session_data = load_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        active_sessions[session_id] = session_data
    
    session = active_sessions[session_id]
    
    # Check if session is still active
    if session.get("status") != "active":
        raise HTTPException(status_code=400, detail="Session is not active")
    
    # Add inputs to session
    session["inputs"].extend(data.inputs)
    
    # Save updated session
    save_session(session_id, session)
    
    return {"message": "Inputs uploaded successfully", "inputsCount": len(session["inputs"])}

@app.get("/end")
async def end_session(sessionId: str):
    """End session and return emotion analysis results"""
    
    # Check if session exists
    if sessionId not in active_sessions:
        # Try to load from file
        session_data = load_session(sessionId)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        active_sessions[sessionId] = session_data
    
    session = active_sessions[sessionId]
    
    # Check if session has inputs
    if not session.get("inputs"):
        raise HTTPException(status_code=400, detail="No inputs found for this session")
    
    # Analyze emotions
    results = analyze_emotions(session["inputs"])
    
    # Update session with results
    session["results"] = results
    session["status"] = "completed"
    session["completed_at"] = datetime.now().isoformat()
    
    # Save final session
    save_session(sessionId, session)
    
    # Clean up from active sessions
    if sessionId in active_sessions:
        del active_sessions[sessionId]
    
    return {
        "sessionId": sessionId,
        "aspects": results
    }

@app.get("/sessions")
async def list_sessions():
    """List all stored sessions with their photo counts and session IDs"""
    try:
        sessions_info = []
        
        # Check if storage directory exists
        if not os.path.exists(STORAGE_DIR):
            return {
                "sessions": [],
                "total_sessions": 0,
                "message": "No sessions found"
            }
        
        # Get all session files
        session_files = [f for f in os.listdir(STORAGE_DIR) if f.endswith('.json')]
        
        for session_file in session_files:
            session_id = session_file[:-5]  # Remove .json extension
            
            try:
                session_data = load_session(session_id)
                if session_data and 'inputs' in session_data:
                    photo_count = len(session_data['inputs'])
                    creation_time = session_data.get('created_at', 'Unknown')
                    
                    sessions_info.append({
                        "sessionId": session_id,
                        "photo_count": photo_count,
                        "created_at": creation_time,
                        "has_results": 'aspects' in session_data
                    })
                else:
                    # Handle sessions without inputs (shouldn't happen but be safe)
                    sessions_info.append({
                        "sessionId": session_id,
                        "photo_count": 0,
                        "created_at": 'Unknown',
                        "has_results": False
                    })
            except Exception as e:
                # Skip corrupted session files
                print(f"Warning: Could not read session {session_id}: {e}")
                continue
        
        # Sort by creation time (most recent first) or session ID if no creation time
        sessions_info.sort(
            key=lambda x: x.get('created_at', ''), 
            reverse=True
        )
        
        return {
            "sessions": sessions_info,
            "total_sessions": len(sessions_info),
            "message": f"Found {len(sessions_info)} session(s)"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Emotion Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "start": "GET /start - Start new session",
            "upload": "POST / - Upload timestamp and file pairs",
            "end": "GET /end?sessionId={id} - End session and get results",
            "sessions": "GET /sessions - List all stored sessions",
            "health": "GET /health - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 