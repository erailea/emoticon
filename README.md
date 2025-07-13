# Emotion Analysis API

A Python FastAPI application that analyzes emotions from images at specific timestamps. This application provides a RESTful API for emotion detection using deep learning models.

## Features

- **Session Management**: Start, manage, and end analysis sessions
- **Emotion Detection**: Analyze emotions from images using DeepFace
- **Multiple Emotions**: Detect happiness, sadness, anger, surprise, fear, disgust, and neutral
- **Timestamp Support**: Track emotions at specific timestamps
- **Persistent Storage**: Session data is stored in JSON files
- **Docker Support**: Fully containerized application

## API Endpoints

### 1. Start Session
```
GET /start
```
Creates a new analysis session and returns a session ID.

**Response:**
```json
{
  "sessionId": "uuid-string"
}
```

### 2. Upload Inputs
```
POST /
```
Upload timestamp and image pairs for analysis.

**Request Body:**
```json
{
  "sessionId": "uuid-string",
  "inputs": [
    {
      "timestamp": 1.5,
      "file": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
    }
  ]
}
```

**Response:**
```json
{
  "message": "Inputs uploaded successfully",
  "inputsCount": 3
}
```

### 3. End Session
```
GET /end?sessionId=uuid-string
```
Finalize the session and get emotion analysis results.

**Response:**
```json
{
  "sessionId": "uuid-string",
  "aspects": {
    "happy": [
      {
        "timestamp": 1.5,
        "value": 85.23
      }
    ],
    "sad": [
      {
        "timestamp": 1.5,
        "value": 12.45
      }
    ]
  }
}
```

### 4. Health Check
```
GET /health
```
Check if the API is running.

### 5. Root Endpoint
```
GET /
```
Get API information and available endpoints.

## Installation and Setup

### Option 1: Docker (Recommended)

1. Build the Docker image:
```bash
docker build -t emotion-analyzer .
```

2. Run the container:
```bash
docker run -p 8000:8000 emotion-analyzer
```

The API will be available at `http://localhost:8000`

### Option 2: Local Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

## Testing

Use the provided test script to test all endpoints:

```bash
# Make sure the API is running first
python main.py

# In another terminal, run the test
python test_api.py
```

The test script will:
- Test all API endpoints
- Use the provided `test.jpg` image
- Display detailed results
- Check session persistence

## File Structure

```
emotion-analyzer/
├── main.py              # Main FastAPI application
├── test_api.py          # API test script
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── README.md           # This file
├── test.jpg            # Test image
└── storage/            # Session data storage
    └── .gitkeep        # Directory placeholder
```

## Dependencies

- **FastAPI**: Web framework for building APIs
- **DeepFace**: Facial emotion analysis
- **OpenCV**: Computer vision library
- **TensorFlow**: Deep learning framework
- **Uvicorn**: ASGI server

## Usage Example

1. Start a session:
```bash
curl -X GET "http://localhost:8000/start"
```

2. Upload images with timestamps:
```bash
curl -X POST "http://localhost:8000/" \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "your-session-id",
    "inputs": [
      {
        "timestamp": 1.5,
        "file": "data:image/jpeg;base64,..."
      }
    ]
  }'
```

3. Get results:
```bash
curl -X GET "http://localhost:8000/end?sessionId=your-session-id"
```

## Technical Details

- **Emotion Detection**: Uses DeepFace library with pre-trained models
- **Image Processing**: Supports base64 encoded images
- **Storage**: Simple file-based JSON storage in the `storage/` directory
- **Session Management**: In-memory active sessions with file persistence
- **Error Handling**: Comprehensive error handling with appropriate HTTP status codes

## Development

To extend or modify the application:

1. **Add new emotions**: Modify the `analyze_emotions` function
2. **Change storage**: Replace file-based storage with database
3. **Add authentication**: Implement session-based or token-based auth
4. **Improve performance**: Add caching or async processing

## License

This project is provided as-is for educational and development purposes. 