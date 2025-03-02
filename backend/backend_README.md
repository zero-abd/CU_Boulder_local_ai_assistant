# BuffAdvisor Flask Server

This Flask server connects the frontend React application with the BuffAdvisor chatbot system. It provides API endpoints to process user queries and return AI-generated responses.

## Prerequisites

- Python 3.8+
- Flask and other required packages
- A PDF file containing CU Boulder information (for the chatbot's knowledge base)

## Installation

1. Install the required Python packages:

```bash
pip install flask flask-cors
```

2. Make sure you have all the dependencies required by the `bot.py` file:

```bash
pip install langchain langchain_ollama langchain_community python-multipart pymupdf faiss-cpu
```

3. Install Ollama if you haven't already (for running the local LLM):
   - Follow instructions at [https://ollama.com/](https://ollama.com/)
   - Make sure to pull the models mentioned in the `bot.py` file:
     ```bash
     ollama pull gemma2:2b
     ollama pull nomic-embed-text
     ```

## Running the Server

1. Place a PDF file containing CU Boulder information in the same directory as the server.

2. Start the Flask server:

```bash
python server.py
```

The server will run on port 5000 by default.

## API Endpoints

The server provides the following endpoints:

### 1. Health Check

```
GET /api/health
```

Returns a simple status check to confirm the API is running.

### 2. Status Check

```
GET /api/status
```

Checks if the BuffAdvisor system is initialized and ready to process queries.

### 3. Chat

```
POST /api/chat
```

Process a user query and return an AI-generated response.

Request body:
```json
{
  "message": "What programs does CU Boulder offer?",
  "style": "balanced",
  "new_session": false
}
```

Parameters:
- `message`: The user's query (required)
- `style`: Response style - "balanced", "brief", "detailed", or "supportive" (optional, default: "balanced")
- `new_session`: Whether to start a new conversation session (optional, default: false)

### 4. Manual Initialization

```
POST /api/initialize
```

Manually initialize the BuffAdvisor system with a specific PDF file.

Request body:
```json
{
  "pdf_path": "/path/to/cuboulderdocs.pdf"
}
```

## Integration with Frontend

This server is designed to work with the React frontend in the `frontend` directory. The frontend makes API calls to these endpoints to interact with the BuffAdvisor system.

The frontend API client at `frontend/src/api/buffAdvisor.js` is already configured to communicate with this server.

## Error Handling

The server provides appropriate error responses with descriptive messages when:
- The BuffAdvisor system is not initialized
- No PDF file is found
- An empty message is provided
- Any other error occurs during processing

## Development Notes

- The server runs in debug mode by default, suitable for development but not for production
- For production deployment, consider using a WSGI server like Gunicorn 