# BuffAdvisor

**A CU Boulder campus assistant that runs a quantized DeepSeek-R1-Distill-Llama-8B model entirely on a local AMD AI PC.** Built at HackCU 2025 (HackCU 11), where it took 1st place in the AMD AI PC track.

**Live demo: https://buffadvisor.vercel.app**

Students ask about programs, resources and campus life by typing or speaking. BuffAdvisor retrieves the relevant passages from a CU Boulder information PDF and answers with DeepSeek-R1-Distill-Llama-8B, quantized with AMD Quark and run through ONNX Runtime GenAI on the local machine. No cloud model is involved.

---

## What it does

- **Local LLM inference.** `DeepSeek-R1-Distill-Llama-8B` quantized with AMD Quark (`amd_llm_quantization.ipynb`) and served with `onnxruntime-genai` on hardware acceleration, falling back to CPU if the device can't schedule the model.
- **Retrieval over campus documents.** The PDF is extracted with PyMuPDF, split into 1,000-character chunks, embedded with `nomic-embed-text` through Ollama and indexed in FAISS. Each question pulls the closest chunks into the prompt.
- **Answer styles.** Balanced, brief, detailed or supportive, chosen per request.
- **Streaming responses.** The Flask API streams tokens over Server-Sent Events, with a non-streaming mode that times out after 120 seconds.
- **Voice UI.** The React frontend supports speech input and spoken replies through the Web Speech API. (The hackathon build also had a webcam face-detection greeting; its face-api model files were never committed, so it was removed.)

---

## Architecture

```mermaid
flowchart LR
  subgraph FE["frontend/ (React + Vite)"]
    UI["Chat UI<br/>voice in/out (Web Speech API)"]
  end
  subgraph BE["backend/ (Flask, :5001)"]
    API["/api/chat (SSE stream)<br/>/api/status, /api/health,<br/>/api/initialize"]
    RAG["BuffAdvisor<br/>PyMuPDF, text splitter,<br/>FAISS retriever, prompt"]
    M["DeepSeekModel<br/>onnxruntime-genai"]
  end
  PDF["CU Boulder info PDF"]
  EMB["Ollama<br/>nomic-embed-text"]
  Q["DeepSeek-R1-Distill-Llama-8B<br/>quantized with AMD Quark"]

  UI -- "fetch + SSE" --> API --> RAG
  PDF --> RAG
  RAG --> EMB
  RAG --> M --> Q
```

---

## Hosted demo

The real model needs an AMD AI PC, so the hosted page at https://buffadvisor.vercel.app (the same `frontend/`, deployed on Vercel) has two modes:

- **Sample answers** (default). Eight campus questions with sample answers. No transcripts from the hackathon run were saved, so these are written from the campus notes and labelled "sample", not presented as model output. The retrieval step runs live in the browser.
- **Live, your key.** Any question, answered with the visitor's own OpenAI, OpenRouter or Groq key (or any OpenAI-compatible endpoint). The browser retrieves the closest campus notes with BM25 keyword search (`frontend/src/rag/retrieve.js`) and sends them with the question using the prompt from `backend/bot.py`. For the listed providers the request passes through `frontend/api/chat.js`, a stateless relay that only forwards to those fixed hosts and never stores or logs the key; a custom base URL is called straight from the browser.

The original campus PDF is not in this repo. The hosted demo retrieves over short notes in `frontend/src/data/campusDocs.js`, summarized from public colorado.edu pages with a source link on each.

When `VITE_API_URL` is set, or under `npm run dev`, the chat also shows an **On-device backend** tab that streams from the Flask server below.

---

## Tech stack

- **Model and inference:** DeepSeek-R1-Distill-Llama-8B, AMD Quark (quantization), ONNX Runtime GenAI
- **Retrieval:** LangChain, FAISS, Ollama embeddings (`nomic-embed-text`), PyMuPDF
- **Backend:** Python, Flask, Flask-CORS
- **Frontend:** React 19, Vite 6, Tailwind CSS 4, Web Speech API

---

## Repository layout

```
amd_llm_quantization.ipynb   Colab notebook: install AMD Quark and quantize DeepSeek-R1-Distill-Llama-8B
backend/
  bot.py                     RAG pipeline, DeepSeek ONNX wrapper, interactive CLI
  server.py                  Flask API (chat with SSE streaming, status, health, initialize)
  run_model.py               minimal onnxruntime-genai generation loop
  requirements.txt
frontend/
  src/components/Chat.jsx    chat UI: sample, live (your key) and on-device modes, voice in/out
  src/api/buffAdvisor.js     clients for the Flask backend (SSE) and OpenAI-compatible APIs
  src/rag/retrieve.js        in-browser BM25 retrieval over the campus notes (hosted demo)
  src/data/                  campus notes and sample Q&A for the hosted demo
  api/chat.js                Vercel function: stateless relay for bring-your-own-key mode
```

---

## Run it locally

The backend was built for a Windows AMD AI PC with the quantized model exported for ONNX Runtime GenAI.

### 1. Quantize the model

Run `amd_llm_quantization.ipynb` (written for Google Colab) to install AMD Quark and quantize `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`. The notebook exports Hugging Face format; the backend loads an ONNX Runtime GenAI model folder, and that conversion step is not included in this repo. The model path is set in `backend/bot.py` (`model_path` in `initialize_advisor`, `LangChain` and `BuffAdvisor`); update it to your folder.

### 2. Backend

```bash
cd backend
pip install flask flask-cors
pip install -r requirements.txt
ollama pull nomic-embed-text      # embedding model, with Ollama running
```

Put a CU Boulder information PDF in `backend/` (the server uses the first `.pdf` it finds), then:

```bash
python server.py                  # http://localhost:5001
```

`python bot.py` starts an interactive terminal chat instead of the API.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev                       # http://localhost:5173
```

In dev mode the chat's **On-device backend** tab calls `http://localhost:5001/api`; set `VITE_API_URL` in `frontend/.env` to change it. Endpoint details are in [`backend/backend_README.md`](backend/backend_README.md).
