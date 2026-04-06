# 🚀 Getting Started with RAG Chatbot

## Prerequisites

1. **Python 3.8+** installed
2. **Virtual environment activated**
3. **API keys configured** (Groq or Gemini)

## Setup (One-time)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Configure API Keys

Create a `.env` file:

```bash
cp .env.example .env
```

Then edit `.env` and add your API key:

**Option A: Using Groq** (Recommended - faster)
```
LLM_PROVIDER=groq
LLM_MODEL=mixtral-8x7b-32768
GROQ_API_KEY=your_groq_api_key_here
```

**Option B: Using Gemini**
```
LLM_PROVIDER=gemini
LLM_MODEL=gemini-1.5-pro
GEMINI_API_KEY=your_gemini_api_key_here
```

## Running the Chatbot

### Automatic Startup (Recommended)

**On Linux/macOS:**
```bash
chmod +x start.sh  # First time only
./start.sh
```

**On Windows:**
```cmd
start.bat
```

This will:
- ✅ Check dependencies
- ✅ Start Flask backend
- ✅ Open frontend in browser
- ✅ Load sample documents

### Manual Startup

**Terminal 1 - Start Backend:**
```bash
cd backend
python app.py
```

**Terminal 2 - Open Frontend:**
- Open `frontend/index.html` in your browser

## Using the Chatbot

### 1. Load Documents
- Click **"Ingest Documents"** to load from `documents/` folder
- Or sample documents will auto-load

### 2. Ask Questions
- Type a question
- Press Enter
- View answer with sources

### 3. Add Your Documents
- Place PDF/DOCX/TXT files in `documents/` folder
- Click "Ingest Documents" to reload

### 4. Reset
- Click "Clear Database" to delete all documents

## Project Structure

```
ETT_Project/
├── backend/                    # Python modules
│   ├── app.py                 # Flask API
│   ├── config.py
│   ├── vector_store.py
│   ├── embeddings.py
│   ├── llm_client.py
│   ├── document_loader.py
│   └── main.py                # CLI version
├── frontend/
│   └── index.html             # Web UI
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── documents/                  # Your documents
├── chroma_db/                  # Vector DB
└── requirements.txt
```

## Troubleshooting

### Connection refused
- Ensure `backend/app.py` is running
- Check `http://localhost:5000/health`

### API key errors
- Edit `.env` with valid API key
- Restart backend (`python app.py`)

### No documents loading
- Add files to `documents/` folder
- Or click "Ingest Documents" for samples

## Support

- Check `docs/API.md` for API details
- Check `docs/DEVELOPMENT.md` for dev info
