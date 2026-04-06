# 🚀 Getting Started with RAG Chatbot Frontend

## Prerequisites

1. **Python 3.8+** installed
2. **Virtual environment activated**
3. **API keys configured** (Groq or Gemini)

## Setup (One-time)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs Flask, CORS, and all RAG components.

### Step 2: Configure API Keys

Create a `.env` file:

```bash
cp .env.example .env
```

Then edit `.env` and add your API key:

**Option A: Using Groq** (Recommended - faster & free)
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

### Method 1: Automatic Startup Script (Easiest)

**On Linux/macOS:**
```bash
chmod +x start.sh  # Make script executable (first time only)
./start.sh
```

**On Windows:**
```cmd
start.bat
```

This will:
1. ✅ Check dependencies
2. ✅ Start Flask backend
3. ✅ Open frontend in browser
4. ✅ Load sample documents automatically

### Method 2: Manual Startup

**Terminal 1 - Start Backend:**
```bash
python app.py
```

Expected output:
```
🤖 LLMClient ready — provider: GROQ, model: 'mixtral-8x7b-32768', temperature: 0.7
 * Running on http://0.0.0.0:5000
```

**Terminal 2 - Open Frontend:**
- Open `frontend.html` in your browser
- Or run: `open frontend.html` (macOS) / `xdg-open frontend.html` (Linux)

## Using the Chatbot

### 1. **Load Documents**
- Click **"📥 Ingest Documents"** to load documents from the `documents/` folder
- If folder is empty, sample documents auto-load
- System shows "Documents loaded ✓"

### 2. **Ask Questions**
- Type a question in the input field
- Press Enter or click Send
- System will:
  - Search documents for relevant context
  - Generate an answer using the LLM
  - Show source documents

### 3. **Add Your Own Documents**
- Place PDF, DOCX, or TXT files in the `documents/` folder
- Click "📥 Ingest Documents" to reload
- Ask about your documents!

### 4. **Reset**
- Click **"🗑️ Clear Database"** to delete all documents
- Requires confirmation

## Troubleshooting

### Issue: "Cannot GET /health"
**Problem**: Browser trying to open backend URL directly
**Solution**: Open `frontend.html` (the HTML file), not the backend API

### Issue: Connection refused / Backend not running
```bash
# Terminal shows error when starting Flask
Error: GROQ_API_KEY is not set
```
**Solution**: 
1. Edit `.env` file
2. Add your actual API key
3. Restart Flask with `python app.py`

### Issue: "No Document Provider"
**Solution**: 
1. Copy `.env.example` to `.env` if missing
2. Set up API key configuration
3. Restart the system

### Issue: CORS errors in browser console
**Solution**: These are normal if you're opening `frontend.html` locally. The system handles it.

## How It Works

```
┌─────────────────────────────────────────┐
│        Your Question (Browser)          │
└──────────────────┬──────────────────────┘
                   │ HTTPS/JSON
                   ▼
        ┌──────────────────────┐
        │   Flask Backend      │
        │  (app.py)            │
        └──────────────────────┘
             │         │
             ▼         ▼
        Embeddings  LLM (Groq)
             │         │
             ▼         ▼
        ┌──────────────────────┐
        │   ChromaDB Vector    │ ◄─── Documents (documents/)
        │      Store           │
        └──────────────────────┘
```

### The Pipeline

1. **Question**: You ask something in the UI
2. **Embed**: Question converted to vector using SentenceTransformer
3. **Retrieve**: Find 3 most similar documents in vector store
4. **Generate**: LLM (Groq/Gemini) writes answer using retrieved context
5. **Display**: Answer shown with source documents

## Project Structure

```
ETT_Project/
├── app.py                  # ← Flask API server (run this!)
├── frontend.html           # ← Open this in browser
├── start.sh                # ← Run on Linux/macOS
├── start.bat               # ← Run on Windows
├── STARTUP_GUIDE.md        # ← You are here
├── FRONTEND_GUIDE.md       # ← Detailed API docs
│
├── main.py                 # Original CLI version
├── vector_store.py         # ChromaDB wrapper
├── embeddings.py           # SentenceTransformer wrapper
├── llm_client.py           # Groq/Gemini wrapper
├── document_loader.py      # PDF/DOCX/TXT loader
├── config.py               # Configuration
│
├── documents/              # 📁 Place your files here
│   └── (PDF, DOCX, TXT)
│
├── chroma_db/              # Persistent vector database
├── .env                    # Your API keys (create from .env.example)
└── requirements.txt        # Python dependencies
```

## Environment Variables

Key `.env` settings:

```bash
# LLM Configuration
LLM_PROVIDER=groq           # "groq" or "gemini"
LLM_MODEL=mixtral-8x7b-32768
LLM_TEMPERATURE=0.7

# API Keys
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key

# Vector Database
CHROMA_COLLECTION=rag_collection
CHROMA_PERSIST_DIR=./chroma_db

# Document Processing
DOCUMENTS_DIR=./documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Tips & Tricks

### Faster Responses
- Use Groq (faster than Gemini)
- Ask specific questions
- Reduce `top_k` in code (default: 3)

### Better Answers
- Use well-formatted documents
- Add more relevant documents
- Ask follow-up questions

### Batch Testing
- Edit `main.py` to run multiple queries at once for testing

## API Reference

See [FRONTEND_GUIDE.md](FRONTEND_GUIDE.md) for full API documentation.

Quick endpoints:
- `POST /api/initialize` - Initialize system
- `POST /api/ingest` - Load documents
- `POST /api/query` - Ask questions
- `POST /api/clear` - Delete documents
- `GET /health` - Check status

## Common Errors & Solutions

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'flask'` | Flask not installed | `pip install -r requirements.txt` |
| `GROQ_API_KEY not set` | Missing API key | Edit `.env` with your key |
| `Connection refused` | Backend not running | Run `python app.py` |
| `Empty documents` | No files in `documents/` | Add files or use samples |
| `CORS error` | Wrong URL opened | Open `frontend.html` not backend |

## Next Steps

1. ✅ Run `./start.sh` (or `start.bat`)
2. ✅ See it open automatically
3. ✅ Click "📥 Ingest Documents"
4. ✅ Ask a question!
5. ✅ Add your own documents to `documents/` folder

## Support

- Check [FRONTEND_GUIDE.md](FRONTEND_GUIDE.md) for API details
- Review [README.md](README.md) for project info
- Check Flask terminal output for errors
- Browser console (F12) shows connection issues

---

**Ready to chat?** Run `./start.sh` or `start.bat` and start asking questions! 🚀
