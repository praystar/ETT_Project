# Setup Guide: Google Gemini API Integration

## What Changed

Your project has been restructured to use **Google's Gemini 2.5 Pro API** (free tier) instead of OpenAI:

### File Updates:
- **config.py**: Replaced `OPENAI_API_KEY` with `GEMINI_API_KEY`, updated default model to `gemini-2.5-pro`
- **llm_client.py**: Replaced OpenAI SDK with `google-generativeai` SDK
- **requirements.txt**: Replaced `openai` dependency with `google-generativeai`

### What Stays the Same:
- **embeddings.py**: Still uses local `sentence-transformers` (free, no API key needed)
- **vector_store.py**: ChromaDB unchanged
- **main.py**: No changes needed (uses the LLMClient abstraction)

---

## Setup Steps

### 1. Get a Free Gemini API Key

Go to **[Google AI Studio](https://aistudio.google.com/app/apikey)** and click "Create API Key"
- Sign in with your Google account (free)
- Copy your API key

### 2. Create `.env` File

```bash
cp .env.example .env
```

Then edit `.env` and paste your API key:
```
GEMINI_API_KEY=your-actual-api-key-here
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Project

```bash
python main.py
```

---

## Available Gemini Models

You can change the model in your `.env` file:

- **`gemini-2.5-pro`** ⭐ Recommended - Latest, most capable
- `gemini-1.5-pro` - High capability
- `gemini-1.5-flash` - Faster, lighter (good for testing)
- `gemini-1.5-pro-latest` - Latest stable

---

## Key Differences from OpenAI

| Feature | OpenAI | Gemini |
|---------|--------|--------|
| API Key | `OPENAI_API_KEY` | `GEMINI_API_KEY` |
| SDK | `openai` | `google-generativeai` |
| Chat Roles | user, assistant, system | user, model |
| Response Field | `response.choices[0].message.content` | `response.text` |

---

## Troubleshooting

**"API key not valid"**
- Check your `.env` file has the correct key
- Make sure `.env` is in the project root

**"Module not found: google.generativeai"**
- Run `pip install -r requirements.txt` again
- Check you're using the correct Python environment

**Rate Limiting**
- Gemini free tier has generous limits (~15 requests per minute for most models)
- Upgrade to paid plan if needed for higher limits

---

## Keeping Embeddings Local & Free

The project uses `sentence-transformers` for embeddings (downloaded once, then cached).
- No API call needed
- No cost whatsoever
- Works completely offline after first download

---

## Next Steps

- Test with: `python main.py`
- Modify documents in `main.py` → `ingest_documents()` function
- Ask questions to test the RAG pipeline!
