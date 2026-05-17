# 🧠 CV RAG API

A **Retrieval-Augmented Generation (RAG)** API that lets you ask HR-style questions against a pool of CVs stored in [Pinecone](https://www.pinecone.io/). Built with **FastAPI**, **LangChain**, **Sentence Transformers**, and powered by **GPT-4o-mini** via [OpenRouter](https://openrouter.ai/).

---

## ✨ Features

- 🔍 **Semantic Search** over CVs using `BAAI/bge-small-en-v1.5` embeddings stored in Pinecone
- 🤖 **LLM-Powered Answers** via GPT-4o-mini through OpenRouter
- ⚡ **Streaming Support** — get token-by-token answers for a chat-like experience
- 🖥️ **Streamlit UI** — a simple browser interface to interact with the API
- 🐳 **Docker Ready** — single command to run the whole application
- 📄 **OpenAPI Docs** — auto-generated Swagger UI at `/docs`

---

## 🏗️ Architecture

```
User Query
    │
    ▼
FastAPI (/ask or /ask-stream)
    │
    ├──► PineconeRetriever  ──► Pinecone Vector DB (Top-K CV chunks)
    │                                  │
    │              ◄────────────────────┘
    │
    ├──► LangChain RAG Chain
    │         ├── ChatPromptTemplate  (system + human)
    │         ├── ChatOpenAI          (OpenRouter / GPT-4o-mini)
    │         └── StrOutputParser
    │
    ▼
JSON Answer  (or streamed text)
```

---

## 📁 Project Structure

```
app/
├── main.py           # FastAPI app — routes & lifespan
├── rag.py            # RAG pipeline (retriever, chain, helpers)
├── ui.py             # Streamlit frontend
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container definition
├── .env              # API keys (not committed)
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A [Pinecone](https://www.pinecone.io/) account with an index named **`cv-rag`**
- An [OpenRouter](https://openrouter.ai/) API key
- Your CV documents already embedded and upserted into the Pinecone index

### 1 — Clone the repository

```bash
git clone https://github.com/Noura-Darwazeh/cv-rag-api.git
cd cv-rag-api
```

### 2 — Set up environment variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Run the API

```bash
uvicorn main:app --reload
```

The API will be available at **`http://localhost:8000`**.

### 5 — (Optional) Run the Streamlit UI

In a separate terminal:

```bash
streamlit run ui.py
```

---

## 🐳 Docker

Build and run the entire application in a container:

```bash
# Build the image
docker build -t cv-rag-api .

# Run the container (pass your .env file)
docker run --env-file .env -p 8000:8000 cv-rag-api
```

---

## 📡 API Reference

### `GET /health`
Quick status check to confirm the app is running.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /ask`
Run the full RAG pipeline and return a JSON answer.

**Request Body:**
```json
{
  "question": "Who has Python experience?",
  "show_context": false
}
```

| Field          | Type    | Default | Description                                    |
|----------------|---------|---------|------------------------------------------------|
| `question`     | string  | —       | The HR question to ask                         |
| `show_context` | boolean | `false` | Include the retrieved CV chunks in the response |

**Response:**
```json
{
  "question": "Who has Python experience?",
  "answer": "Based on the CVs, candidates with Python experience include...",
  "context": null
}
```

---

### `POST /ask-stream`
Same as `/ask` but streams the answer tokens as they are generated — ideal for building a real-time chat UI.

**Request Body:** Same as `/ask`

**Response:** `text/plain` streamed response

---

## 🌐 Interactive Docs

Once the server is running, visit:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## ⚙️ Configuration

All model/index settings are defined at the top of `rag.py`:

| Variable           | Default Value           | Description                        |
|--------------------|-------------------------|------------------------------------|
| `EMBEDDING_MODEL`  | `BAAI/bge-small-en-v1.5`| Sentence Transformer model         |
| `PINECONE_INDEX`   | `cv-rag`                | Name of your Pinecone index        |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini`    | LLM model via OpenRouter           |
| `RETRIEVER_TOP_K`  | `5`                     | Number of CV chunks to retrieve    |

---

## 🛠️ Tech Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| API         | FastAPI + Uvicorn                   |
| RAG         | LangChain Core                      |
| Embeddings  | Sentence Transformers (BAAI/bge)    |
| Vector DB   | Pinecone                            |
| LLM         | GPT-4o-mini via OpenRouter          |
| UI          | Streamlit                           |
| Container   | Docker                              |

---

## 📝 License

This project is for educational purposes.
