

# 🧠 Semantic Question Search API

This is a Flask-based semantic search and question management tool integrated with Pinecone for vector storage. It allows you to add, search, and detect duplicate questions using vector embeddings generated from a sentence transformer model.

---

## 🚀 Features

- ✅ Add questions with embedding storage in Pinecone
- 🔍 Search semantically similar questions
- ❗ Detect duplicate questions based on similarity threshold
- 🌐 Simple web frontend (HTML + JS)
- 🐳 Docker and Docker Compose support
- 🤖 Uses `all-MiniLM-L6-v2` model from HuggingFace and projects embeddings to 1536-dim

---

## 🛠️ Prerequisites

- Python 3.9+
- Docker & Docker Compose
- [Pinecone](https://www.pinecone.io/) API Key

---

## 🔧 Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/semantic-question-search.git
cd semantic-question-search
```

### 2. Create a `.env` file

```bash
echo "PINECONE_API_KEY=your_pinecone_api_key_here" > .env
```

Replace `your_pinecone_api_key_here` with your actual Pinecone API key.

---

### 3. Run using Docker Compose

Ensure Docker and Docker Compose are installed.

```bash
docker-compose up --build
```

Once running, open your browser at:
👉 [http://localhost:5001](http://localhost:5001)

---

## 📂 Project Structure

```
├── app.py                   # Main Flask application
├── templates/
│   └── index.html           # Web interface for search/add/view
├── Dockerfile               # Docker setup
├── docker-compose.yaml      # Docker orchestration
├── .env                     # API keys (ignored by git)
├── .gitignore               # Ignored files
└── README.md                # This file
```

---

## 📡 API Endpoints

| Endpoint             | Method | Description                                  |
| -------------------- | ------ | -------------------------------------------- |
| `/get_embedding`     | POST   | Get 1536-dim vector for a question text      |
| `/add_embedding`     | POST   | Add new question with vector and metadata    |
| `/search_embeddings` | POST   | Retrieve top-K similar questions             |
| `/find_duplicates`   | POST   | Find duplicates above a similarity threshold |

---

## 🔍 Embedding Model

* Uses: `sentence-transformers/all-MiniLM-L6-v2`
* Projects 384-dim embeddings to 1536-dim to match Pinecone index using a random projection matrix.

---

## 🧊 Pinecone Setup

This app automatically creates the index if it doesn't exist:

* Name: `question-bank-index`
* Dimension: `1536`
* Metric: `cosine`
* Cloud: `aws`
* Region: `us-east-1`

---

## 📝 Example Usage (via Web UI)

1. Enter a new question and click **"Add Question"**
2. Search similar questions via the **Search** field
3. Use **"Find Duplicates"** to identify matching ones (≥0.9 similarity)

---

## 🐳 Docker Notes

### Dockerfile

The app installs Python dependencies, sets up Flask, and runs the app on port 5001.

### docker-compose.yaml

Includes service for Flask app with automatic environment file loading.

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

* [HuggingFace Transformers](https://huggingface.co/)
* [Pinecone Vector Database](https://www.pinecone.io/)
* [Flask Framework](https://flask.palletsprojects.com/)

