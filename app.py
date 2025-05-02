import os
import uuid
import time
import logging
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "question-bank-index"
DIMENSION = 1536
PROJECTION_FILE = 'projection.npy'

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to index
if INDEX_NAME not in pc.list_indexes().names():
    app.logger.info("Creating new Pinecone index...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    # Wait for index initialization
    while True:
        index_status = pc.describe_index(INDEX_NAME).status
        if index_status.ready:
            break
        app.logger.info("Waiting for index initialization...")
        time.sleep(5)

index = pc.Index(INDEX_NAME)

# Load embedding model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Handle projection matrix
if os.path.exists(PROJECTION_FILE):
    projection_matrix = np.load(PROJECTION_FILE)
else:
    projection_matrix = np.random.randn(384, DIMENSION)
    np.save(PROJECTION_FILE, projection_matrix)
    app.logger.info("Created new projection matrix")


def generate_embedding(text):
    try:
        inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        projected = np.dot(embedding, projection_matrix)
        return projected.tolist()
    except Exception as e:
        app.logger.error(f"Embedding error: {str(e)}")
        raise


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text'"}), 400

        embedding = generate_embedding(data['text'])
        return jsonify(embedding), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/add_embedding', methods=['POST'])
def add_embedding():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text'"}), 400

        text = data['text']
        embedding = generate_embedding(text)
        custom_metadata = data.get('metadata', {})

        # Combine text with custom metadata
        metadata = {"text": text, **custom_metadata}

        embedding_id = str(uuid.uuid4())
        index.upsert(vectors=[{
            "id": embedding_id,
            "values": embedding,
            "metadata": metadata
        }])

        return jsonify({
            "message": "Question added successfully",
            "id": embedding_id
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/search_embeddings', methods=['POST'])
def search_embeddings():
    try:
        data = request.get_json()
        if not data or 'query_embedding' not in data:
            return jsonify({"error": "Missing query embedding"}), 400

        query_embedding = data['query_embedding']
        if len(query_embedding) != DIMENSION:
            return jsonify({"error": f"Embedding must be {DIMENSION} dimensions"}), 400

        results = index.query(
            vector=query_embedding,
            top_k=data.get('top_k', 5),
            include_metadata=True
        )

        matches = [{
            "text": match.metadata.get("text", "Question text not available"),
            "score": match.score,
            "metadata": match.metadata
        } for match in results.matches]

        return jsonify({"matches": matches}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/find_duplicates', methods=['POST'])
def find_duplicates():
    try:
        data = request.get_json()
        if not data or 'query_embedding' not in data:
            return jsonify({"error": "Missing query embedding"}), 400

        query_embedding = data['query_embedding']
        if len(query_embedding) != DIMENSION:
            return jsonify({"error": f"Embedding must be {DIMENSION} dimensions"}), 400

        results = index.query(
            vector=query_embedding,
            top_k=100,
            include_metadata=True
        )

        threshold = data.get('similarity_threshold', 0.6)
        duplicates = [{
            "text": match.metadata.get("text", "Question text not available"),
            "score": match.score,
            "metadata": match.metadata
        } for match in results.matches if match.score >= threshold]

        return jsonify({"duplicates": duplicates}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)