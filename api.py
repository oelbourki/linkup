from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import chromadb
from utils import clean_article_pipeline

app = Flask(__name__)

# Initialize sentence transformer
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased")

# Initialize ChromaDB
persist_directory = ".chroma"
client = chromadb.PersistentClient(path=persist_directory)

# Create collection. get_collection, get_or_create_collection, delete_collection also available!
collection = client.get_collection(name="news-public")


@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data["query"]
    # Generate embedding for query
    query_embedding = model.encode(query).tolist()
    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2,
        include=["metadatas", "distances"],
    )
    # Format results
    formatted_results = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        metadata["content"] = clean_article_pipeline(metadata["content"])
        formatted_results.append({"metadata": metadata, "distance": distance})

    return jsonify(formatted_results)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
