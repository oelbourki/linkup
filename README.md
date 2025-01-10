# Gossip Semantic Search app

Gossip Semantic Search app is a news retrieval application designed to scrape, embed, and search news articles efficiently using cutting-edge NLP models. It supports French language processing and provides a streamlined user experience through its Flask API and Streamlit UI.

## Installation

### Requirements
Ensure you have Python installed (version 3.7 or higher). Install the required packages using the following command:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
To scrape news articles and embed them:
```bash
python data.py --use_saved
```
- **`--use_scrape`**: Optional flag. If included, it will start scraping the `public.fr` website for fresh news articles. if not, the script will use the already saved pickle file with news data.

The scraped articles will then be embedded using the `sentence-transformers/distiluse-base-multilingual-cased` model, which is highly effective for French text processing. The embeddings and metadata are stored in ChromaDB for efficient retrieval.

### 2. Running the API
To start the Flask API for querying news:
```bash
python api.py
```
The API takes a user query, embeds it, and searches for relevant news articles. It supports enhanced retrieval mechanisms like hybrid search and parent document retrieval, ensuring comprehensive results by chunking and aggregating news data.

### 3. Running the UI
To launch the Streamlit UI for an interactive search experience:
```bash
streamlit run app.py
```
This interface allows users to input their queries and receive results visually.

## NLP Model Performance
The application leverages the `distiluse-base-multilingual-cased` model from Sentence Transformers. This model is optimized for multilingual text embeddings, particularly in French. Below are the model's evaluation metrics from the Semantic Textual Similarity (STS) 2017 dataset:

| Model                        | FR-EN | IT-EN | EN-EN | Average |
|------------------------------|-------|-------|-------|---------|
| distiluse-base-multilingual-cased | 80.2  | 80.5  | 85.4  | 80.1    |

This ensures high accuracy in semantic similarity tasks for French and cross-lingual data.

## Features
- **News Scraping**: Automatically scrapes articles from `public.fr`.
- **Efficient Embedding**: Uses a multilingual sentence transformer model to generate embeddings.
- **ChromaDB Storage**: Stores embeddings and metadata for fast retrieval.
- **Hybrid Search**: Combines vector similarity with metadata-based filtering.
- **Parent Document Retrieval**: Aggregates news chunks to provide complete context.

## Future Enhancements
- Support for additional languages and news sources.
- Advanced hybrid search for better relevance.
- Improved UI/UX in the Streamlit application.
