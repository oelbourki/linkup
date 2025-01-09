import pickle
import requests
from bs4 import BeautifulSoup
import time
import calendar
import random
from typing import List, Dict, Any
import datetime
from sentence_transformers import SentenceTransformer
import chromadb
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import logging
import json
from pathlib import Path
from urllib.parse import urlparse

# Ensure NLTK resources are available
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")


class RandomHeaders:
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/109.0",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36",
        ]
        self.accept_languages = [
            "en-US,en;q=0.9",
            "fr-FR,fr;q=0.8,en-US;q=0.5,en;q=0.3",
            "de-DE,de;q=0.9,en-US;q=0.7,en;q=0.3",
        ]

    def get_random_headers(self):
        headers = {
            "User-Agent": random.choice(self.user_agents),
            "Accept-Language": random.choice(self.accept_languages),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Connection": "keep-alive",
        }
        return headers


def get_headers():
    # Initialize the RandomHeaders class
    header_generator = RandomHeaders()

    # Generate random headers
    headers = header_generator.get_random_headers()
    return headers


# Function to save the list to a file
def save_list_to_file(lst, filename):
    with open(filename, "wb") as file:
        pickle.dump(lst, file)


# Function to load the list from a file
def load_list_from_file(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def get_years_range(archive_url: str) -> List[int]:
    """Get available years for scraping."""
    years = get_archive_years(archive_url)
    current_year = datetime.datetime.now().year

    if years:
        return sorted([int(year) for year in years])
    return list(range(2011, current_year + 1))


def clean_news_data(news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean and filter news data."""
    # Filter archive links
    filtered_data = [news for news in news_data if news["link"].find("archives")]
    filtered_data = filtered_data[1:]

    # Add content to each news item
    for news in filtered_data:
        try:
            news_metadata = scrape_news_content(news["link"])
            news.update(news_metadata)
            time.sleep(random.uniform(1, 3))
        except Exception as e:
            continue

    return [news for news in filtered_data if "content" in news]


def get_model(
    model_name: str = "sentence-transformers/distiluse-base-multilingual-cased",
) -> Any:
    """Create embeddings from text content."""
    model = SentenceTransformer(model_name)
    return model


def create_embeddings(texts: List[str], model) -> Any:
    """Create embeddings from text content."""
    embeddings = model.encode(texts)
    return embeddings


def setup_chromadb(
    embeddings: List[List[float]],
    metadata: List[Dict[str, Any]],
    persist_directory: str = ".chroma",
) -> None:
    """Setup ChromaDB and store embeddings."""
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name="news-public")
    print(len(embeddings))
    print(len(metadata))
    ids = [f"doc{i}" for i in range(len(metadata))]
    collection.add(embeddings=embeddings, metadatas=metadata, ids=ids)


def setup_logging():
    """Configure logging for both file and console output."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = (
        log_dir / f"scraper_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,  # Log level set to INFO (adjust as needed)
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # Log to file
            logging.StreamHandler(),  # Log to console
        ],
    )


def log_data(data, filename):
    """Log structured data to JSON file."""
    log_dir = Path("data_logs")
    log_dir.mkdir(exist_ok=True)

    filepath = (
        log_dir / f"{filename}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_archive_years(url):
    logging.info(f"Fetching archive years from {url}")
    years = []
    try:
        headers = get_headers()
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            archive_section = soup.find("div", class_="grid gap-3")
            if archive_section:
                links = archive_section.find_all("a")
                years = [
                    year.get_text(strip=True)
                    for year in links
                    if year.get_text(strip=True).isdigit()
                ]
                logging.info(f"Found {len(years)} archive years: {years}")
            else:
                logging.warning("Archive section not found")
        else:
            logging.error(f"Failed to fetch archive page: {response.status_code}")
    except Exception as e:
        logging.error(f"Error in get_archive_years: {str(e)}")

    log_data({"years": years}, "archive_years")
    return years


def scrape_news_links(years):
    logging.info(f"Starting news scraping for years: {years}")
    news_data = []

    for year in years:
        for month in range(1, 13):
            month_str = str(month).zfill(2)
            num_days = calendar.monthrange(year, month)[1]

            for day in range(1, num_days + 1):
                day_str = str(day).zfill(2)
                url = f"https://www.public.fr/archives/{year}/{month_str}/{day_str}"
                logging.info(f"Scraping: {url}")

                try:
                    headers = get_headers()
                    # print(headers)
                    response = requests.get(url, headers=headers)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, "html.parser")
                        main_section = soup.find(
                            "main",
                            class_="skin:bg-white skin:!max-w-none pt-20 md:pt-[4.5rem] pb-20 flex-grow content-start container grid gap-6",
                        )

                        if main_section:
                            articles = main_section.find_all("a")
                            for article in articles:
                                row = {
                                    "title": article.get_text(),
                                    "link": article.get("href"),
                                    "date": f"{day_str}/{month_str}/{year}",
                                    "day": day_str,
                                    "month": month_str,
                                    "year": year,
                                }
                                if row["title"] and row["link"]:
                                    news_data.append(row)
                                    logging.info(f"Found article: {row['title']}")
                                    # print(row)
                                    log_data(
                                        row, f"article_{year}_{month_str}_{day_str}"
                                    )
                        else:
                            logging.warning(
                                f"No articles found for {year}-{month_str}-{day_str}"
                            )
                    else:
                        logging.error(f"Failed to fetch {url}: {response.status_code}")

                except Exception as e:
                    logging.error(f"Error scraping {url}: {str(e)}")

                time.sleep(random.uniform(1, 5))

    log_data({"total_articles": len(news_data)}, "scraping_summary")
    return news_data


def scrape_news_content(url):
    logging.info(f"Scraping content from {url}")
    try:
        response = requests.get(url, headers=get_headers())
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")

            row = {
                "url": url,
                "title": (
                    soup.find(
                        "h1", class_="text-2xl md:text-3xl font-bold inline"
                    ).get_text(strip=True)
                    if soup.find("h1")
                    else "Title not found"
                ),
                "publish_date": (
                    soup.find("time", {"datetime": True})["datetime"]
                    if soup.find("time")
                    else "Date not found"
                ),
                "author": (
                    soup.find("a", class_="text-primary font-medium").get_text(
                        strip=True
                    )
                    if soup.find("a", class_="text-primary font-medium")
                    else "Author not found"
                ),
                "website_name": urlparse(url).netloc,
                "tags": (
                    [
                        tag.get_text(strip=True)
                        for tag in soup.find(
                            "div", class_=lambda v: v and "max-w-full" in v
                        ).find_all("a")
                    ]
                    if soup.find("div", class_=lambda v: v and "max-w-full" in v)
                    else []
                ),
                "content": (
                    "\n".join(
                        [
                            p.get_text()
                            for p in soup.find("div", id="post-content").find_all("p")
                        ]
                    )
                    if soup.find("div", id="post-content")
                    else "Content not found"
                ),
            }

            logging.info(f"Successfully scraped content: {row['title']}")
            log_data(row, f"content_{urlparse(url).path.replace('/', '_')}")
            return row

    except Exception as e:
        logging.error(f"Error scraping content from {url}: {str(e)}")
        return {"error": str(e)}


# Step 1: Remove HTML tags
def remove_html_tags(text):
    """
    Remove HTML tags from the text using BeautifulSoup.
    """
    return BeautifulSoup(text, "html.parser").get_text()


# Step 2: Remove extra spaces, newline characters, and tabs
def remove_extra_spaces(text):
    """
    Remove extra spaces, newline characters, and tabs from the text.
    """
    return re.sub(r"\s+", " ", text).strip()


# Step 3: Remove URLs and email addresses
def remove_urls_and_emails(text):
    """
    Remove URLs and email addresses from the text.
    """
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    return re.sub(r"\S+@\S+", "", text)  # Remove email addresses


# Step 4: Remove numbers
def remove_numbers(text):
    """
    Remove all numbers from the text.
    """
    return re.sub(r"\d+", "", text)


# Step 5: Remove punctuation
def remove_punctuation(text):
    """
    Remove punctuation from the text.
    """
    return text.translate(str.maketrans("", "", string.punctuation))


# Step 6: Tokenize the text into words
def tokenize_text(text):
    """
    Tokenize the text into individual words.
    """
    return word_tokenize(text)


# Step 7: Remove stopwords
def remove_stopwords(words):
    """
    Remove stopwords from the tokenized words.
    """
    stop_words = set(stopwords.words("english"))
    return [word for word in words if word.lower() not in stop_words]


# Step 8: Lowercase the text
def lowercase_text(text):
    """
    Convert the text to lowercase.
    """
    return text.lower()


# Full Pipeline: Combine all steps
def clean_article_pipeline(article_text):
    """
    Clean the article by applying all cleaning functions in sequence.
    """
    # Step 1: Remove HTML tags
    cleaned_text = remove_html_tags(article_text)

    # Step 2: Remove extra spaces
    cleaned_text = remove_extra_spaces(cleaned_text)

    # Step 3: Remove URLs and emails
    cleaned_text = remove_urls_and_emails(cleaned_text)

    # Step 4: Remove numbers
    cleaned_text = remove_numbers(cleaned_text)

    # Step 5: Remove punctuation
    cleaned_text = remove_punctuation(cleaned_text)

    # Step 6: Tokenize the text into words
    words = tokenize_text(cleaned_text)

    # Step 7: Remove stopwords
    words = remove_stopwords(words)

    # Step 8: Reassemble words into a cleaned string and lowercase the text
    cleaned_text = " ".join(words)
    cleaned_text = lowercase_text(cleaned_text)

    return cleaned_text


def load_saved_articles():
    articles = load_list_from_file("articlesx.pkl")
    return articles
