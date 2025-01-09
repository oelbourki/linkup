import argparse
from utils import (
    scrape_news_links,
    setup_chromadb,
    get_years_range,
    clean_news_data,
    get_model,
    create_embeddings,
    setup_logging,
    load_saved_articles,
)

# Setup logging
setup_logging()


def main(use_scrape: bool):
    # Configuration
    archive_url = "https://www.public.fr/"

    if not use_scrape:
        # Load saved articles (you can replace this with actual loading logic, e.g., reading from a file or database)
        print("Loading saved articles...")
        # Assuming a function `load_saved_articles()` exists
        saved_news = load_saved_articles()
        print(f"Loaded {len(saved_news)} saved articles.")

        # Process embeddings and store in ChromaDB if necessary
        content_to_embed = [news["content"] for news in saved_news]

        model = get_model()
        embeddings = create_embeddings(content_to_embed, model=model)

        # Store in ChromaDB
        setup_chromadb(embeddings.tolist(), saved_news)
    else:
        # Scrape news data
        # Get available years
        years = get_years_range(archive_url)
        print("Available archive years:", years)

        # Scrape news links and content
        news_data = scrape_news_links(years)
        cleaned_news = clean_news_data(news_data)

        # Create embeddings
        content_to_embed = [news["content"] for news in cleaned_news]

        model = get_model()
        embeddings = create_embeddings(content_to_embed, model=model)

        # Store in ChromaDB
        setup_chromadb(embeddings.tolist(), cleaned_news)


if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="News scraping and article embedding")
    parser.add_argument(
        "--use_scrape",
        action="store_true",
        default=False,
        help="If set, load saved articles instead of scraping new ones",
    )

    args = parser.parse_args()
    main(use_scrape=args.use_scrape)
