"""
News Sentiment Scanner
======================
Fetches recent news articles about gold markets from Google News RSS feeds
and performs sentiment analysis on the headlines using FinBERT to gauge
overall market sentiment (Positive, Negative, or Neutral).
"""

# --- Standard library imports ---
import ssl
import urllib.request
from datetime import datetime
from urllib.parse import quote

# --- Third-party imports ---
import feedparser                   # Parses RSS/Atom feeds into Python objects
import requests                     # HTTP library for fetching web pages
from bs4 import BeautifulSoup       # HTML parser for extracting article text
from textblob import TextBlob       # Alternative sentiment analysis (not currently used)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # Rule-based sentiment analyzer
import certifi                      # Provides Mozilla's CA certificate bundle for SSL verification

# --- Hugging Face / PyTorch imports (for FinBERT alternative) ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load the FinBERT model pre-trained on financial text.
# This runs at startup and downloads the model on first use (~400MB).
# FinBERT classifies text into: Positive, Negative, or Neutral.
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")

# Sentiment labels corresponding to FinBERT's output indices
labels = ['Positive', 'Negative', 'Neutral']


def fetch_news(query, num_articles=10):
    """
    Fetch news articles from Google News RSS feed for a given search query.

    Args:
        query: Search term (e.g., "gold price").
        num_articles: Maximum number of articles to return (default 10).

    Returns:
        A list of dicts, each containing 'title', 'link', 'published', and 'content'.
    """
    # Build the Google News RSS URL, encoding the query for safe URL usage
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}"

    # Create an SSL context using certifi's CA bundle to fix certificate
    # verification issues commonly seen on macOS Python installations
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    handler = urllib.request.HTTPSHandler(context=ssl_context)

    # Parse the RSS feed with a User-Agent header to avoid being blocked
    feed = feedparser.parse(rss_url, request_headers={"User-Agent": "Mozilla/5.0"},
                            handlers=[handler])

    # Limit results to the requested number of articles
    news_items = feed.entries[:num_articles]

    # Extract relevant fields from each RSS entry and fetch full article content
    articles = []
    for item in news_items:
        title = item.title
        link = item.link
        published = item.published
        content = fetch_article_content(link)

        articles.append({
            "title": title,
            "link": link,
            "published": published,
            "content": content
        })

    return articles


def fetch_article_content(url):
    """
    Scrape the full text content from a news article URL.

    Extracts all <p> (paragraph) tags from the page HTML and joins them
    into a single string. Returns a fallback message if the request fails
    (e.g., due to paywalls, bot blocking, or network errors).

    Args:
        url: The full URL of the news article.

    Returns:
        The article's text content, or "Content not retrieved." on failure.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP error codes (4xx, 5xx)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text from all paragraph tags — this captures the main article body
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content.strip()
    except requests.RequestException:
        return "Content not retrieved."


# --- VADER Alternative ---
# Uncomment this function (and comment out the FinBERT version below) to use
# VADER, a rule-based sentiment analyzer. Faster and lighter than FinBERT
# but less accurate for financial news.
#
# def analyze_sentiment(text):
#     """Analyze sentiment using VADER (rule-based sentiment analyzer)."""
#     analyzer = SentimentIntensityAnalyzer()
#     scores = analyzer.polarity_scores(text)
#     polarity = scores['compound']  # Normalized composite score (-1 to +1)
#
#     if polarity > 0.05:
#         sentiment = 'Positive'
#     elif polarity < -0.05:
#         sentiment = 'Negative'
#     else:
#         sentiment = 'Neutral'
#
#     return polarity, sentiment


def analyze_sentiment(text):
    """
    Analyze the sentiment of a text string using FinBERT.

    FinBERT is a BERT model fine-tuned on financial text (yiyanghkust/finbert-tone).
    It classifies text into Positive, Negative, or Neutral with a confidence score.
    More accurate than rule-based approaches for financial news headlines.

    Args:
        text: The text to analyze (typically a news headline).

    Returns:
        A tuple of (confidence, sentiment) where confidence is a float (0 to 1)
        and sentiment is one of 'Positive', 'Negative', or 'Neutral'.
    """
    if not text.strip():
        return 0.0, 'Neutral'

    # Tokenize and truncate to FinBERT's max input length (512 tokens)
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # Run inference without tracking gradients (faster, less memory)
    with torch.no_grad():
        outputs = finbert_model(**inputs)

    # Convert logits to probabilities and pick the highest-confidence label
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).numpy()[0]
    max_index = np.argmax(probabilities)
    sentiment = labels[max_index]
    confidence = probabilities[max_index]

    return confidence, sentiment


def summarize_sentiments(articles):
    """
    Analyze all articles and print a sentiment distribution summary.

    Counts how many headlines are positive, negative, and neutral,
    then prints the totals and percentages.

    Args:
        articles: List of article dicts (each must have a 'title' key).
    """
    summary = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0
    }

    # Classify each article headline and tally the results
    for article in articles:
        _, sentiment = analyze_sentiment(article['title'])
        summary[sentiment] += 1

    total = len(articles)
    print("\n--- Market Sentiment Summary ---")
    print(f"Total articles analyzed: {total}")

    # Guard against division by zero when no articles were fetched
    if total == 0:
        print("No articles were fetched. Please check your internet connection or try different queries.")
        return

    for sentiment, count in summary.items():
        percent = (count / total) * 100
        print(f"{sentiment}: {count} ({percent:.2f}%)")


def main():
    """
    Main entry point. Fetches gold-related news across multiple search queries,
    prints per-article sentiment analysis, and outputs an overall summary.
    """
    # Multiple query variations to get broader coverage of gold market news
    queries = [
        "gold market",
        "gold price",
        "gold news",
        "gold trends",
        "gold analysis",
        "gold forecast",
        "gold investment"
    ]
    num_articles_per_query = 10
    all_articles = []

    # Fetch articles for each query and combine into a single list
    for query in queries:
        print(f"Fetching news articles for '{query}'...\n")
        articles = fetch_news(query, num_articles_per_query)
        all_articles.extend(articles)

    # Print per-article sentiment results
    for idx, article in enumerate(all_articles, 1):
        print(f"Article {idx}: {article['title']}")
        print(f"Link: {article['link']}")
        print(f"Published: {article['published']}")

        confidence, sentiment = analyze_sentiment(article['title'])
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})\n")

    # Print the overall sentiment distribution
    summarize_sentiments(all_articles)


if __name__ == "__main__":
    main()