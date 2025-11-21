import os
import re
import json
import feedparser
import requests
from bs4 import BeautifulSoup
from readability.readability import Document as ReadabilityDocument
from datetime import datetime, timezone
from dateutil import parser as dateparser

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTORSTORE_FOLDER = "data/vectorstore/faiss_index"
DATA_FILE = "articles.json"
MIN_ARTICLE_LENGTH = 200
MIN_CHUNK_LENGTH = 20
HEADERS = {"User-Agent": "Mozilla/5.0"}

RSS_FEEDS = [
    "https://blog.hubspot.com/marketing/rss.xml",
    "https://moz.com/blog/feed",
    "https://www.shopify.com/blog/topics/marketing.rss",
    "https://www.searchenginejournal.com/category/seo/feed/",
    "https://www.socialmediaexaminer.com/social-media-marketing/feed/",
    "https://review.firstround.com/articles/pr-and-marketing/feed",
    "https://thecmo.com/marketing-operations/feed/",
    "https://thecmo.com/marketing-strategy/feed/",
    "https://thecmo.com/customer-marketing/feed/",
    "https://buffer.com/resources/social-media-marketing/feed",
    "https://buffer.com/resources/creator/feed",
    "https://sproutsocial.com/insights/social-media-marketing-resources/feed",
    "https://www.blogdigital.fr/category/marketing/feed/",
    "https://www.marketingdive.com/topic/Social-media-marketing/feed/",
    "https://contentmarketinginstitute.com/strategy-planning/content-marketing-strategy/feed/",
    "https://contentmarketinginstitute.com/strategy-planning/ai-in-marketing/feed/",
    "https://contentmarketinginstitute.com/measurement-optimization/analytics-data/feed/",
    "https://digitalmarketinginstitute.com/blog/feed/",
    "https://embryo.com/category/digital-marketing/feed/",
    "https://martech.org/topic/social-media-marketing/feed/",
    "https://martech.org/topic/marketing-automation/feed/",
    "https://martech.org/topic/content-marketing/feed/",
    "https://martech.org/topic/marketing-analytics/feed/"
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[_]{2,}", " ", text)
    return text.strip()

# --------------------- Load / Save Articles ---------------------
def load_articles():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {a["url"]: a for a in data if "url" in a}
    return {}

def save_articles(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(list(data.values()), f, indent=2, ensure_ascii=False)

# --------------------- Scrape Full Article ---------------------
def scrape_full_article(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        doc = ReadabilityDocument(r.text)
        html = doc.summary()
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 50]
        paragraphs = list(dict.fromkeys(paragraphs))
        content = "\n".join(paragraphs)
        return content if len(content) >= MIN_ARTICLE_LENGTH else None
    except:
        return None

# --------------------- Scrape RSS ---------------------
def scrape_rss_feeds():
    articles = load_articles()
    added = 0
    now = datetime.now(timezone.utc)
    today = now.date()

    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            url = entry.get("link")
            title = entry.get("title")
            if not url or not title:
                continue

            published_str = entry.get("published") or entry.get("updated")
            if not published_str:
                continue

            try:
                published_date = dateparser.parse(published_str)
                if published_date.tzinfo is None:
                    published_date = published_date.replace(tzinfo=timezone.utc)
                else:
                    published_date = published_date.astimezone(timezone.utc)
            except:
                continue

            # Filter: only today
            if published_date.date() != today:
                continue

            if url in articles:
                continue

            content = scrape_full_article(url)
            if not content:
                continue

            articles[url] = {
                "url": url,
                "title": title,
                "date": published_date.isoformat(),
                "content": content
            }
            added += 1

    save_articles(articles)
    print(f"✔ Added {added} new articles from today")
    return list(articles.values())

# --------------------- Update FAISS ---------------------
def update_rag_with_articles(vectorstore, articles):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, add_start_index=True)
    new_docs = []

    for article in articles:
        chunks = text_splitter.split_text(article["content"])
        for c in chunks:
            cleaned = clean_text(c)
            if len(cleaned) >= MIN_CHUNK_LENGTH:
                new_docs.append(Document(page_content=cleaned, metadata={"source": article["url"], "title": article["title"]}))

    if new_docs:
        vectorstore.add_documents(new_docs)
        print(f"✔ Added {len(new_docs)} article chunks to the vector store")

# --------------------- MAIN ---------------------
def main():
    # Load existing FAISS index
    if os.path.exists(VECTORSTORE_FOLDER):
        vectorstore = FAISS.load_local(VECTORSTORE_FOLDER, embeddings, allow_dangerous_deserialization=True)
        print("✔ Loaded existing FAISS index")
    else:
        vectorstore = FAISS.from_documents([], embeddings)
        print("✔ Created new FAISS index")

    # Scrape articles & update RAG
    articles = scrape_rss_feeds()
    update_rag_with_articles(vectorstore, articles)

    # Save FAISS index
    vectorstore.save_local(VECTORSTORE_FOLDER)
    print("✔ FAISS index updated")

if __name__ == "__main__":
    main()
