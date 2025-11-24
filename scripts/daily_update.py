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
NEW_FILE = "new_articles.json"
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

def save_articles(data, file_path=DATA_FILE):
    with open(file_path, "w", encoding="utf-8") as f:
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
from datetime import timedelta

def scrape_rss_feeds():
    all_articles = load_articles()
    new_articles = []
    added = 0
    today = datetime.now(timezone.utc).date()  # current UTC date

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            print(f"⚠ Failed to parse RSS feed {feed_url}: {e}")
            continue

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
            except Exception as e:
                print(f"⚠ Failed to parse date for {url}: {e}")
                continue

            # Only articles published today (UTC)
            if published_date.date() != today:
                continue

            if url in all_articles:
                continue

            content = scrape_full_article(url)
            if not content:
                continue

            article_data = {
                "url": url,
                "title": title,
                "date": published_date.isoformat(),
                "content": content
            }

            all_articles[url] = article_data
            new_articles.append(article_data)
            added += 1

    # Save new articles and update main file
    save_articles({a["url"]: a for a in new_articles}, NEW_FILE)
    save_articles(all_articles)
    print(f"✔ Added {added} new articles from today")
    return list(all_articles.values()), new_articles

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
# --------------------- MAIN ---------------------
def main():
    # Load existing FAISS index
    if os.path.exists(VECTORSTORE_FOLDER):
        vectorstore = FAISS.load_local(VECTORSTORE_FOLDER, embeddings, allow_dangerous_deserialization=True)
        print("✔ Loaded existing FAISS index")
    else:
        vectorstore = FAISS.from_documents([], embeddings)
        print("✔ Created new FAISS index")

    # Scrape articles & get today's new articles only
    all_articles, new_articles_today = scrape_rss_feeds()

    # Update FAISS with only today's new articles
    update_rag_with_articles(vectorstore, new_articles_today)

    # Save FAISS index
    vectorstore.save_local(VECTORSTORE_FOLDER)
    print("✔ FAISS index updated")

    # Save list of today's new articles to new_articles.txt
    if new_articles_today:
        with open("new_articles.txt", "w", encoding="utf-8") as f:
            for a in new_articles_today:
                f.write(f"{a['title']} - {a['url']}\n")
    else:
        with open("new_articles.txt", "w", encoding="utf-8") as f:
            f.write("NO_NEW_ARTICLES")

if __name__ == "__main__":
    main()
