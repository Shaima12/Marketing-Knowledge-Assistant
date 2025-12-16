
import os
import re
import json
import uuid
import feedparser
import requests
from bs4 import BeautifulSoup
from readability.readability import Document as ReadabilityDocument
from datetime import datetime, timezone
from dateutil import parser as dateparser

from sentence_transformers import SentenceTransformer, util
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
# ===================== Connect Qdrant Client =====================

qdrant = QdrantClient(
    url="https://d7bb08f9-84c8-4901-b9ea-c30ad9c70822.europe-west3-0.gcp.cloud.qdrant.io:6333",  # your cluster URL
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.VFLaw0AblT3k3pKyICEV0JdXmBx-y_YJacDqT5Belz0"
)
# ===================== CONFIG =====================
COLLECTION_NAME = "general_docs"
DATA_FILE = "articles.json"
NEW_FILE = "new_articles.json"
MIN_ARTICLE_LENGTH = 200
MIN_CHUNK_LENGTH = 40
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


# ===================== MODELS =====================
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
VECTOR_SIZE = embedding_model.get_sentence_embedding_dimension()
# ===================== UTILS =====================
def clean_text(text):
    """
    Clean text for embedding: remove extra whitespace, newlines, non-ASCII, unicode artifacts, underscores
    """
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\\u[0-9a-fA-F]{4}", " ", text)  # Unicode escape sequences
    text = re.sub(r"\s+", " ", text)  # multiple spaces -> one
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ascii
    text = re.sub(r"[_]{2,}", " ", text)
    return text.strip()
# ===================== STORAGE =====================
def load_articles():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {a["url"]: a for a in data}
    return {}

def save_articles(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(data.values()), f, indent=2, ensure_ascii=False)

# ===================== SCRAPING =====================
def scrape_full_article(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None

        doc = ReadabilityDocument(r.text)
        soup = BeautifulSoup(doc.summary(), "html.parser")

        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 50
        ]

        content = "\n".join(dict.fromkeys(paragraphs))
        return content if len(content) >= MIN_ARTICLE_LENGTH else None

    except Exception as e:
        print(f"⚠ Scraping failed {url}: {e}")
        return None

# ===================== SEMANTIC CHUNKING =====================
def semantic_chunk(text, similarity_threshold=0.65, max_len=1500):
    sentences = re.split(r"(?<=[.!?]) +", text)
    if len(sentences) < 2:
        return []

    embeddings = embedding_model.encode(sentences, convert_to_tensor=True)

    chunks = []
    current = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = util.cos_sim(embeddings[i], embeddings[i - 1]).item()
        if sim < similarity_threshold or sum(len(s) for s in current) > max_len:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])

    if current:
        chunks.append(" ".join(current))

    return [clean_text(c) for c in chunks if len(c) >= MIN_CHUNK_LENGTH]
# ===================== RSS INGESTION =====================
def scrape_rss():
    all_articles = load_articles()
    new_articles = []
    today = datetime.now(timezone.utc).date()

    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)

        for entry in feed.entries:
            url = entry.get("link")
            title = entry.get("title")

            if not url or not title or url in all_articles:
                continue

            published = entry.get("published") or entry.get("updated")
            if not published:
                continue

            try:
                date = dateparser.parse(published).astimezone(timezone.utc)
            except:
                continue

            if date.date() != today:
                continue

            content = scrape_full_article(url)
            if not content:
                continue

            article = {
                "url": url,
                "title": title,
                "date": date.isoformat(),
                "content": content
            }

            all_articles[url] = article
            new_articles.append(article)

            print(f"🆕 New article: {title}")

    save_articles(all_articles, DATA_FILE)
    save_articles({a["url"]: a for a in new_articles}, NEW_FILE)

    return new_articles

# ===================== QDRANT UPSERT =====================
def upsert_to_qdrant(articles):
    points = []

    for article in articles:
        chunks = semantic_chunk(article["content"])

        print(f"📄 '{article['title']}' → {len(chunks)} chunks")

        for idx, chunk in enumerate(chunks):
            vector = embedding_model.encode(chunk).tolist()

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "title": article["title"],
                        "url": article["url"],
                        "content": chunk,
                        "chunk_index": idx,
                        "published_date": article["date"],
                        "source": "rss_marketing"
                    }
                )
            )

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ {len(points)} chunks successfully stored in Qdrant")
    else:
        print("ℹ No new chunks to store")
# ===================== MAIN =====================
def main():
    print("🚀 Starting daily RAG update")

    new_articles = scrape_rss()

    if new_articles:
        upsert_to_qdrant(new_articles)

        with open("Articles\new_articles.txt", "w", encoding="utf-8") as f:
            for a in new_articles:
                f.write(f"{a['title']} - {a['url']}\n")
    else:
        with open("Articles\new_articles.txt", "w", encoding="utf-8") as f:
            f.write("NO_NEW_ARTICLES")

    print("🎉 Pipeline finished")

if __name__ == "__main__":
    main()
