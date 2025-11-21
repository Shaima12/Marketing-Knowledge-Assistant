import feedparser
import json
import os
import requests
import faiss
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
from sentence_transformers import SentenceTransformer

# =====================================================
# ------------------ SETTINGS -------------------------
# =====================================================

TOPIC =  """
Marketing includes: digital marketing, advertising, campaigns,
brand strategy, social media, SEO, content marketing, analytics,
growth marketing, consumer psychology, customer acquisition.
"""
SIM_THRESHOLD = 0.45

# Google News RSS for last 24h articles on marketing
RSS_URL = "https://news.google.com/rss/search?q=marketing+when:1d&hl=en-US&gl=US&ceid=US:en"

ARTICLES_PATH = "articles.json"
VECTOR_DIR = "vectorstore"
VECTOR_PATH = f"{VECTOR_DIR}/index.faiss"
METADATA_PATH = f"{VECTOR_DIR}/metadata.json"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
EMB_DIM = model.get_sentence_embedding_dimension()

print("Model loaded ✔", "Embedding dim:", EMB_DIM)

# =====================================================
# ------------------ HELPERS ---------------------------
# =====================================================

def load_articles():
    """Load JSON as dict(url → article) even if file is a list."""
    if not os.path.exists(ARTICLES_PATH):
        return {}

    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):  
        print("⚠ articles.json is a LIST → converting to DICT")
        converted = {}
        for a in data:
            if "url" in a:
                converted[a["url"]] = {
                    "title": a.get("title", ""),
                    "date": a.get("date", ""),
                    "content": a.get("content", "")
                }
        return converted

    return data


def save_articles(data):
    with open(ARTICLES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_vectorstore():
    if not os.path.exists(VECTOR_PATH):
        return None, []

    index = faiss.read_index(VECTOR_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def save_vectorstore(index, metadata):
    os.makedirs(VECTOR_DIR, exist_ok=True)
    faiss.write_index(index, VECTOR_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# =====================================================
# ------------------ SCRAPER ---------------------------
# =====================================================

HEADERS = {"User-Agent": "Mozilla/5.0"}

def extract_full_text(url):
    """Extract readable article text (improved)"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        # Try best containers
        candidates = [
            "article",
            "div.post-content",
            "div.entry-content",
            "section.article-content",
            "section.content"
        ]

        for sel in candidates:
            block = soup.select_one(sel)
            if block:
                paragraphs = [p.get_text(" ", strip=True) for p in block.find_all("p")]
                if len(" ".join(paragraphs)) > 200:
                    return "\n".join(paragraphs)

        # Fallback: all <p>
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        return "\n".join(paragraphs[:80])

    except Exception as e:
        print("Error scraping:", e)
        return ""


# =====================================================
# ------------------ MAIN LOGIC ------------------------
# =====================================================

def run_scraper():
    print("🔍 Fetching RSS…")
    feed = feedparser.parse(RSS_URL)

    existing = load_articles()
    new_articles = {}

    index, metadata = load_vectorstore()

    if index is None:
        print("Creating new FAISS index")
        index = faiss.IndexFlatL2(EMB_DIM)
        metadata = []

    added_count = 0

    for entry in feed.entries:
        url = entry.link

        # Google News adds redirects → clean them
        if "news.google.com" in url and "url=" in url:
            url = url.split("url=")[-1]

        if url in existing:
            continue

        print("\n→ Fetching:", url)

        content = extract_full_text(url)
        if len(content) < 200:
            print("⚠ Article too short")
            continue

        # Embeddings
        topic_emb = model.encode([TOPIC], convert_to_numpy=True)
        art_emb = model.encode([content], convert_to_numpy=True)

        # Similarity
        sim = float(np.dot(topic_emb, art_emb.T)[0][0])

        print("semantic sim:", round(sim, 3))

        if sim < SIM_THRESHOLD:
            print("❌ Not relevant → skipped")
            continue

        # Store article
        new_articles[url] = {
            "title": entry.title,
            "date": entry.get("published", ""),
            "content": content
        }

        # Add embedding to vectorstore
        index.add(art_emb.astype("float32"))
        metadata.append({"url": url, "title": entry.title})

        added_count += 1

    # Save updates
    existing.update(new_articles)
    save_articles(existing)
    save_vectorstore(index, metadata)

    # Output
    print("\n===== SUMMARY =====")
    print("Added:", added_count)
    print("Total stored:", len(existing))

    if added_count > 0:
        print("\nNew articles:")
        for v in new_articles.values():
            print("-", v["title"])


# =====================================================
# ------------------ RUN SCRIPT ------------------------
# =====================================================

run_scraper()