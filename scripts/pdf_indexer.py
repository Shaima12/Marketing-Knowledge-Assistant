import os
import re
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore # Added for direct FAISS index creation

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader

PDF_FOLDER = "data/pdfs"
VECTORSTORE_FOLDER = "data/vectorstore/faiss_index"
MIN_CHUNK_LENGTH = 20

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[_]{2,}", " ", text)
    return text.strip()

def load_and_index_pdfs():
    loader = PyPDFDirectoryLoader(PDF_FOLDER)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, add_start_index=True)

    pdf_docs = []
    for doc in docs:
        chunks = text_splitter.split_text(doc.page_content)
        cleaned = [clean_text(c) for c in chunks]
        for c in cleaned:
            if len(c) >= MIN_CHUNK_LENGTH:
                pdf_docs.append(Document(page_content=c, metadata={"source": doc.metadata.get("source", "pdf")}))

    print(f"✔ Loaded {len(pdf_docs)} PDF chunks")

    # Save FAISS index
    faiss_index = FAISS.from_documents(pdf_docs, embeddings)
    faiss_index.save_local(VECTORSTORE_FOLDER)
    print(f"✔ FAISS index saved at {VECTORSTORE_FOLDER}")

if __name__ == "__main__":
    load_and_index_pdfs()
