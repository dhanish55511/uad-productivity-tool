import numpy as np
import faiss
import json
import os
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from typing import List, Tuple
from config import (
    PDF_PATH, FAISS_INDEX_PATH, DOCUMENTS_PATH, 
    EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_NAME
)

# Global variables to hold the RAG components
embedding_model: SentenceTransformer = None
DOCUMENTS: List[str] = []
FAISS_INDEX: faiss.IndexFlatL2 = None

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        return full_text
    except Exception as e:
        print(f"Error reading PDF file {pdf_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size=500, overlap=100) -> List[str]:
    """Splits text into chunks by word count."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def get_or_create_rag_data() -> Tuple[List[str], faiss.IndexFlatL2]:
    """
    Initializes the embedding model and loads RAG data from files if available,
    otherwise, processes the PDF and creates the index.
    """
    global embedding_model, DOCUMENTS, FAISS_INDEX

    try:
        # Initialize the embedding model
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        print("Please ensure you have run: pip install sentence-transformers")
        exit()

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCUMENTS_PATH):
        print("Loading existing RAG data from files...")
        try:
            FAISS_INDEX = faiss.read_index(FAISS_INDEX_PATH)
            with open(DOCUMENTS_PATH, 'r', encoding='utf-8') as f:
                DOCUMENTS = json.load(f)
            print(f"Successfully loaded index and {len(DOCUMENTS)} documents.")
            return DOCUMENTS, FAISS_INDEX
        except Exception as e:
            print(f"Error loading files: {e}. Re-generating...")

    print("Creating new RAG data...")
    # 1. Load and chunk the PDF
    try:
        raw_text = extract_text_from_pdf(PDF_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please update the PDF_PATH in config.py.")
        exit()
        
    DOCUMENTS = chunk_text(raw_text)
    print(f"Chunked text into {len(DOCUMENTS)} documents.")

    # 2. Embed and index chunks
    print("Embedding documents... (This may take a moment)")
    doc_embeddings = embedding_model.encode(DOCUMENTS, show_progress_bar=True)
    dimension = doc_embeddings.shape[1]
    
    # FAISS requires float32 for index.add
    FAISS_INDEX = faiss.IndexFlatL2(dimension)
    FAISS_INDEX.add(np.array(doc_embeddings).astype('float32')) 

    # 3. Save to disk
    try:
        faiss.write_index(FAISS_INDEX, FAISS_INDEX_PATH)
        with open(DOCUMENTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(DOCUMENTS, f, indent=4)
        print(f"Saved new index to {FAISS_INDEX_PATH} and documents to {DOCUMENTS_PATH}")
    except Exception as e:
        print(f"Error saving RAG data: {e}")

    return DOCUMENTS, FAISS_INDEX

def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Performs the retrieval part of RAG using the initialized global index.
    """
    if FAISS_INDEX is None or embedding_model is None:
        raise RuntimeError("RAG components not initialized. Call get_or_create_rag_data() first.")
        
    print(f"\n---> [RETRIEVAL START] Searching context for query: '{query}'")
    
    # Embed query
    query_embedding = embedding_model.encode([query])
    
    # Vector search
    # FAISS requires float32 for search
    distances, indices = FAISS_INDEX.search(np.array(query_embedding).astype('float32'), top_k)
    retrieved_chunks = [DOCUMENTS[i] for i in indices[0]]
    
    # Format context
    context = "\n---\n".join(retrieved_chunks)
    print(f"---> [RETRIEVAL COMPLETE] Context of {len(context)} characters returned.")
    return context