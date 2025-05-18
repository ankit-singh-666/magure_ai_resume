import os
import re
import uuid
import json
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

VECTOR_STORE_DIR = "vector_store"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.index")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "chunk_metadata.json")

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = text.strip()
    return text

def naive_sentence_tokenize(text):
    return re.split(r'(?<=[.!?]) +', text.strip())

def chunk_text(text, chunk_size=500):
    sentences = naive_sentence_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def create_chunks_with_metadata(chunks, filename):
    chunk_data = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "id": str(uuid.uuid4()),
            "chunk_index": i,
            "text": chunk,
            "source_file": filename
        }
        chunk_data.append(metadata)
    return chunk_data


def process_and_store_embeddings(pdf_path, original_filename,new_file_name):
    print(f"📄 Processing {original_filename}")
    raw_text = extract_text_from_pdf(pdf_path)
    clean = clean_text(raw_text)
    chunks = chunk_text(clean)
    chunk_metadata = create_chunks_with_metadata(chunks, new_file_name)

    # Encode
    texts = [chunk["text"] for chunk in chunk_metadata]
    embeddings = model.encode(texts)
    embedding_matrix = np.array(embeddings).astype("float32")

    # Load or create FAISS index
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        index = faiss.IndexFlatL2(embedding_matrix.shape[1])

    index.add(embedding_matrix)
    faiss.write_index(index, INDEX_PATH)

    # Append metadata
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            existing_metadata = json.load(f)
    else:
        existing_metadata = []

    existing_metadata.extend(chunk_metadata)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_metadata, f, indent=2)

    print(f"✅ Stored {len(chunk_metadata)} chunks from {original_filename}({new_file_name})")
    return chunk_metadata

def delete_cv_data(new_file_name):
    print(f"🗑 Deleting CV data for: {new_file_name}")

    # Load metadata
    if not os.path.exists(METADATA_PATH) or not os.path.exists(INDEX_PATH):
        print("⚠️ No index or metadata file found.")
        return

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        all_metadata = json.load(f)

    # Filter out chunks for the CV
    remaining_metadata = []
    delete_indices = []

    for i, chunk in enumerate(all_metadata):
        if chunk["source_file"] == new_file_name:
            delete_indices.append(i)
        else:
            remaining_metadata.append(chunk)

    if not delete_indices:
        print("❌ No chunks found to delete.")
        return

    # Reload original FAISS index
    index = faiss.read_index(INDEX_PATH)

    print(f"🔁 Rebuilding FAISS index after deleting {len(delete_indices)} entries...")

    # Rebuild index from kept embeddings
    texts = [m["text"] for m in remaining_metadata]
    if texts:
        embeddings = model.encode(texts, show_progress_bar=False)
        new_index = faiss.IndexFlatL2(embeddings.shape[1])
        new_index.add(np.array(embeddings).astype("float32"))
        faiss.write_index(new_index, INDEX_PATH)
    else:
        # No embeddings left
        os.remove(INDEX_PATH)
        print("⚠️ All embeddings deleted, FAISS index removed.")

    # Save updated metadata
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(remaining_metadata, f, indent=2)

    print(f"✅ Deleted metadata and updated FAISS index.")
