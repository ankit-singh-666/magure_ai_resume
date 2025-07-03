import os
import re
import uuid
import json
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from docx import Document

VECTOR_STORE_DIR = "vector_store"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])




def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
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


def create_chunks_with_metadata(chunks, filename, group):
    chunk_data = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "id": str(uuid.uuid4()),
            "chunk_index": i,
            "text": chunk,
            "source_file": filename,
            "group": group
        }
        chunk_data.append(metadata)
    return chunk_data


def get_paths_for_group(group):
    safe_group = group.replace(" ", "_").lower()
    index_path = os.path.join(VECTOR_STORE_DIR, f"{safe_group}_faiss_index.index")
    metadata_path = os.path.join(VECTOR_STORE_DIR, f"{safe_group}_chunk_metadata.json")
    return index_path, metadata_path


def process_and_store_embeddings(pdf_path, original_filename, new_file_name, group="general"):
    ext = original_filename.rsplit(".", 1)[-1].lower()
    index_path, metadata_path = get_paths_for_group(group)
    raw_text=""
    print('type',ext)

    if ext == "pdf":
        raw_text = extract_text_from_pdf(pdf_path)
    elif ext == "docx":
        raw_text = extract_text_from_docx(pdf_path)
    else:
        raise ValueError("Unsupported file type")


    clean = clean_text(raw_text)
    chunks = chunk_text(clean)
    chunk_metadata = create_chunks_with_metadata(chunks, new_file_name, group)

    texts = [chunk["text"] for chunk in chunk_metadata]
    embeddings = model.encode(texts)
    embedding_matrix = np.array(embeddings).astype("float32")

    # Load or create FAISS index
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = faiss.IndexFlatL2(embedding_matrix.shape[1])

    index.add(embedding_matrix)
    faiss.write_index(index, index_path)

    # Append metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            existing_metadata = json.load(f)
    else:
        existing_metadata = []

    existing_metadata.extend(chunk_metadata)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(existing_metadata, f, indent=2)

    print(f"📥 Stored {len(chunk_metadata)} chunks from {original_filename} ({new_file_name}) under group '{group}'")
    return chunk_metadata


def delete_cv_data(new_file_name, group="general"):
    print(f"🗑 Deleting CV data for: {new_file_name} under group '{group}'")
    index_path, metadata_path = get_paths_for_group(group)

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print("No index or metadata file found.")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        all_metadata = json.load(f)

    # Filter out chunks for the CV
    remaining_metadata = []
    for chunk in all_metadata:
        if chunk["source_file"] != new_file_name:
            remaining_metadata.append(chunk)

    if len(remaining_metadata) == len(all_metadata):
        print("No chunks found to delete.")
        return

    # Rebuild index
    print(f"🔄 Rebuilding FAISS index after deletion...")

    texts = [m["text"] for m in remaining_metadata]
    if texts:
        embeddings = model.encode(texts, show_progress_bar=False)
        new_index = faiss.IndexFlatL2(embeddings.shape[1])
        new_index.add(np.array(embeddings).astype("float32"))
        faiss.write_index(new_index, index_path)
    else:
        os.remove(index_path)
        print("All embeddings deleted, FAISS index removed.")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(remaining_metadata, f, indent=2)

    print("✅ Deleted metadata and updated FAISS index.")
