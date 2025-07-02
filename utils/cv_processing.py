import os
import re
import uuid
import json
import numpy as np
import tempfile
import requests
import logging
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


logger = logging.getLogger(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
VECTOR_STORE_DIR = os.path.join(basedir, "..", "vector_store")
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)


embedding_model = SentenceTransformer('all-MiniLM-L6-v2')




def extract_text_from_pdf(path):
    """Extracts text from a PDF file at a given local path."""
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        logger.error(f"Failed to read PDF {path}: {e}")
        return ""

def clean_text(text):
    """Cleans raw text by removing extra whitespace and non-ASCII characters."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def naive_sentence_tokenize(text):
    """Splits text into sentences based on punctuation."""
    return re.split(r'(?<=[.!?]) +', text.strip())

def chunk_text(text, chunk_size=500):
    """Chunks text into smaller pieces of a maximum size."""
    sentences = naive_sentence_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def create_chunks_with_metadata(chunks, unique_file_id, group):
    """Creates a list of dictionaries, each representing a chunk with metadata."""
    return [{
        "id": str(uuid.uuid4()),
        "chunk_index": i,
        "text": chunk,
        "source_file": unique_file_id,
        "group": group
    } for i, chunk in enumerate(chunks)]




def get_paths_for_group(group):
    """Generates standardized file paths for a group's index and metadata."""
    safe_group = re.sub(r'[^a-zA-Z0-9_-]', '', group.replace(" ", "_").lower())
    index_path = os.path.join(VECTOR_STORE_DIR, f"{safe_group}_faiss_index.index")
    metadata_path = os.path.join(VECTOR_STORE_DIR, f"{safe_group}_chunk_metadata.json")
    return index_path, metadata_path



def process_and_store_embeddings(file_url, original_filename, public_id, group_name="general"):
    """
    Downloads a file from a URL, processes it, and stores its vector embeddings.
    This is the updated function to bridge Cloudinary with your local processing.
    """
    tmp_filepath = None
    try:
     
        logger.info(f"Downloading {original_filename} from Cloudinary...")
        response = requests.get(file_url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(original_filename)[1]) as tmp_file:
            tmp_file.write(response.content)
            tmp_filepath = tmp_file.name

        index_path, metadata_path = get_paths_for_group(group_name)
        raw_text = extract_text_from_pdf(tmp_filepath)
        
        if not raw_text:
            logger.warning(f"No text could be extracted from {original_filename}. Skipping embedding.")
            return

        clean = clean_text(raw_text)
        chunks = chunk_text(clean)
        chunk_metadata = create_chunks_with_metadata(chunks, public_id, group_name)

        if not chunk_metadata:
            logger.warning(f"No processable chunks found in {original_filename}.")
            return

        texts = [chunk["text"] for chunk in chunk_metadata]
        embeddings = embedding_model.encode(texts)
        embedding_matrix = np.array(embeddings).astype("float32")

      
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
        else:
            index = faiss.IndexFlatL2(embedding_matrix.shape[1])

        index.add(embedding_matrix)
        faiss.write_index(index, index_path)

  
        existing_metadata = []
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                existing_metadata = json.load(f)
        
        existing_metadata.extend(chunk_metadata)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(existing_metadata, f, indent=2)

        logger.info(f"📥 Stored {len(chunk_metadata)} chunks from {original_filename} ({public_id}) under group '{group_name}'")

    except Exception as e:
        logger.error(f"Failed to process embeddings for {original_filename}: {e}", exc_info=True)
    finally:
   
        if tmp_filepath and os.path.exists(tmp_filepath):
            os.remove(tmp_filepath)
            logger.info(f"Cleaned up temporary file: {tmp_filepath}")


def delete_cv_data(public_id, group="general"):
    """Deletes a CV's data from the metadata file and rebuilds the FAISS index."""
    logger.info(f"🗑 Deleting CV data for: {public_id} under group '{group}'")
    index_path, metadata_path = get_paths_for_group(group)

    if not os.path.exists(metadata_path):
        logger.warning(f"No metadata file found for group '{group}'. Nothing to delete.")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        all_metadata = json.load(f)

    remaining_metadata = [chunk for chunk in all_metadata if chunk.get("source_file") != public_id]

    if len(remaining_metadata) == len(all_metadata):
        logger.info(f"No chunks found for '{public_id}' to delete.")
        return

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(remaining_metadata, f, indent=2)

    if not remaining_metadata:
        if os.path.exists(index_path): os.remove(index_path)
        if os.path.exists(metadata_path): os.remove(metadata_path)
        logger.info(f"All chunks deleted for group '{group}', removed index and metadata files.")
        return

    logger.info("🔄 Rebuilding FAISS index after deletion...")
    texts = [m["text"] for m in remaining_metadata]
    embeddings = embedding_model.encode(texts)
    new_index = faiss.IndexFlatL2(embeddings.shape[1])
    new_index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(new_index, index_path)
    logger.info("✅ Deleted metadata and successfully rebuilt FAISS index.")


def retrieve_similar_chunks(query, k, group):
    """
    Searches the FAISS index for the most similar chunks to a query.
    This function is required by the search_api endpoint.
    """
    index_path, metadata_path = get_paths_for_group(group)
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No FAISS index or metadata found for group '{group}'")
        
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype("float32"), k=min(k, len(metadata)))
    
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if 0 <= idx < len(metadata):
   
            results.append({
                "page_content": metadata[idx].get("text"),
                "metadata": {
                    "source": metadata[idx].get("source_file"),
                    "group": metadata[idx].get("group"),
                    "chunk_index": metadata[idx].get("chunk_index"),
                },
                "distance": float(distances[0][i])
            })
    return results