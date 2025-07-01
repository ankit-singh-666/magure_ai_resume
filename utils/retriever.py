import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

VECTOR_STORE_DIR = "vector_store"

def get_paths_for_group(group):
    safe_group = group.replace(" ", "_").lower()
    index_path = os.path.join(VECTOR_STORE_DIR, f"{safe_group}_faiss_index.index")
    metadata_path = os.path.join(VECTOR_STORE_DIR, f"{safe_group}_chunk_metadata.json")
    return index_path, metadata_path

def load_index_and_metadata(group):
    index_path, metadata_path = get_paths_for_group(group)

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No FAISS index or metadata found for group '{group}'")

    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata

def retrieve_similar_chunks(query: str, k: int = 5, group: str = "general"):
    index, metadata_list = load_index_and_metadata(group)

    # Encode query
    query_embedding = model.encode([query])
    query_vector = np.array(query_embedding).astype("float32")

    # Search FAISS
    D, I = index.search(query_vector, k)

    results = []
    for idx_pos, idx in enumerate(I[0]):
        if idx < len(metadata_list):
            chunk = metadata_list[idx]
            chunk["score"] = round(float(D[0][idx_pos]), 2)
            results.append(chunk)

    return results
