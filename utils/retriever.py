import os
import json
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Constants
VECTOR_STORE_DIR = "vector_store"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Model
model = SentenceTransformer("all-MiniLM-L6-v2")



def get_paths_for_group(group: str):
    safe_group = group.replace(" ", "_").lower()
    index_path = os.path.join(VECTOR_STORE_DIR, f"{safe_group}_faiss_index.index")
    metadata_path = os.path.join(VECTOR_STORE_DIR, f"{safe_group}_chunk_metadata.json")
    return index_path, metadata_path


def load_index_and_metadata(group: str):
    """
    Load the FAISS index and corresponding metadata JSON file for a group.
    """
    try:
        index_path, metadata_path = get_paths_for_group(group)

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Missing index or metadata for group '{group}'")

        index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return index, metadata

    except Exception as e:
        logger.error(f"Failed to load index/metadata for group '{group}': {e}")
        raise




def get_all_groups_with_indexes():
    """
    Get all available groups that have both FAISS index and metadata files.
    """
    groups = []
    try:
        for filename in os.listdir(VECTOR_STORE_DIR):
            if filename.endswith("_faiss_index.index"):
                group = filename.replace("_faiss_index.index", "")
                index_path, metadata_path = get_paths_for_group(group)
                if os.path.exists(index_path) and os.path.exists(metadata_path):
                    groups.append(group)
    except Exception as e:
        logger.warning(f"Error scanning vector store directory: {e}")

    return groups


def retrieve_similar_chunks(query: str, k: int = 5, group: str = None):
    """
    Retrieve top-K similar chunks from FAISS.
    If group is None, searches across all groups.
    """
    try:
        query_embedding = model.encode([query])
        query_vector = np.array(query_embedding).astype("float32")
    except Exception as e:
        logger.error(f"Failed to encode query '{query}': {e}")
        return []

    all_results = []
    target_groups = [group] if group else get_all_groups_with_indexes()

    for grp in target_groups:
        try:
            index, metadata_list = load_index_and_metadata(grp)
            D, I = index.search(query_vector, k)

            logger.info(f"[{grp}] FAISS returned indices: {I[0].tolist()}")
            logger.info(f"[{grp}] Distance scores: {D[0].tolist()}")
            logger.info(f"[{grp}] Metadata count: {len(metadata_list)}")

            for idx_pos, idx in enumerate(I[0]):
                if idx == -1 or idx >= len(metadata_list):
                    continue  # Skip invalid index

                chunk = metadata_list[idx]
                chunk["score"] = round(float(D[0][idx_pos]), 2)
                chunk["group"] = grp
                all_results.append(chunk)

        except FileNotFoundError:
            logger.info(f"Skipping missing group: {grp}")
            continue
        except Exception as e:
            logger.warning(f"Error retrieving chunks from group '{grp}': {e}")
            continue

    # Only include valid results with score and sort by similarity (lower score = better)
    valid_results = [r for r in all_results if "score" in r]
    sorted_results = sorted(valid_results, key=lambda x: x["score"])[:k]

    logger.info(f"🔍 Total valid results returned: {len(sorted_results)}")
    return sorted_results
