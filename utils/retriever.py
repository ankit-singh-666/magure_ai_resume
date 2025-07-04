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





def expand_query_with_keywords(query):
    keyword_map = {
        "frontend": [
            "react", "angular", "vue", "html", "css", "tailwind", "javascript", "typescript"
        ],
        "backend": [
            "java", ".net", "python", "node.js", "spring", "django", "express", "api", "server"
        ],
        "qa": [
            "automation", "testing", "quality assurance", "selenium", "postman", "jmeter",
            "cypress", "manual testing", "test cases"
        ],
        "fullstack": [
            "react", "node.js", "express", "mongodb", "sql", "python", "html", "css"
        ],
        "data": [
            "python", "pandas", "numpy", "sql", "etl", "data analysis", "data engineering",
            "data science", "machine learning", "statistics"
        ],
        "devops": [
            "docker", "kubernetes", "ci/cd", "aws", "azure", "jenkins", "terraform", "linux", "monitoring"
        ]
    }

    query_lower = query.lower()
    matched_keywords = []

    for key, terms in keyword_map.items():
        if key in query_lower:
            matched_keywords.extend(terms)

    # Remove duplicates and avoid adding terms already in query
    unique_terms = [term for term in set(matched_keywords) if term not in query_lower]

    # Combine into a single expanded query string
    expanded_query = query.strip() + " " + " ".join(unique_terms) if unique_terms else query.strip()
    return expanded_query
'''

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
    '''

def get_all_groups_with_indexes():
    """
    Scan the vector_store directory to get all groups that have both FAISS index and metadata files.
    """
    groups = []
    for filename in os.listdir(VECTOR_STORE_DIR):
        if filename.endswith("_faiss_index.index"):
            group = filename.replace("_faiss_index.index", "")
            index_path, metadata_path = get_paths_for_group(group)
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                groups.append(group)
    return groups


def retrieve_similar_chunks(query: str, k: int = 5, group: str = None):
    """
    Search FAISS index(es). If group is provided, search only in that group.
    If group is None, search all available groups and merge results.
    """
    query_embedding = model.encode([query])
    query_vector = np.array(query_embedding).astype("float32")

    all_results = []

    target_groups = [group] if group else get_all_groups_with_indexes()

    for grp in target_groups:
        try:
            index, metadata_list = load_index_and_metadata(grp)
            D, I = index.search(query_vector, k)

            for idx_pos, idx in enumerate(I[0]):
                if idx < len(metadata_list):
                    chunk = metadata_list[idx]
                    chunk["score"] = round(float(D[0][idx_pos]), 2)
                    chunk["group"] = grp
                    all_results.append(chunk)
        except FileNotFoundError:
            continue

    # Sort results across groups and return top k
    sorted_results = sorted(all_results, key=lambda x: x["score"])[:k]
    return sorted_results

