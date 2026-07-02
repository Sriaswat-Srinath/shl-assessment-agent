import json
import os
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# Path Setup
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "shl_catalog_cleaned.json")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "shl_assessments"

# ==========================================
# Embedding Function (Local/Offline)
# ==========================================
embedding_func = embedding_functions.DefaultEmbeddingFunction()

# ==========================================
# Data Pre-processor
# ==========================================
def format_assessment_for_indexing(item: dict) -> str:
    name = item.get("name", "Unknown Assessment")
    description = item.get("description", "No description available.")
    job_levels = ", ".join(item.get("job_levels", [])) or "General Population"
    keys = ", ".join(item.get("keys", [])) or "General Assessment"
    duration = item.get("duration", "Variable")
    languages = ", ".join(item.get("languages", [])) or "English (USA) (Default)"

    return f"""
    TITLE: {name}
    TARGET_JOB_LEVELS: {job_levels}
    TEST_TYPE_CATEGORY: {keys}
    DURATION: {duration}
    LANGUAGES_AVAILABLE: {languages}
    
    DESCRIPTION: {description}
    """

def get_metadata_for_embedding(item: dict) -> dict:
    """
    ChromaDB metadata MUST have string values. We convert lists to comma-separated strings.
    """
    return {
        "entity_id": str(item.get("entity_id")),
        "name": item.get("name"),
        "url": item.get("link"),
        # Convert list to string for ChromaDB compliance
        "test_type": ", ".join(item.get("keys", [])), 
        # Convert list to string for ChromaDB compliance
        "job_levels": ", ".join(item.get("job_levels", [])),
        "duration": item.get("duration", "")
    }

# ==========================================
# Vector Store Initialization
# ==========================================
def init_vector_store():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Catalog JSON not found at {DATA_PATH}")

    with open(DATA_PATH, 'r') as f:
        catalog = json.load(f)

    persistent_client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Delete existing collection if it exists
    try:
        persistent_client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
    except ValueError:
        print(f"Collection '{COLLECTION_NAME}' not found, creating new one.")
        pass

    collection = persistent_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    documents = []
    metadatas = []
    ids = []

    for idx, item in enumerate(catalog):
        documents.append(format_assessment_for_indexing(item))
        metadatas.append(get_metadata_for_embedding(item))
        ids.append(str(item.get("entity_id", idx)))

    print(f"Indexing {len(documents)} assessments...")
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print("Indexing complete!")
    return collection

collection = None

def get_collection():
    global collection
    if collection is None:
        try:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            collection = client.get_collection(COLLECTION_NAME)
            print(f"Loaded existing collection '{COLLECTION_NAME}'")
        except ValueError:
            print(f"Collection '{COLLECTION_NAME}' not found, initializing...")
            collection = init_vector_store()
    return collection

# ==========================================
# Retrieval with Strict Filtering
# ==========================================
def retrieve_assessments(query: str, extracted_filters: dict, top_k: int = 20) -> List[Dict]:
    if not query:
        return []

    coll = get_collection()
    
    # CRITICAL CHANGE: Increase top_k to ensure we grab OPQ and DSI if they exist in the context
    results = coll.query(query_texts=[query], n_results=30) 
    candidates = results['metadatas'][0] if results['metadatas'] else []

    user_job_level = extracted_filters.get("job_level", "General Population").lower()

    if user_job_level in ["", "any", "general", "all"]:
        return candidates

    filtered_candidates = []
    
    level_map = {
        "senior": ["Senior", "Manager", "Director", "Executive", "Front Line Manager", "Professional Individual Contributor"],
        "director": ["Director", "Executive"],
        "executive": ["Executive", "Director"],
        "manager": ["Manager", "Front Line Manager"],
        "graduate": ["Graduate", "Entry-Level"],
        "entry": ["Entry-Level", "Graduate"],
        "mid": ["Mid-Professional"],
        "professional": ["Professional Individual Contributor"]
    }

    catalog_levels_to_keep = []
    for key, vals in level_map.items():
        if key in user_job_level:
            catalog_levels_to_keep.extend(vals)

    if not catalog_levels_to_keep:
        catalog_levels_to_keep = [user_job_level.title()]

    for item in candidates:
        item_levels_str = item.get('job_levels', '')
        if any(level in item_levels_str for level in catalog_levels_to_keep):
            filtered_candidates.append(item)

    return filtered_candidates

# ==========================================
# URL Validation (Prevents Hallucinations)
# ==========================================
def validate_url_exists(url: str) -> bool:
    if not os.path.exists(DATA_PATH):
        return False
    with open(DATA_PATH, 'r') as f:
        catalog = json.load(f)
    for item in catalog:
        if item.get("link") == url:
            return True
    return False