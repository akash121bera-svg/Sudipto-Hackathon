import logging
import numpy as np
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from backend.app.core.config import settings

logger = logging.getLogger("vector_store")

# Lazy load sentence transformers for embedding generation
_encoder = None
def get_encoder():
    global _encoder
    if _encoder is None:
        if settings.ENABLE_MOCK_FALLBACK:
            # Simple mock encoder returning random vectors or hashed text vectors
            class MockEncoder:
                def encode(self, sentences: Any, convert_to_tensor=False) -> np.ndarray:
                    if isinstance(sentences, str):
                        sentences = [sentences]
                    # Generate a deterministic pseudo-random vector based on string hash
                    vectors = []
                    for s in sentences:
                        val = sum(ord(c) for c in s) / 1000.0
                        np.random.seed(int(val * 1000) % 2**32)
                        vec = np.random.randn(384)
                        vec = vec / np.linalg.norm(vec)
                        vectors.append(vec)
                    return np.array(vectors)
            _encoder = MockEncoder()
        else:
            try:
                from sentence_transformers import SentenceTransformer
                _encoder = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=settings.HF_CACHE_DIR)
            except Exception as e:
                logger.warning(f"Failed to load sentence-transformers: {e}. Falling back to MockEncoder.")
                class MockEncoder:
                    def encode(self, sentences: Any, convert_to_tensor=False) -> np.ndarray:
                        if isinstance(sentences, str):
                            sentences = [sentences]
                        vectors = []
                        for s in sentences:
                            val = sum(ord(c) for c in s) / 1000.0
                            np.random.seed(int(val * 1000) % 2**32)
                            vec = np.random.randn(384)
                            vec = vec / np.linalg.norm(vec)
                            vectors.append(vec)
                        return np.array(vectors)
                _encoder = MockEncoder()
    return _encoder

class InMemoryVectorStore:
    """Fallback in-memory vector store when Qdrant is unavailable."""
    def __init__(self):
        self.stores: Dict[str, List[Dict[str, Any]]] = {}

    def recreate_collection(self, collection_name: str, vector_size: int):
        self.stores[collection_name] = []
        logger.info(f"[InMemoryDB] Recreated collection: {collection_name}")

    def upsert(self, collection_name: str, points: List[Any]):
        if collection_name not in self.stores:
            self.stores[collection_name] = []
        for point in points:
            # Handle list or object points
            p_dict = {
                "id": getattr(point, "id", None) or point.get("id"),
                "vector": getattr(point, "vector", None) or point.get("vector"),
                "payload": getattr(point, "payload", None) or point.get("payload")
            }
            # Remove existing if same ID
            self.stores[collection_name] = [item for item in self.stores[collection_name] if item["id"] != p_dict["id"]]
            self.stores[collection_name].append(p_dict)
        logger.info(f"[InMemoryDB] Upserted {len(points)} points into {collection_name}")

    def search(self, collection_name: str, query_vector: List[float], limit: int = 5) -> List[Any]:
        if collection_name not in self.stores or not self.stores[collection_name]:
            return []
        
        q_vec = np.array(query_vector)
        q_norm = np.linalg.norm(q_vec)
        
        results = []
        for item in self.stores[collection_name]:
            i_vec = np.array(item["vector"])
            i_norm = np.linalg.norm(i_vec)
            if q_norm == 0 or i_norm == 0:
                score = 0.0
            else:
                score = float(np.dot(q_vec, i_vec) / (q_norm * i_norm))
            
            # Create a ScoredPoint mock
            class ScoredPoint:
                def __init__(self, id, score, payload):
                    self.id = id
                    self.score = score
                    self.payload = payload
            
            results.append(ScoredPoint(item["id"], score, item["payload"]))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

# Connect Qdrant Client
qdrant_client = None
is_mock_qdrant = False

try:
    qdrant_client = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        timeout=3.0
    )
    # Ping
    qdrant_client.get_collections()
    logger.info("Successfully connected to Qdrant vector database.")
except Exception as e:
    logger.warning(f"Qdrant connection failed: {e}. Falling back to InMemoryVectorStore.")
    qdrant_client = InMemoryVectorStore()
    is_mock_qdrant = True

def init_vector_db():
    """Initializes collections for form templates and handwriting correction history."""
    collections = ["form_templates", "correction_memory"]
    vector_size = 384  # Dimension of all-MiniLM-L6-v2
    
    for col in collections:
        try:
            if is_mock_qdrant:
                qdrant_client.recreate_collection(col, vector_size)
            else:
                # Check if collection exists
                try:
                    qdrant_client.get_collection(col)
                    logger.info(f"Qdrant collection '{col}' already exists.")
                except Exception:
                    qdrant_client.recreate_collection(
                        collection_name=col,
                        vectors_config=qmodels.VectorParams(
                            size=vector_size,
                            distance=qmodels.Distance.COSINE
                        )
                    )
                    logger.info(f"Created Qdrant collection '{col}'.")
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection '{col}': {e}")

# Helper functions for vector memory
def store_embedding(collection: str, doc_id: str, text: str, payload: dict):
    encoder = get_encoder()
    vector = encoder.encode(text).tolist()
    
    if is_mock_qdrant:
        qdrant_client.upsert(
            collection_name=collection,
            points=[{"id": doc_id, "vector": vector, "payload": payload}]
        )
    else:
        qdrant_client.upsert(
            collection_name=collection,
            points=[
                qmodels.PointStruct(
                    id=doc_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )

def search_embeddings(collection: str, query_text: str, limit: int = 3) -> List[Dict[str, Any]]:
    encoder = get_encoder()
    vector = encoder.encode(query_text).tolist()
    
    results = qdrant_client.search(
        collection_name=collection,
        query_vector=vector,
        limit=limit
    )
    
    output = []
    for res in results:
        output.append({
            "id": res.id,
            "score": res.score,
            "payload": res.payload
        })
    return output
