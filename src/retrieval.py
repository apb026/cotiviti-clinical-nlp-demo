"""
retrieval.py
Semantic document retrieval using a free, local Hugging Face embedding model
(sentence-transformers/all-MiniLM-L6-v2) and a FAISS index for vector search.

No paid APIs. No cloud calls. Everything runs locally on CPU.
"""

import json
import os
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "clinical_notes.json")


class ClinicalRetriever:
    """Loads clinical documents, embeds them, and serves semantic search via FAISS."""

    def __init__(self, data_path: str = DATA_PATH, model_name: str = EMBEDDING_MODEL_NAME):
        self.data_path = data_path
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.documents: List[Dict] = self._load_documents()
        self.index, self.embeddings = self._build_index()

    def _load_documents(self) -> List[Dict]:
        with open(self.data_path, "r") as f:
            return json.load(f)

    def _build_index(self) -> Tuple[faiss.Index, np.ndarray]:
        texts = [doc["content"] for doc in self.documents]
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        dim = embeddings.shape[1]
        # Inner product on normalized vectors == cosine similarity
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index, embeddings

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Return top_k documents most semantically similar to the query."""
        query_vec = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = dict(self.documents[idx])
            doc["similarity_score"] = float(score)
            results.append(doc)
        return results

    def get_document_by_id(self, doc_id: str) -> Dict:
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc
        raise ValueError(f"Document {doc_id} not found")
