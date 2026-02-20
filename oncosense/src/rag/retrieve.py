"""
Evidence retrieval module for OncoSense RAG system.
Handles query-based retrieval from the knowledge base.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json


def load_config(config_path: str = "configs/rag.yaml") -> dict:
    """Load RAG configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class EvidenceRetriever:
    """
    Evidence retriever for RAG system.
    
    Retrieves relevant evidence chunks based on query embeddings.
    """
    
    def __init__(
        self,
        kb_dir: str = "knowledge_base",
        config_path: str = "configs/rag.yaml"
    ):
        """
        Initialize retriever.
        
        Args:
            kb_dir: Directory containing knowledge base files.
            config_path: Path to RAG configuration.
        """
        try:
            self.config = load_config(config_path)
        except FileNotFoundError:
            self.config = {
                "embedding": {"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
                "retrieval": {"top_k": 5, "similarity_threshold": 0.5}
            }
        
        kb_path = Path(kb_dir)
        chunks_path = kb_path / "chunks.json"
        index_path = kb_path / "faiss_index.bin"
        
        # Load chunks
        with open(chunks_path, "r") as f:
            self.chunks = json.load(f)
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Initialize encoder
        self.encoder = SentenceTransformer(
            self.config["embedding"]["model_name"]
        )
        
        # Retrieval settings
        self.top_k = self.config["retrieval"]["top_k"]
        self.similarity_threshold = self.config["retrieval"]["similarity_threshold"]
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        category_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant evidence for a query.
        
        Args:
            query: Search query.
            top_k: Number of results (default from config).
            category_filter: Optional category to filter by.
            
        Returns:
            List of evidence chunks with similarity scores.
        """
        top_k = top_k or self.top_k
        
        # Encode query
        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True
        ).astype(np.float32)
        
        # Search index (get more results if filtering)
        search_k = top_k * 3 if category_filter else top_k
        scores, indices = self.index.search(query_embedding, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[idx].copy()
            chunk["similarity"] = float(score)
            
            # Apply category filter
            if category_filter and chunk.get("category") != category_filter:
                continue
            
            results.append(chunk)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def retrieve_for_prediction(
        self,
        predicted_class: str,
        confidence: float,
        abstained: bool = False,
        top_k_per_category: int = 2
    ) -> Dict[str, List[Dict]]:
        """
        Retrieve evidence relevant to a model prediction.
        
        Args:
            predicted_class: Predicted class name.
            confidence: Prediction confidence.
            abstained: Whether model abstained.
            top_k_per_category: Results per category.
            
        Returns:
            Dictionary mapping categories to evidence lists.
        """
        results = {}
        
        # Build queries for each required field
        queries = {
            "predicted_class_description": f"What is {predicted_class}? Description and characteristics of {predicted_class} tumors.",
            "typical_imaging_features": f"MRI imaging features of {predicted_class}. What does {predicted_class} look like on brain MRI?",
            "general_next_steps_suggestion": f"Next steps when {predicted_class} is suspected. Follow-up and management considerations.",
            "uncertainty_note": f"Model uncertainty and confidence interpretation. When to trust AI predictions."
        }
        
        for category, query in queries.items():
            evidence = self.retrieve(
                query,
                top_k=top_k_per_category,
                category_filter=category
            )
            results[category] = evidence
        
        return results
    
    def retrieve_multi_query(
        self,
        queries: List[str],
        top_k: int = 5,
        deduplicate: bool = True
    ) -> List[Dict]:
        """
        Retrieve evidence for multiple queries with deduplication.
        
        Args:
            queries: List of search queries.
            top_k: Total number of results to return.
            deduplicate: Whether to remove duplicate chunks.
            
        Returns:
            List of unique evidence chunks.
        """
        all_results = []
        seen_ids = set()
        
        for query in queries:
            results = self.retrieve(query, top_k=top_k)
            
            for chunk in results:
                chunk_id = chunk["evidence_id"]
                
                if deduplicate and chunk_id in seen_ids:
                    continue
                
                seen_ids.add(chunk_id)
                all_results.append(chunk)
        
        # Sort by similarity and return top_k
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        return all_results[:top_k]
    
    def get_chunk_by_id(self, evidence_id: int) -> Optional[Dict]:
        """
        Get a specific chunk by its evidence ID.
        
        Args:
            evidence_id: The evidence ID to look up.
            
        Returns:
            Chunk dictionary or None if not found.
        """
        for chunk in self.chunks:
            if chunk["evidence_id"] == evidence_id:
                return chunk
        return None


def retrieve_evidence(
    predicted_class: str,
    confidence: float,
    kb_dir: str = "knowledge_base",
    config_path: str = "configs/rag.yaml"
) -> Dict[str, List[Dict]]:
    """
    Convenience function to retrieve evidence for a prediction.
    
    Args:
        predicted_class: Predicted class name.
        confidence: Prediction confidence.
        kb_dir: Knowledge base directory.
        config_path: Config path.
        
    Returns:
        Dictionary of evidence by category.
    """
    retriever = EvidenceRetriever(kb_dir, config_path)
    return retriever.retrieve_for_prediction(predicted_class, confidence)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test evidence retrieval")
    parser.add_argument("query", type=str, nargs="?", default="glioma imaging features",
                        help="Search query")
    parser.add_argument("--kb-dir", "-k", type=str, default="knowledge_base",
                        help="Knowledge base directory")
    parser.add_argument("--top-k", "-n", type=int, default=5,
                        help="Number of results")
    
    args = parser.parse_args()
    
    retriever = EvidenceRetriever(args.kb_dir)
    results = retriever.retrieve(args.query, args.top_k)
    
    print(f"Query: {args.query}")
    print(f"Results ({len(results)}):\n")
    
    for i, result in enumerate(results):
        print(f"{i+1}. [{result['similarity']:.3f}] {result['title']}")
        print(f"   Category: {result.get('category', 'N/A')}")
        print(f"   {result['text'][:150]}...")
        print()
