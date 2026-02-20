"""
Knowledge Base builder for OncoSense RAG system.
Creates and manages the FAISS index for evidence retrieval.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import yaml

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_config(config_path: str = "configs/rag.yaml") -> dict:
    """Load RAG configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Default knowledge base content for brain tumor classification
DEFAULT_KB_CHUNKS = [
    {
        "evidence_id": 1,
        "title": "Glioma Imaging Characteristics",
        "source": "Radiology Review",
        "category": "predicted_class_description",
        "text": "Gliomas are the most common primary brain tumors arising from glial cells. On MRI, they typically appear as irregular, infiltrative masses with heterogeneous signal intensity. High-grade gliomas often show ring enhancement, central necrosis, and significant surrounding edema."
    },
    {
        "evidence_id": 2,
        "title": "Glioma Location and Features",
        "source": "Neuroradiology Handbook",
        "category": "typical_imaging_features",
        "text": "Gliomas most commonly occur in the cerebral hemispheres, particularly in the frontal and temporal lobes. Key imaging features include mass effect, midline shift in larger tumors, and invasion across white matter tracts. T2/FLAIR sequences show hyperintense signal extending beyond the enhancing tumor margin."
    },
    {
        "evidence_id": 3,
        "title": "Meningioma Imaging Characteristics",
        "source": "Radiology Review",
        "category": "predicted_class_description",
        "text": "Meningiomas are benign tumors arising from the meninges. They appear as well-circumscribed, extra-axial masses with broad dural attachment. Classic features include homogeneous enhancement, dural tail sign, and CSF cleft between the tumor and brain parenchyma."
    },
    {
        "evidence_id": 4,
        "title": "Meningioma Location and Features",
        "source": "Neuroradiology Handbook",
        "category": "typical_imaging_features",
        "text": "Meningiomas commonly occur along the falx, convexity, sphenoid wing, and parasagittal regions. They may cause hyperostosis of adjacent bone. On MRI, they are isointense to gray matter on T1 and T2, with intense homogeneous contrast enhancement. Calcification is common."
    },
    {
        "evidence_id": 5,
        "title": "Pituitary Tumor Imaging Characteristics",
        "source": "Radiology Review",
        "category": "predicted_class_description",
        "text": "Pituitary adenomas are benign tumors of the pituitary gland. Microadenomas (<10mm) appear as focal hypointense lesions on contrast-enhanced T1. Macroadenomas (>10mm) may extend superiorly compressing the optic chiasm and laterally into the cavernous sinus."
    },
    {
        "evidence_id": 6,
        "title": "Pituitary Tumor Location and Features",
        "source": "Neuroradiology Handbook",
        "category": "typical_imaging_features",
        "text": "Pituitary tumors are centered in the sella turcica. Key features include suprasellar extension (snowman appearance), deviation of the pituitary stalk, and compression of normal pituitary tissue. Dynamic contrast imaging shows delayed enhancement compared to normal pituitary."
    },
    {
        "evidence_id": 7,
        "title": "Normal Brain MRI Appearance",
        "source": "Radiology Review",
        "category": "predicted_class_description",
        "text": "A normal brain MRI shows symmetric hemispheres with preserved gray-white matter differentiation. There should be no abnormal signal intensity, mass effect, or contrast enhancement. The ventricles are normal in size and configuration."
    },
    {
        "evidence_id": 8,
        "title": "Normal Brain MRI Features",
        "source": "Neuroradiology Handbook",
        "category": "typical_imaging_features",
        "text": "Normal brain features include preserved cortical ribbon thickness, normal basal ganglia signal, patent cerebral aqueduct, and no midline shift. Age-appropriate changes such as mild cerebral atrophy in elderly patients should be considered."
    },
    {
        "evidence_id": 9,
        "title": "General Next Steps for Suspected Brain Tumor",
        "source": "Clinical Guidelines",
        "category": "general_next_steps_suggestion",
        "text": "When a brain lesion is detected, general next steps may include: clinical correlation with patient symptoms, comparison with prior imaging if available, consideration of advanced MRI sequences (spectroscopy, perfusion), and multidisciplinary discussion. Final diagnosis requires clinical-radiological-pathological correlation."
    },
    {
        "evidence_id": 10,
        "title": "Follow-up Recommendations",
        "source": "Clinical Guidelines",
        "category": "general_next_steps_suggestion",
        "text": "Imaging findings should be correlated with clinical presentation. Referral to neurology or neurosurgery may be appropriate depending on findings. Surveillance imaging intervals depend on lesion characteristics and clinical context."
    },
    {
        "evidence_id": 11,
        "title": "Understanding Model Uncertainty",
        "source": "AI Documentation",
        "category": "uncertainty_note",
        "text": "Model predictions include uncertainty estimates derived from Monte Carlo dropout inference. Higher entropy or lower confidence scores indicate increased prediction uncertainty. Cases with high uncertainty should receive additional scrutiny."
    },
    {
        "evidence_id": 12,
        "title": "Limitations of AI Analysis",
        "source": "AI Documentation",
        "category": "uncertainty_note",
        "text": "This AI system is a research prototype and is not intended for clinical diagnosis. Predictions should be interpreted in conjunction with clinical findings and expert radiological review. The system may be uncertain on atypical presentations or image quality issues."
    },
    {
        "evidence_id": 13,
        "title": "Glioma WHO Grading",
        "source": "Neuropathology Reference",
        "category": "predicted_class_description",
        "text": "Gliomas are classified by WHO grade (I-IV). Low-grade gliomas (I-II) are typically well-differentiated with better prognosis. High-grade gliomas (III-IV) show anaplastic features and aggressive behavior. Glioblastoma (Grade IV) is the most common and aggressive primary brain tumor."
    },
    {
        "evidence_id": 14,
        "title": "Meningioma WHO Grading",
        "source": "Neuropathology Reference",
        "category": "predicted_class_description",
        "text": "Most meningiomas (80-90%) are WHO Grade I (benign). Atypical meningiomas (Grade II) show increased mitotic activity. Anaplastic meningiomas (Grade III) are malignant with aggressive behavior. Imaging cannot reliably differentiate grades, though irregular borders and heterogeneity may suggest higher grade."
    },
    {
        "evidence_id": 15,
        "title": "Pituitary Adenoma Classification",
        "source": "Endocrine Radiology",
        "category": "predicted_class_description",
        "text": "Pituitary adenomas are classified as functioning (hormone-secreting) or non-functioning. Common subtypes include prolactinomas, growth hormone-secreting, and ACTH-secreting adenomas. Size classification: microadenoma (<10mm), macroadenoma (≥10mm), giant adenoma (>40mm)."
    },
    {
        "evidence_id": 16,
        "title": "Differential Diagnosis Considerations",
        "source": "Neuroradiology Guide",
        "category": "general_next_steps_suggestion",
        "text": "Differential diagnosis for brain lesions includes primary tumors, metastases, infection, inflammation, and vascular lesions. Clinical history, lesion location, imaging characteristics, and multiplicity help narrow the differential."
    },
    {
        "evidence_id": 17,
        "title": "MRI Sequence Interpretation",
        "source": "Radiology Fundamentals",
        "category": "typical_imaging_features",
        "text": "Standard brain tumor MRI protocol includes T1, T2, FLAIR, diffusion-weighted imaging (DWI), and post-contrast T1. T2/FLAIR highlights edema and tumor extent. Enhancement patterns on post-contrast imaging help characterize tumor vascularity and blood-brain barrier disruption."
    },
    {
        "evidence_id": 18,
        "title": "Calibrated Confidence Interpretation",
        "source": "AI Documentation",
        "category": "uncertainty_note",
        "text": "Model confidence scores have been calibrated using temperature scaling to improve reliability. A calibrated confidence of 80% means that among all predictions with similar confidence, approximately 80% are expected to be correct. Lower confidence warrants additional review."
    },
    {
        "evidence_id": 19,
        "title": "Abstention Criteria",
        "source": "AI Documentation",
        "category": "uncertainty_note",
        "text": "The model abstains from making predictions when confidence is below threshold or uncertainty is high. Abstention is a safety mechanism indicating the case may be challenging or outside the model's training distribution. These cases require expert review."
    },
    {
        "evidence_id": 20,
        "title": "Explanation Stability",
        "source": "AI Documentation",
        "category": "uncertainty_note",
        "text": "Grad-CAM explanations are evaluated for stability under small input perturbations. Stable explanations consistently highlight the same regions regardless of minor image variations. Unstable explanations are suppressed as they may be unreliable."
    }
]


class KnowledgeBase:
    """
    Knowledge Base manager for RAG system.
    
    Handles chunk storage, embedding, and FAISS index management.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        config_path: str = "configs/rag.yaml"
    ):
        """
        Initialize Knowledge Base.
        
        Args:
            embedding_model: Name of the sentence transformer model.
            config_path: Path to RAG configuration.
        """
        try:
            self.config = load_config(config_path)
        except FileNotFoundError:
            self.config = {"embedding": {"embedding_dim": 384}}
        
        self.embedding_model_name = embedding_model
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        self.chunks = []
        self.embeddings = None
        self.index = None
    
    def add_chunks(self, chunks: List[Dict]):
        """
        Add chunks to the knowledge base.
        
        Args:
            chunks: List of chunk dictionaries with required fields.
        """
        for chunk in chunks:
            assert "evidence_id" in chunk, "Chunk must have evidence_id"
            assert "text" in chunk, "Chunk must have text"
            self.chunks.append(chunk)
    
    def build_index(self):
        """Build FAISS index from chunks."""
        if not self.chunks:
            raise ValueError("No chunks to index")
        
        print(f"Building index for {len(self.chunks)} chunks...")
        
        # Compute embeddings
        texts = [chunk["text"] for chunk in self.chunks]
        self.embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.embeddings.astype(np.float32))
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def save(self, chunks_path: str, index_path: str):
        """
        Save knowledge base to files.
        
        Args:
            chunks_path: Path to save chunks JSON.
            index_path: Path to save FAISS index.
        """
        # Save chunks
        Path(chunks_path).parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_path, "w") as f:
            json.dump(self.chunks, f, indent=2)
        print(f"Saved {len(self.chunks)} chunks to {chunks_path}")
        
        # Save index
        if self.index is not None:
            faiss.write_index(self.index, index_path)
            print(f"Saved FAISS index to {index_path}")
    
    def load(self, chunks_path: str, index_path: str):
        """
        Load knowledge base from files.
        
        Args:
            chunks_path: Path to chunks JSON.
            index_path: Path to FAISS index.
        """
        with open(chunks_path, "r") as f:
            self.chunks = json.load(f)
        print(f"Loaded {len(self.chunks)} chunks")
        
        self.index = faiss.read_index(index_path)
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query.
            top_k: Number of results to return.
            
        Returns:
            List of (chunk, similarity_score) tuples.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True
        ).astype(np.float32)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    **self.chunks[idx],
                    "similarity": float(score)
                })
        
        return results


def build_knowledge_base(
    output_dir: str = "knowledge_base",
    config_path: str = "configs/rag.yaml",
    custom_chunks: Optional[List[Dict]] = None
) -> KnowledgeBase:
    """
    Build and save the knowledge base.
    
    Args:
        output_dir: Directory to save KB files.
        config_path: Path to configuration.
        custom_chunks: Optional custom chunks (uses default if None).
        
    Returns:
        Built KnowledgeBase instance.
    """
    try:
        config = load_config(config_path)
        embedding_model = config["embedding"]["model_name"]
    except FileNotFoundError:
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    
    kb = KnowledgeBase(embedding_model, config_path)
    
    # Add chunks
    chunks = custom_chunks if custom_chunks is not None else DEFAULT_KB_CHUNKS
    kb.add_chunks(chunks)
    
    # Build index
    kb.build_index()
    
    # Save
    output_path = Path(output_dir)
    chunks_path = str(output_path / "chunks.json")
    index_path = str(output_path / "faiss_index.bin")
    
    kb.save(chunks_path, index_path)
    
    return kb


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build knowledge base")
    parser.add_argument("--output", "-o", type=str, default="knowledge_base",
                        help="Output directory")
    parser.add_argument("--config", "-c", type=str, default="configs/rag.yaml",
                        help="Path to RAG config")
    parser.add_argument("--test-query", "-t", type=str, default=None,
                        help="Test query after building")
    
    args = parser.parse_args()
    
    kb = build_knowledge_base(args.output, args.config)
    
    if args.test_query:
        print(f"\nTest query: {args.test_query}")
        results = kb.search(args.test_query, top_k=3)
        for i, result in enumerate(results):
            print(f"\n{i+1}. [{result['similarity']:.3f}] {result['title']}")
            print(f"   {result['text'][:200]}...")
