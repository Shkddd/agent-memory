"""
Text embedding utilities using sentence-transformers
"""
import logging
from typing import Union, List
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingUtil:
    """Text-to-vector conversion utility"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Args: 
            model_name: HuggingFace model identifier
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector
        
        Args:
            text: Input text string
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding. astype(np.float32)
        except Exception as e: 
            logger.error(f"Error encoding text:  {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Convert multiple texts to embeddings (batch mode)
        
        Args: 
            texts: List of text strings
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
            return embeddings. astype(np.float32)
        except Exception as e: 
            logger.error(f"Error batch encoding texts: {e}")
            raise
    
    def get_vector_dim(self) -> int:
        """Get embedding vector dimension"""
        return self.model.get_sentence_embedding_dimension()
