import numpy as np
from typing import List, Dict, Any
import json
from pathlib import Path
import logging

from openai import OpenAI
from .config import Config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Handles generation of embeddings using OpenAI API"""
    
    def __init__(self):
        """Initialize the embedding generator with OpenAI client"""
        Config.validate()
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.EMBEDDING_MODEL
        
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text string
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array containing the embedding vector
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embedding = response.data[0].embedding
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Error generating embedding for '{text}': {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts in a batch
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of numpy arrays containing embedding vectors
        """
        if not texts:
            return []
            
        try:
            # OpenAI API can handle batch requests
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            embeddings = [np.array(data.embedding) for data in response.data]
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def save_embeddings(self, embeddings: List[np.ndarray], names: List[str],
                       embeddings_path: Path, metadata_path: Path):
        """Save embeddings and metadata to files
        
        Args:
            embeddings: List of embedding vectors
            names: List of corresponding names
            embeddings_path: Path to save numpy array
            metadata_path: Path to save metadata JSON
        """
        # Convert list of arrays to single numpy array
        embeddings_array = np.vstack(embeddings)
        
        # Save embeddings
        np.save(embeddings_path, embeddings_array)
        logger.info(f"Saved embeddings to {embeddings_path}")
        
        # Save metadata
        metadata = {
            "model": self.model,
            "num_embeddings": len(embeddings),
            "embedding_dim": embeddings_array.shape[1],
            "names": names
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load_embeddings(self, embeddings_path: Path, metadata_path: Path) -> Dict[str, Any]:
        """Load embeddings and metadata from files
        
        Args:
            embeddings_path: Path to numpy array file
            metadata_path: Path to metadata JSON file
            
        Returns:
            Dictionary containing embeddings array and metadata
        """
        # Load embeddings
        embeddings = np.load(embeddings_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return {
            "embeddings": embeddings,
            "names": metadata["names"],
            "model": metadata["model"],
            "num_embeddings": metadata["num_embeddings"],
            "embedding_dim": metadata["embedding_dim"]
        }