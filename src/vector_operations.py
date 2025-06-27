import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VectorOperations:
    """Handles vector arithmetic and similarity calculations"""
    
    @staticmethod
    def combine_vectors(vectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """Combine multiple vectors into a single vector
        
        Args:
            vectors: List of vectors to combine
            weights: Optional weights for each vector (defaults to equal weights)
            
        Returns:
            Combined vector (weighted average)
        """
        if not vectors:
            raise ValueError("Cannot combine empty list of vectors")
        
        vectors_array = np.vstack(vectors)
        
        if weights is None:
            # Equal weights
            return np.mean(vectors_array, axis=0)
        else:
            if len(weights) != len(vectors):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of vectors ({len(vectors)})")
            
            # Normalize weights
            weights_array = np.array(weights)
            weights_array = weights_array / np.sum(weights_array)
            
            # Weighted average
            return np.average(vectors_array, axis=0, weights=weights_array)
    
    @staticmethod
    def subtract_vectors(positive_vector: np.ndarray, negative_vector: np.ndarray, 
                        subtraction_weight: float = 0.5) -> np.ndarray:
        """Subtract negative vector from positive vector
        
        Args:
            positive_vector: The base positive vector
            negative_vector: The vector to subtract
            subtraction_weight: Weight for the subtraction (0-1)
            
        Returns:
            Result vector after subtraction
        """
        # Ensure subtraction weight is in valid range
        subtraction_weight = np.clip(subtraction_weight, 0.0, 1.0)
        
        # Perform weighted subtraction
        result = positive_vector - (subtraction_weight * negative_vector)
        
        # Normalize the result to maintain unit length
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        
        return result
    
    @staticmethod
    def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        # Handle zero vectors
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            logger.warning("Zero vector encountered in cosine similarity calculation")
            return 0.0
        
        # Calculate cosine similarity
        dot_product = np.dot(vector1, vector2)
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure result is in valid range due to floating point precision
        return float(np.clip(similarity, -1.0, 1.0))
    
    @staticmethod
    def batch_cosine_similarity(target_vector: np.ndarray, 
                               corpus_vectors: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between target vector and multiple corpus vectors
        
        Args:
            target_vector: The target vector to compare against
            corpus_vectors: Matrix of corpus vectors (each row is a vector)
            
        Returns:
            Array of similarity scores
        """
        # Normalize target vector
        target_norm = np.linalg.norm(target_vector)
        if target_norm == 0:
            logger.warning("Zero target vector in batch similarity calculation")
            return np.zeros(len(corpus_vectors))
        
        normalized_target = target_vector / target_norm
        
        # Normalize corpus vectors
        corpus_norms = np.linalg.norm(corpus_vectors, axis=1)
        
        # Handle zero vectors in corpus
        zero_mask = corpus_norms == 0
        if np.any(zero_mask):
            logger.warning(f"Found {np.sum(zero_mask)} zero vectors in corpus")
        
        # Avoid division by zero
        corpus_norms[zero_mask] = 1.0
        normalized_corpus = corpus_vectors / corpus_norms[:, np.newaxis]
        
        # Calculate similarities
        similarities = np.dot(normalized_corpus, normalized_target)
        
        # Set similarity to 0 for zero vectors
        similarities[zero_mask] = 0.0
        
        return similarities
    
    @staticmethod
    def get_top_similar(target_vector: np.ndarray, corpus_vectors: np.ndarray, 
                       names: List[str], top_n: int = 10, 
                       min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        """Get top N most similar items from corpus
        
        Args:
            target_vector: The target vector
            corpus_vectors: Matrix of corpus vectors
            names: List of names corresponding to corpus vectors
            top_n: Number of top results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (name, similarity_score) tuples sorted by similarity
        """
        if len(names) != len(corpus_vectors):
            raise ValueError(f"Number of names ({len(names)}) must match number of vectors ({len(corpus_vectors)})")
        
        # Calculate similarities
        similarities = VectorOperations.batch_cosine_similarity(target_vector, corpus_vectors)
        
        # Create pairs of (name, similarity)
        results = [(name, float(sim)) for name, sim in zip(names, similarities) 
                  if sim >= min_similarity]
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        return results[:top_n]