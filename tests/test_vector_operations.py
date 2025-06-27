import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_operations import VectorOperations


class TestVectorOperations(unittest.TestCase):
    """Test cases for VectorOperations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.vec_ops = VectorOperations()
        
        # Create test vectors
        self.vector1 = np.array([1.0, 0.0, 0.0])
        self.vector2 = np.array([0.0, 1.0, 0.0])
        self.vector3 = np.array([0.0, 0.0, 1.0])
        self.zero_vector = np.array([0.0, 0.0, 0.0])
        
    def test_combine_vectors_equal_weights(self):
        """Test combining vectors with equal weights"""
        vectors = [self.vector1, self.vector2, self.vector3]
        result = self.vec_ops.combine_vectors(vectors)
        
        expected = np.array([1/3, 1/3, 1/3])
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_combine_vectors_with_weights(self):
        """Test combining vectors with custom weights"""
        vectors = [self.vector1, self.vector2]
        weights = [0.75, 0.25]
        result = self.vec_ops.combine_vectors(vectors, weights)
        
        expected = np.array([0.75, 0.25, 0.0])
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_combine_vectors_empty_list(self):
        """Test combining empty list of vectors raises error"""
        with self.assertRaises(ValueError):
            self.vec_ops.combine_vectors([])
            
    def test_combine_vectors_mismatched_weights(self):
        """Test combining vectors with mismatched weights raises error"""
        vectors = [self.vector1, self.vector2]
        weights = [0.5]  # Only one weight for two vectors
        
        with self.assertRaises(ValueError):
            self.vec_ops.combine_vectors(vectors, weights)
            
    def test_subtract_vectors(self):
        """Test vector subtraction"""
        positive = np.array([1.0, 1.0, 0.0])
        negative = np.array([0.5, 0.0, 0.0])
        
        result = self.vec_ops.subtract_vectors(positive, negative, subtraction_weight=1.0)
        
        # Result should be normalized
        expected_unnormalized = np.array([0.5, 1.0, 0.0])
        expected = expected_unnormalized / np.linalg.norm(expected_unnormalized)
        
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_subtract_vectors_partial_weight(self):
        """Test vector subtraction with partial weight"""
        positive = np.array([1.0, 1.0, 0.0])
        negative = np.array([1.0, 0.0, 0.0])
        
        result = self.vec_ops.subtract_vectors(positive, negative, subtraction_weight=0.5)
        
        # With 0.5 weight: [1, 1, 0] - 0.5 * [1, 0, 0] = [0.5, 1, 0]
        expected_unnormalized = np.array([0.5, 1.0, 0.0])
        expected = expected_unnormalized / np.linalg.norm(expected_unnormalized)
        
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity for orthogonal vectors"""
        similarity = self.vec_ops.cosine_similarity(self.vector1, self.vector2)
        self.assertAlmostEqual(similarity, 0.0)
        
    def test_cosine_similarity_identical(self):
        """Test cosine similarity for identical vectors"""
        similarity = self.vec_ops.cosine_similarity(self.vector1, self.vector1)
        self.assertAlmostEqual(similarity, 1.0)
        
    def test_cosine_similarity_opposite(self):
        """Test cosine similarity for opposite vectors"""
        opposite = -self.vector1
        similarity = self.vec_ops.cosine_similarity(self.vector1, opposite)
        self.assertAlmostEqual(similarity, -1.0)
        
    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector"""
        similarity = self.vec_ops.cosine_similarity(self.vector1, self.zero_vector)
        self.assertEqual(similarity, 0.0)
        
    def test_batch_cosine_similarity(self):
        """Test batch cosine similarity calculation"""
        target = self.vector1
        corpus = np.vstack([self.vector1, self.vector2, self.vector3, -self.vector1])
        
        similarities = self.vec_ops.batch_cosine_similarity(target, corpus)
        
        expected = np.array([1.0, 0.0, 0.0, -1.0])
        np.testing.assert_array_almost_equal(similarities, expected)
        
    def test_batch_cosine_similarity_with_zero_vectors(self):
        """Test batch cosine similarity with zero vectors in corpus"""
        target = self.vector1
        corpus = np.vstack([self.vector1, self.zero_vector, self.vector2])
        
        similarities = self.vec_ops.batch_cosine_similarity(target, corpus)
        
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(similarities, expected)
        
    def test_get_top_similar(self):
        """Test getting top similar items"""
        target = self.vector1
        corpus = np.vstack([
            self.vector2,  # Orthogonal
            self.vector1,  # Identical
            0.9 * self.vector1 + 0.1 * self.vector2,  # Very similar
            -self.vector1  # Opposite
        ])
        names = ["Orthogonal", "Identical", "Similar", "Opposite"]
        
        results = self.vec_ops.get_top_similar(target, corpus, names, top_n=3)
        
        # Should be sorted by similarity
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], "Identical")
        self.assertAlmostEqual(results[0][1], 1.0, places=5)
        self.assertEqual(results[1][0], "Similar")
        self.assertTrue(results[1][1] > 0.9)
        
    def test_get_top_similar_with_threshold(self):
        """Test getting top similar items with minimum threshold"""
        target = self.vector1
        corpus = np.vstack([self.vector1, self.vector2, 0.5 * self.vector1 + 0.5 * self.vector2])
        names = ["Identical", "Orthogonal", "Mixed"]
        
        results = self.vec_ops.get_top_similar(
            target, corpus, names, top_n=10, min_similarity=0.5
        )
        
        # Only "Identical" and "Mixed" should pass the threshold
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "Identical")
        self.assertEqual(results[1][0], "Mixed")
        
    def test_get_top_similar_mismatched_lengths(self):
        """Test get_top_similar with mismatched names and vectors raises error"""
        target = self.vector1
        corpus = np.vstack([self.vector1, self.vector2])
        names = ["Only One Name"]
        
        with self.assertRaises(ValueError):
            self.vec_ops.get_top_similar(target, corpus, names)


if __name__ == "__main__":
    unittest.main()