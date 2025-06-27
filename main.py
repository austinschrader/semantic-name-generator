#!/usr/bin/env python3
"""
Semantic Name Generator
A tool for generating brand names based on semantic similarity
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.embedding_generator import EmbeddingGenerator
from src.corpus_loader import CorpusLoader
from src.vector_operations import VectorOperations

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SemanticNameGenerator:
    """Main application class for semantic name generation"""
    
    def __init__(self):
        """Initialize the semantic name generator"""
        self.embedding_generator = EmbeddingGenerator()
        self.corpus_loader = CorpusLoader(Config.NAME_CORPUS_FILE)
        self.vector_ops = VectorOperations()
        self.corpus_embeddings = None
        self.corpus_names = []
        
    def setup(self):
        """Set up the generator by loading or generating corpus embeddings"""
        # Load corpus
        self.corpus_names = self.corpus_loader.load()
        if not self.corpus_names:
            logger.error("No names found in corpus. Please add names to the corpus file.")
            sys.exit(1)
        
        # Check if embeddings already exist
        if Config.CORPUS_EMBEDDINGS_FILE.exists() and Config.CORPUS_METADATA_FILE.exists():
            logger.info("Loading existing corpus embeddings...")
            data = self.embedding_generator.load_embeddings(
                Config.CORPUS_EMBEDDINGS_FILE,
                Config.CORPUS_METADATA_FILE
            )
            self.corpus_embeddings = data["embeddings"]
            
            # Verify corpus matches
            if data["names"] != self.corpus_names:
                logger.warning("Corpus has changed. Regenerating embeddings...")
                self._generate_corpus_embeddings()
        else:
            logger.info("No existing embeddings found. Generating...")
            self._generate_corpus_embeddings()
    
    def _generate_corpus_embeddings(self):
        """Generate and save embeddings for the corpus"""
        logger.info(f"Generating embeddings for {len(self.corpus_names)} names...")
        
        # Generate embeddings in batches
        batch_size = 20
        all_embeddings = []
        
        for i in range(0, len(self.corpus_names), batch_size):
            batch = self.corpus_names[i:i + batch_size]
            embeddings = self.embedding_generator.generate_embeddings_batch(batch)
            all_embeddings.extend(embeddings)
            logger.info(f"Processed {min(i + batch_size, len(self.corpus_names))}/{len(self.corpus_names)} names")
        
        # Save embeddings
        self.embedding_generator.save_embeddings(
            all_embeddings,
            self.corpus_names,
            Config.CORPUS_EMBEDDINGS_FILE,
            Config.CORPUS_METADATA_FILE
        )
        
        self.corpus_embeddings = np.vstack(all_embeddings)
    
    def generate_names(self, positive_concepts: List[str], 
                      negative_concepts: Optional[List[str]] = None,
                      top_n: int = None,
                      subtraction_weight: float = 0.5) -> List[tuple]:
        """Generate brand names based on semantic concepts
        
        Args:
            positive_concepts: List of positive concept words
            negative_concepts: Optional list of negative concept words to avoid
            top_n: Number of results to return
            subtraction_weight: Weight for negative concept subtraction (0-1)
            
        Returns:
            List of (name, similarity_score) tuples
        """
        if not positive_concepts:
            raise ValueError("At least one positive concept is required")
        
        top_n = top_n or Config.DEFAULT_TOP_N
        negative_concepts = negative_concepts or []
        
        # Generate embeddings for concepts
        logger.info("Generating embeddings for input concepts...")
        positive_embeddings = self.embedding_generator.generate_embeddings_batch(positive_concepts)
        
        # Combine positive vectors
        positive_vector = self.vector_ops.combine_vectors(positive_embeddings)
        
        # Handle negative concepts if provided
        if negative_concepts:
            negative_embeddings = self.embedding_generator.generate_embeddings_batch(negative_concepts)
            negative_vector = self.vector_ops.combine_vectors(negative_embeddings)
            target_vector = self.vector_ops.subtract_vectors(
                positive_vector, negative_vector, subtraction_weight
            )
        else:
            target_vector = positive_vector
        
        # Find similar names
        results = self.vector_ops.get_top_similar(
            target_vector,
            self.corpus_embeddings,
            self.corpus_names,
            top_n=top_n,
            min_similarity=Config.MIN_SIMILARITY_THRESHOLD
        )
        
        return results
    
    def interactive_mode(self):
        """Run the generator in interactive mode"""
        print("\n=== Semantic Name Generator ===")
        print("Generate brand names based on semantic concepts\n")
        
        while True:
            try:
                # Get positive concepts
                print("\nEnter positive concepts (comma-separated):")
                positive_input = input("> ").strip()
                if not positive_input:
                    print("Please enter at least one positive concept.")
                    continue
                
                positive_concepts = [c.strip() for c in positive_input.split(",") if c.strip()]
                
                # Get negative concepts
                print("\nEnter negative concepts to avoid (comma-separated, or press Enter to skip):")
                negative_input = input("> ").strip()
                negative_concepts = [c.strip() for c in negative_input.split(",") if c.strip()] if negative_input else []
                
                # Get number of results
                print("\nHow many names to generate? (default: 10):")
                top_n_input = input("> ").strip()
                top_n = int(top_n_input) if top_n_input.isdigit() else 10
                
                # Generate names
                print("\nGenerating names...")
                results = self.generate_names(positive_concepts, negative_concepts, top_n)
                
                # Display results
                print(f"\n=== Top {len(results)} Brand Name Suggestions ===")
                for i, (name, score) in enumerate(results, 1):
                    print(f"{i:2d}. {name:<20} (similarity: {score:.3f})")
                
                # Ask if user wants to continue
                print("\nGenerate more names? (y/n):")
                if input("> ").strip().lower() != 'y':
                    break
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\nError: {e}")


def main():
    """Main entry point"""
    try:
        # Validate configuration
        Config.validate()
        
        # Create and setup generator
        generator = SemanticNameGenerator()
        generator.setup()
        
        # Run interactive mode
        generator.interactive_mode()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import numpy as np  # Import here to ensure it's available
    main()