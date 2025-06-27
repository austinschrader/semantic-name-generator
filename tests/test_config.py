import unittest
import os
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config


class TestConfig(unittest.TestCase):
    """Test cases for Config"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Save original environment
        self.original_api_key = os.environ.get('OPENAI_API_KEY')
        
    def tearDown(self):
        """Clean up test fixtures"""
        # Restore original environment
        if self.original_api_key:
            os.environ['OPENAI_API_KEY'] = self.original_api_key
        elif 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
            
    def test_config_paths(self):
        """Test that config paths are correctly set"""
        self.assertTrue(Config.PROJECT_ROOT.exists())
        self.assertEqual(Config.DATA_DIR.name, "data")
        self.assertEqual(Config.EMBEDDINGS_DIR.name, "embeddings")
        self.assertTrue(str(Config.NAME_CORPUS_FILE).endswith("name_corpus.txt"))
        self.assertTrue(str(Config.CORPUS_EMBEDDINGS_FILE).endswith("corpus_embeddings.npy"))
        
    def test_config_settings(self):
        """Test default configuration settings"""
        self.assertEqual(Config.EMBEDDING_MODEL, "text-embedding-3-small")
        self.assertEqual(Config.DEFAULT_TOP_N, 10)
        self.assertEqual(Config.MIN_SIMILARITY_THRESHOLD, 0.0)
        
    def test_validate_with_api_key(self):
        """Test validation with API key set"""
        os.environ['OPENAI_API_KEY'] = 'test-key-123'
        
        # Should not raise an error
        result = Config.validate()
        self.assertTrue(result)
        
    def test_validate_without_api_key(self):
        """Test validation without API key raises error"""
        # Remove API key from environment
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        
        # Force reload of config to pick up environment change
        Config.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        
        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            Config.validate()
        
        self.assertIn("OPENAI_API_KEY", str(context.exception))
        
    def test_validate_creates_directories(self):
        """Test that validate creates necessary directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Temporarily override paths
            original_data_dir = Config.DATA_DIR
            original_embeddings_dir = Config.EMBEDDINGS_DIR
            
            Config.DATA_DIR = Path(temp_dir) / "test_data"
            Config.EMBEDDINGS_DIR = Config.DATA_DIR / "test_embeddings"
            
            # Set API key for validation
            os.environ['OPENAI_API_KEY'] = 'test-key'
            
            # Directories shouldn't exist yet
            self.assertFalse(Config.DATA_DIR.exists())
            self.assertFalse(Config.EMBEDDINGS_DIR.exists())
            
            # Validate should create them
            Config.validate()
            
            # Now they should exist
            self.assertTrue(Config.DATA_DIR.exists())
            self.assertTrue(Config.EMBEDDINGS_DIR.exists())
            
            # Restore original paths
            Config.DATA_DIR = original_data_dir
            Config.EMBEDDINGS_DIR = original_embeddings_dir


if __name__ == "__main__":
    unittest.main()