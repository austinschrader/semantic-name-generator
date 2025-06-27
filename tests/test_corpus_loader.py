import unittest
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus_loader import CorpusLoader


class TestCorpusLoader(unittest.TestCase):
    """Test cases for CorpusLoader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir) / "test_corpus.txt"
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_valid_corpus(self):
        """Test loading a valid corpus file"""
        # Create test corpus
        test_names = ["Apple", "Banana", "Cherry", "Date"]
        with open(self.temp_path, 'w') as f:
            for name in test_names:
                f.write(f"{name}\n")
        
        # Load corpus
        loader = CorpusLoader(self.temp_path)
        names = loader.load()
        
        # Verify
        self.assertEqual(len(names), 4)
        self.assertEqual(set(names), set(test_names))
        
    def test_load_empty_file(self):
        """Test loading an empty corpus file"""
        # Create empty file
        self.temp_path.touch()
        
        # Load corpus
        loader = CorpusLoader(self.temp_path)
        names = loader.load()
        
        # Verify
        self.assertEqual(len(names), 0)
        
    def test_load_nonexistent_file(self):
        """Test loading a nonexistent corpus file"""
        loader = CorpusLoader(Path("/nonexistent/path.txt"))
        names = loader.load()
        
        # Should return empty list
        self.assertEqual(len(names), 0)
        
    def test_duplicate_removal(self):
        """Test that duplicates are removed"""
        # Create corpus with duplicates
        with open(self.temp_path, 'w') as f:
            f.write("Apple\nBanana\nApple\nCherry\nBanana\n")
        
        # Load corpus
        loader = CorpusLoader(self.temp_path)
        names = loader.load()
        
        # Verify
        self.assertEqual(len(names), 3)
        self.assertEqual(set(names), {"Apple", "Banana", "Cherry"})
        
    def test_name_validation(self):
        """Test name validation"""
        # Create corpus with invalid names
        with open(self.temp_path, 'w') as f:
            f.write("Valid Name\n")
            f.write("A\n")  # Too short
            f.write("This is a very long name that exceeds the maximum allowed length for a brand name\n")  # Too long
            f.write("Name@#$%\n")  # Invalid characters
            f.write("Good-Name\n")  # Valid with hyphen
            f.write("Name & Co.\n")  # Valid with special chars
        
        # Load corpus
        loader = CorpusLoader(self.temp_path)
        names = loader.load()
        
        # Verify only valid names are loaded
        self.assertIn("Valid Name", names)
        self.assertIn("Good-Name", names)
        self.assertIn("Name & Co.", names)
        self.assertNotIn("A", names)
        self.assertNotIn("Name@#$%", names)
        
    def test_save_corpus(self):
        """Test saving a corpus"""
        # Save names
        test_names = ["Zebra", "Elephant", "Lion"]
        loader = CorpusLoader(self.temp_path)
        loader.save(test_names)
        
        # Load and verify
        saved_names = loader.load()
        self.assertEqual(set(saved_names), set(test_names))
        
    def test_iterator(self):
        """Test iterator functionality"""
        # Create corpus
        test_names = ["Alpha", "Beta", "Gamma"]
        with open(self.temp_path, 'w') as f:
            for name in test_names:
                f.write(f"{name}\n")
        
        # Load corpus
        loader = CorpusLoader(self.temp_path)
        loader.load()
        
        # Test iteration
        iterated_names = list(loader)
        self.assertEqual(set(iterated_names), set(test_names))
        
        # Test len
        self.assertEqual(len(loader), 3)


if __name__ == "__main__":
    unittest.main()