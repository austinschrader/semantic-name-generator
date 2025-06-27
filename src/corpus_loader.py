import logging
from pathlib import Path
from typing import List, Set
import re

logger = logging.getLogger(__name__)


class CorpusLoader:
    """Loads and manages the name corpus"""
    
    def __init__(self, corpus_path: Path):
        """Initialize corpus loader with path to corpus file
        
        Args:
            corpus_path: Path to the text file containing names
        """
        self.corpus_path = corpus_path
        self._names: List[str] = []
        
    def load(self) -> List[str]:
        """Load names from corpus file
        
        Returns:
            List of unique names from the corpus
        """
        if not self.corpus_path.exists():
            logger.warning(f"Corpus file not found: {self.corpus_path}")
            return []
        
        names_set: Set[str] = set()
        
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Clean and normalize the name
                    name = line.strip()
                    if name and self._is_valid_name(name):
                        names_set.add(name)
            
            self._names = sorted(list(names_set))
            logger.info(f"Loaded {len(self._names)} unique names from corpus")
            return self._names
            
        except Exception as e:
            logger.error(f"Error loading corpus: {e}")
            raise
    
    def _is_valid_name(self, name: str) -> bool:
        """Check if a name is valid for use as a brand name
        
        Args:
            name: The name to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Basic validation rules
        if len(name) < 2 or len(name) > 50:
            return False
        
        # Allow letters, numbers, spaces, hyphens, and some special characters
        if not re.match(r'^[\w\s\-&.]+$', name):
            return False
        
        return True
    
    def save(self, names: List[str], corpus_path: Path = None):
        """Save a list of names to corpus file
        
        Args:
            names: List of names to save
            corpus_path: Optional path to save to (uses instance path if not provided)
        """
        path = corpus_path or self.corpus_path
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                for name in sorted(set(names)):
                    if self._is_valid_name(name):
                        f.write(f"{name}\n")
            
            logger.info(f"Saved {len(names)} names to {path}")
            
        except Exception as e:
            logger.error(f"Error saving corpus: {e}")
            raise
    
    @property
    def names(self) -> List[str]:
        """Get the loaded names"""
        return self._names
    
    def __len__(self) -> int:
        """Get the number of names in the corpus"""
        return len(self._names)
    
    def __iter__(self):
        """Iterate over names in the corpus"""
        return iter(self._names)