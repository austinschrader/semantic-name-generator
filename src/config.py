import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for the Semantic Name Generator"""
    
    # API Settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    EMBEDDINGS_DIR = DATA_DIR / "embeddings"
    
    # Files
    NAME_CORPUS_FILE = DATA_DIR / "name_corpus.txt"
    CORPUS_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "corpus_embeddings.npy"
    CORPUS_METADATA_FILE = EMBEDDINGS_DIR / "corpus_metadata.json"
    
    # Application Settings
    DEFAULT_TOP_N = 10
    MIN_SIMILARITY_THRESHOLD = 0.0
    
    @classmethod
    def validate(cls):
        """Validate configuration settings"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Create directories if they don't exist
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.EMBEDDINGS_DIR.mkdir(exist_ok=True)
        
        return True