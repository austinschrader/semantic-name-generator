# Semantic Name Generator

A Python-based tool that generates brand name suggestions using semantic similarity. The tool leverages OpenAI's text embeddings to find names that align with positive concepts while avoiding negative ones.

## Features

- Generate brand names based on semantic concepts
- Support for positive and negative concept filtering
- Pre-computed embeddings for fast name generation
- Interactive command-line interface
- Customizable name corpus
- Production-ready with comprehensive error handling and testing

## Architecture

The system follows a modular architecture as specified in `ARCHITECTURE.md`:
- **Embedding Generator**: Interfaces with OpenAI API for text embeddings
- **Corpus Loader**: Manages the name corpus
- **Vector Operations**: Handles similarity calculations and vector arithmetic
- **Main Application**: Provides the interactive CLI

## Prerequisites

- Python 3.7+
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd semantic-name-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

### Basic Usage

Run the interactive mode:
```bash
python main.py
```

Follow the prompts to:
1. Enter positive concepts (comma-separated)
2. Enter negative concepts to avoid (optional)
3. Specify the number of results (default: 10)

### Example

```
Enter positive concepts (comma-separated):
> technology, innovation, future

Enter negative concepts to avoid (comma-separated, or press Enter to skip):
> old, traditional

How many names to generate? (default: 10):
> 15

=== Top 15 Brand Name Suggestions ===
 1. Quantum              (similarity: 0.832)
 2. Nexus                (similarity: 0.815)
 3. Flux                 (similarity: 0.789)
 ...
```

## Customizing the Name Corpus

The default name corpus is located at `data/name_corpus.txt`. You can modify this file to add your own names:

1. Edit `data/name_corpus.txt`
2. Add one name per line
3. Delete the existing embeddings to regenerate them:
   ```bash
   rm -rf data/embeddings/
   ```
4. Run the application - it will automatically generate new embeddings

## Configuration

Key configuration options in `src/config.py`:
- `EMBEDDING_MODEL`: OpenAI embedding model to use (default: "text-embedding-3-small")
- `DEFAULT_TOP_N`: Default number of results to return (default: 10)
- `MIN_SIMILARITY_THRESHOLD`: Minimum similarity score threshold (default: 0.0)

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Individual test files:
- `tests/test_config.py`: Configuration validation tests
- `tests/test_corpus_loader.py`: Corpus loading and validation tests
- `tests/test_vector_operations.py`: Vector arithmetic and similarity tests

## Project Structure

```
semantic-name-generator/
├── main.py                 # Main application entry point
├── src/
│   ├── config.py          # Configuration settings
│   ├── embedding_generator.py  # OpenAI embedding generation
│   ├── corpus_loader.py   # Name corpus management
│   └── vector_operations.py    # Vector math operations
├── data/
│   ├── name_corpus.txt    # Default name corpus
│   └── embeddings/        # Cached embeddings (generated)
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variable template
└── README.md             # This file
```

## How It Works

1. **Initialization**: Loads the name corpus and generates/loads embeddings
2. **Input Processing**: Converts user concepts into embedding vectors
3. **Vector Combination**: Averages positive concept vectors
4. **Vector Subtraction**: Subtracts negative concept vectors (if provided)
5. **Similarity Search**: Finds corpus names most similar to the target vector
6. **Results**: Returns top N names ranked by cosine similarity

## Limitations

- Requires internet connection for initial embedding generation
- OpenAI API costs apply for embedding generation
- Quality depends on the name corpus and chosen concepts
- Results are based on semantic similarity, not domain availability

## License

[Add your license here]

## Contributing

[Add contribution guidelines if applicable]