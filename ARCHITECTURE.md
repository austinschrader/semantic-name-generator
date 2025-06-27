Semantic Name Generator - Local Pure Python Backend

1. Overview
   This document outlines the architecture for a local-only, pure Python backend application designed to generate brand name ideas based on semantic similarity and exclusion, leveraging pre-trained OpenAI text embeddings. The goal is to provide a minimalist, functional prototype.

2. Goals
   Simplicity: Minimal dependencies, straightforward logic.

Local Only: No external web services beyond the initial OpenAI API call for embeddings (which can be pre-generated if needed for true offline operation after the first run).

Pure Python Backend: No complex frameworks, just standard library and a few core scientific computing libraries.

Semantic Filtering: Allow users to input positive concept words and negative exclusion words.

Similarity Search: Identify potential names that align with positive concepts while avoiding negative ones.

3. Architecture
   The system is composed of four main conceptual modules:

Code snippet

graph TD
A[User Input] --> B[Embedding Generator]
B --> C[Vector Operations]
C --> D[Name Corpus Embeddings]
D -- (Similarity Search) --> C
C --> E[Name Output]
3.1. Components
User Input Module:

Responsibility: Collects "positive concept" keywords and "negative exclusion" keywords from the user.

Implementation: Simple Python input() prompts or command-line arguments.

Embedding Generator:

Responsibility: Converts raw text (user keywords, and later, the name corpus) into numerical vector embeddings.

Implementation: Utilizes the openai Python library to call the OpenAI Embeddings API. Requires an API key.

Note: For true "local only" after initial setup, embeddings for the name corpus would be generated once and saved locally (e.g., as a .npy file). User input embeddings would still require an API call unless a local embedding model is integrated (which goes beyond "pure simple Python" for this scope).

Name Corpus & Embeddings Storage:

Responsibility: Stores a list of potential brand names and their pre-computed embeddings.

Implementation:

Name Corpus: A simple Python list of strings (e.g., common words, abstract terms, dictionary words).

Embeddings: A NumPy array (.npy file) containing the vectors for each name in the corpus.

Vector Operations Module:

Responsibility: Performs the core mathematical operations on vectors (combination, subtraction, similarity calculation).

Implementation: Uses numpy for efficient array operations.

Vector Combination: Averaging positive concept vectors.

Vector Subtraction: Subtracting negative concept vectors from the combined positive vector.

Cosine Similarity: Calculating the similarity between the resulting target vector and each name vector in the corpus.

Name Output Module:

Responsibility: Presents the most semantically relevant names to the user.

Implementation: Simple Python print() statements, possibly sorting names by similarity score.

4. Data Flow
   User provides positive and negative keywords.

Keywords are sent to the OpenAI Embeddings API to get their vectors.

Positive keyword vectors are averaged to form a "positive centroid."

Negative keyword vectors are averaged to form a "negative centroid."

The negative centroid is subtracted from the positive centroid to get the "target vector."

The target vector is compared (using cosine similarity) against the pre-computed embeddings of all names in the corpus.

Names are ranked by similarity score, and the top N are displayed to the user.

5. Key Technologies
   Python 3.x: Core language.

openai library: For interacting with OpenAI Embeddings API.

numpy library: For efficient vector arithmetic and similarity calculations.

Standard Python file I/O: For loading/saving name corpus and embeddings.

6. Development Considerations
   OpenAI API Key: Required for Embedding Generator. Must be handled securely (e.g., environment variable, not hardcoded).

Name Corpus Size: A larger corpus yields more diverse results but increases memory usage and initial embedding generation time.

Pre-computation: To minimize repeated API calls and ensure "local only" after setup, pre-compute embeddings for the entire name corpus once and save them.

Similarity Metric: Cosine similarity is standard for embeddings.

Thresholding/Ranking: Decide how many and which names to show (e.g., top 10, or all above a certain similarity score).

7. Future Enhancements (Beyond Simple Scope)
   Web interface (Flask/FastAPI).

More sophisticated vector combination/subtraction methods.

Integration with local embedding models (e.g., from Hugging Face).

Automated domain/trademark availability checks.

Fuzzy matching or phonetic similarity.

User feedback loop for refinement.
