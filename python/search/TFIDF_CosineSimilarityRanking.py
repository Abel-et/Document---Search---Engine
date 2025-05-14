import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample large collection (for a real system, load this from a database or filesystem)
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Never jump over the lazy dog quickly",
    "The fox is quick and the dog is lazy",
    "A quick brown fox jumps past a sleepy dog",
    "Lazy dogs are often jumped over quickly by agile foxes",
    # Add as many documents as necessary to simulate the large collection
]

# Query
query = "quick fox lazy dog"

### Step 1: Vectorize the Documents and Query Using TF-IDF
# TfidfVectorizer automatically tokenizes documents and calculates term weights
vectorizer = TfidfVectorizer()

# Fit the vectorizer to documents and transform them into a term-document matrix
doc_term_matrix = vectorizer.fit_transform(documents)  # Sparse matrix representation

# Transform the query into the same space (sparse query vector)
query_vector = vectorizer.transform([query])

### Step 2: Calculate Similarity Scores
# Compute cosine similarity between the query and all document vectors
similarity_scores = cosine_similarity(query_vector, doc_term_matrix).flatten()

### Step 3: Rank Documents
# Rank the documents by their similarity score
ranked_indices = np.argsort(similarity_scores)[::-1]  # Sort in descending order
ranked_documents = [(documents[i], similarity_scores[i]) for i in ranked_indices]

# Display ranked results
print("Ranked Documents (based on similarity):")
for doc, score in ranked_documents:
    print(f"Document: {doc}\nSimilarity Score: {score:.4f}\n")