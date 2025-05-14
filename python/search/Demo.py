import math
import re
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess documents
def preprocess(doc):
    stemmer = PorterStemmer()
    words = re.findall(r'\b\w+\b', doc.lower())
    filtered = [stemmer.stem(w) for w in words if w not in ENGLISH_STOP_WORDS]
    print(filtered)
    return filtered

# Load documents from file
def load_documents(file_path):
    with open(file_path, 'r') as f:
        docs = f.readlines()
    return [preprocess(doc) for doc in docs]

# Build TF, DF and IDF
def build_index(docs):
    tf = []
    df = defaultdict(int)
    for doc in docs:
        counts = Counter(doc)
        tf.append(counts)
        for term in set(doc):
            df[term] += 1
    idf = {term: math.log(len(docs) / df[term]) for term in df}
    return tf, idf

# Compute TF-IDF weights
def compute_weights(tf, idf):
    weights = []
    for doc_tf in tf:
        doc_weights = {}
        for term in doc_tf:
            doc_weights[term] = doc_tf[term] * idf.get(term, 0)
        weights.append(doc_weights)
    return weights

# Convert to vector
def vectorize(weights, vocab):
    vectors = []
    for weight_dict in weights:
        vector = np.array([weight_dict.get(term, 0.0) for term in vocab])
        vectors.append(vector)
    return vectors

# Cosine similarity for query
def rank_documents(query, doc_vectors, vocab):
    query_tokens = preprocess(query)
    query_counts = Counter(query_tokens)
    query_weights = {term: query_counts[term] * idf.get(term, 0) for term in query_counts}
    query_vector = np.array([query_weights.get(term, 0.0) for term in vocab]).reshape(1, -1)
    doc_matrix = np.array(doc_vectors)
    similarities = cosine_similarity(query_vector, doc_matrix).flatten()
    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    return ranked, similarities

# Main Execution
if __name__ == "__main__":
    import sys

    file_path = "D:\doc\Artificial Intelligence.txt" # Replace with your path
    docs = load_documents(file_path)
    tf, idf = build_index(docs)
    weights = compute_weights(tf, idf)
    vocab = sorted(idf.keys())
    doc_vectors = vectorize(weights, vocab)

    print(f"Number of index terms: {len(vocab)}")
    print(f"Document lengths: {[len(doc) for doc in docs]}")
    print("\nQuery Results:")

    queries = ["data mining", "Artificail intellegence", "vector model"]
    for i, query in enumerate(queries, 1):
        ranked, similarities = rank_documents(query, doc_vectors, vocab)
        print(f"\nQuery {i}: '{query}'")
        for rank, (doc_id, score) in enumerate(ranked, 1):
            print(f"Rank {rank}: Document {doc_id+1} (Similarity: {score:.4f})")
