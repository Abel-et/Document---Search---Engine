import os
import math
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

def preprocess(text):
    tokens = re.findall(r'\w+', text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return stemmed_tokens

def read_corpus(corpus_dir):
    documents = []
    filenames = []
    for filename in sorted(os.listdir(corpus_dir)):
        if filename.endswith('.txt'):
            filenames.append(filename)
            with open(os.path.join(corpus_dir, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                processed_tokens = preprocess(text)
                documents.append(processed_tokens)
    return documents, filenames

def compute_tf(documents):
    tf = [defaultdict(int) for _ in documents]
    for i, doc in enumerate(documents):
        for term in doc:
            tf[i][term] += 1
    return tf

def compute_idf(documents, index_terms):
    df = defaultdict(int)
    N = len(documents)
    for term in index_terms:
        df[term] = sum(1 for doc in documents if term in doc)
    idf = {}
    for term in index_terms:
        idf[term] = math.log(N / df[term]) if df[term] > 0 else 0.0
    return idf

def compute_tfidf(tf, idf, index_terms):
    tfidf = [defaultdict(float) for _ in tf]
    for i in range(len(tf)):
        for term in index_terms:
            tf_val = tf[i].get(term, 0)
            tfidf[i][term] = tf_val * idf.get(term, 0.0)
    return tfidf

def cosine_similarity(query_vec, doc_vec):
    dot_product = 0.0
    query_norm = 0.0
    doc_norm = 0.0
    all_terms = set(query_vec.keys()).union(doc_vec.keys())
    for term in all_terms:
        q_weight = query_vec.get(term, 0.0)
        d_weight = doc_vec.get(term, 0.0)
        dot_product += q_weight * d_weight
        query_norm += q_weight ** 2
        doc_norm += d_weight ** 2
    query_norm = math.sqrt(query_norm) if query_norm != 0 else 0.0
    doc_norm = math.sqrt(doc_norm) if doc_norm != 0 else 0.0
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    return dot_product / (query_norm * doc_norm)

def main():
    corpus_dir = '../project/corpus'
    documents, filenames = read_corpus(corpus_dir)
    if not documents:
        print("No documents found in the corpus directory.")
        return

    index_terms = sorted(set(term for doc in documents for term in doc))
    num_index_terms = len(index_terms)
    print(f"Number of index terms identified: {num_index_terms}")

    doc_lengths = [len(doc) for doc in documents]
    print("\nDocument lengths:")
    for i, length in enumerate(doc_lengths):
        print(f"{filenames[i]}: {length}")

    tf = compute_tf(documents)
    print("\nTerm Frequencies (TF):")
    for i, doc_tf in enumerate(tf):
        print(f"\nDocument {filenames[i]}:")
        for term in index_terms:
            count = doc_tf.get(term, 0)
            if count > 0:
                print(f"{term}: {count}")

    idf = compute_idf(documents, index_terms)
    print("\nInverse Document Frequencies (IDF):")
    for term in index_terms:
        print(f"{term}: {idf[term]:.4f}")

    tfidf = compute_tfidf(tf, idf, index_terms)
    print("\nTF-IDF Weights:")
    for i, doc_weights in enumerate(tfidf):
        print(f"\nDocument {filenames[i]}:")
        for term in index_terms:
            weight = doc_weights.get(term, 0.0)
            if weight > 0:
                print(f"{term}: {weight:.4f}")

    queries = ["machine learning", "information retrieval", "data mining"]
    print("\nQueries:")
    for i, q in enumerate(queries):
        print(f"Query {i+1}: {q}")

    preprocessed_queries = [preprocess(q) for q in queries]
    query_vectors = []
    for pq in preprocessed_queries:
        q_tf = defaultdict(int)
        for term in pq:
            q_tf[term] += 1
        q_vector = {term: q_tf[term] * idf.get(term, 0.0) for term in q_tf}
        query_vectors.append(q_vector)

    similarities = []
    for qv in query_vectors:
        sims = []
        for i, dv in enumerate(tfidf):
            sim = cosine_similarity(qv, dv)
            sims.append((i, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        similarities.append(sims)

    print("\nCosine Similarities:")
    for q_idx, sims in enumerate(similarities):
        print(f"\nQuery {q_idx+1} ({queries[q_idx]}):")
        for doc_idx, sim in sims:
            print(f"Document {filenames[doc_idx]}: {sim:.4f}")

    print("\nDocument Ranks:")
    for q_idx, sims in enumerate(similarities):
        print(f"\nQuery {q_idx+1} ({queries[q_idx]}):")
        for rank, (doc_idx, sim) in enumerate(sims):
            print(f"Rank {rank+1}: Document {filenames[doc_idx]} ({sim:.4f})")

if __name__ == "__main__":
    main()