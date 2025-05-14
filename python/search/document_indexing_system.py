from flask import Flask, request, render_template, redirect, url_for
import os
import math
import re
import numpy as np
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_files'  # Folder to store uploaded files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global data structures to store the index
idf = {}
vocab = []
doc_vectors = []
filenames = []


# Preprocess a single document
def preprocess(text):
    stemmer = PorterStemmer()
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [stemmer.stem(token) for token in tokens if token not in ENGLISH_STOP_WORDS]


# Load documents and process them for indexing
def load_documents(folder_path):
    documents = []
    file_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                documents.append(preprocess(content))
                file_names.append(filename)
    return documents, file_names


# Build inverted index and weights
def build_indices(documents):
    tf_list = []
    df = defaultdict(int)
    for i, doc in enumerate(documents):
        tf = Counter(doc)
        tf_list.append(tf)
        for term in set(doc):
            df[term] += 1
    idf_values = {term: math.log(len(documents) / df[term]) for term in df}
    return tf_list, idf_values


def compute_tf_idf(tf_list, idf):
    weights = []
    for tf in tf_list:
        weight = {term: tf.get(term, 0) * idf.get(term, 0) for term in tf}
        weights.append(weight)
    return weights


def vectorize(weights, vocab):
    vectors = []
    for weight in weights:
        vec = np.array([weight.get(term, 0.0) for term in vocab])
        vectors.append(vec)
    return vectors


# Process query and retrieve results
def process_query(query, idf, doc_vectors, vocab, filenames):
    query_terms = preprocess(query)
    query_tf = Counter(query_terms)
    query_weights = {term: query_tf.get(term, 0) * idf.get(term, 0.0) for term in query_tf}
    query_vector = np.array([query_weights.get(term, 0.0) for term in vocab]).reshape(1, -1)

    doc_matrix = np.array(doc_vectors)
    similarities = cosine_similarity(query_vector, doc_matrix).flatten()
    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    results = [{"rank": rank + 1, "document": filenames[doc_id], "similarity": sim}
               for rank, (doc_id, sim) in enumerate(ranked) if sim > 0.0]
    return results


@app.route('/', methods=['GET', 'POST'])
def index():
    global idf, vocab, doc_vectors, filenames

    if request.method == 'POST':
        # Clear old uploaded files
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))

        # Save new files
        uploaded_files = request.files.getlist('files')
        for file in uploaded_files:
            if file.filename.endswith('.txt'):
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

        # Reload documents and rebuild index
        documents, filenames = load_documents(app.config['UPLOAD_FOLDER'])
        tf_list, idf = build_indices(documents)
        weights = compute_tf_idf(tf_list, idf)
        vocab = sorted(idf.keys())
        doc_vectors = vectorize(weights, vocab)

        return redirect(url_for('index'))

    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    if not query or not doc_vectors or not vocab:
        return render_template('index.html', results=[], error="No documents indexed or query is missing.")

    results = process_query(query, idf, doc_vectors, vocab, filenames)
    return render_template('index.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)