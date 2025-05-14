from flask import Flask, render_template, request, send_from_directory
import os
import math
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'D:\doc'
app.config['STATIC_FOLDER'] = 'static'


def preprocess(text):
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())  # Only retain alphabetic characters
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


# Flask routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']

    # Process corpus
    documents, filenames = read_corpus(app.config['UPLOAD_FOLDER'])
    index_terms = sorted(set(term for doc in documents for term in doc))

    # Compute statistics
    tf = compute_tf(documents)
    idf = compute_idf(documents, index_terms)
    tfidf = compute_tfidf(tf, idf, index_terms)
    doc_lengths = [len(doc) for doc in documents]

    # Process only the user's query
    pq = preprocess(query)
    q_tf = defaultdict(int)
    for term in pq:
        q_tf[term] += 1
    q_vector = {term: q_tf[term] * idf.get(term, 0.0) for term in q_tf}

    similarities = []
    for i, dv in enumerate(tfidf):
        sim = cosine_similarity(q_vector, dv)
        similarities.append((i, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    # Prepare detailed statistics
    stats = {
        'num_terms': len(index_terms),
        'doc_lengths': dict(zip(filenames, doc_lengths)),
        'tf': tf,
        'idf': idf,
        'tfidf': tfidf,
        'index_terms': index_terms,
        'filenames': filenames
    }

    return render_template('results.html',
                           query=query,
                           results=similarities,
                           stats=stats,
                           total_docs=len(documents))


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/stats')
def show_stats():
    documents, filenames = read_corpus(app.config['UPLOAD_FOLDER'])
    index_terms = sorted(set(term for doc in documents for term in doc))

    stats = {
        'num_terms': len(index_terms),
        'doc_lengths': dict(zip(filenames, [len(doc) for doc in documents])),
        'tf': compute_tf(documents),
        'idf': compute_idf(documents, index_terms),
        'tfidf': compute_tfidf(compute_tf(documents), compute_idf(documents, index_terms), index_terms),
        'index_terms': index_terms,
        'filenames': filenames
    }

    return render_template('stats.html',
                           stats=stats,
                           total_docs=len(documents))

if __name__ == '__main__':
    app.run(debug=True)