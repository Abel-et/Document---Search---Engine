import math
import os

from spacy.matcher.dependencymatcher import defaultdict

from text_operation import TextOperation


class Indexing:
    def __init__(self):
        pass
    # Function to process multiple documents
    def process_multiple_files(self,folder_path):
        # List all files in the folder
        files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

        dictionary = {}
        for file in files:
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                document_content = f.read()

            # Perform text operation for the document
            text_operation = TextOperation(document_content)
            stemmed = text_operation.stem_word()
            stemmed.sort()
            dictionary[file] = stemmed
        return dictionary

    def file_registering_posting(self,dictionary):
        # Save posting lists to a file
        with open("postings.txt", "w") as postings_file:
            for term, postings in dictionary.items():
                postings_file.write(f"{term}: {postings}\n")

    def file_registering_vocabulary(self,dictionary):
        # Save vocabulary to a file
        with open("vocabulary.txt", "w") as vocab_file:
            for term, metadata in dictionary.items():
                vocab_file.write(
                    f"{term}\t{metadata['doc_frequency']}\t{metadata['total_frequency']}\t{metadata['posting_pointer']}\n")


index = Indexing()
i = index.process_multiple_files("D:\doc")
# print(type ,i)
key_values = list(i.keys())
corpus = []
for i,j in i.items():
    corpus.append(j)



# for term in range(len(corpus)):
#     print(term,corpus[term])

vocabulary = {}
posting_lists = defaultdict(list)

for doc_id,terms in enumerate(corpus):
    term_frequencies = defaultdict(int)

    for term in terms:
        term_frequencies[term] +=1
    for term,frequency in term_frequencies.items():
        posting_lists[term].append((key_values[doc_id] ,frequency))
# for term ,postings in posting_lists.items():
#     for doc_id,frequency in postings:
#         print(term, doc_id, frequency)
print(posting_lists)



def max_frequency(target,posting_lists):
    frequencies = [
        frequency for term, postings in posting_lists.items() for doc_id, frequency in postings if doc_id == target
    ]
    return max(frequencies) if frequencies else 0


for term,posting in posting_lists.items():
    vocabulary[term] = {
        'doc_frequency': len(posting),
        'total_frequency': sum(freq for _, freq in posting),
        'posting_pointer': posting
    }
# for term,metadata in vocabulary.items():
#     print(term,vocabulary[term]['total_frequency'])


#  this is finding the tf value of a term in a document
query = "allow"
tf_term = {}
if query in vocabulary:
    posting_term = vocabulary[query]['posting_pointer']
    for id , freq in posting_term:
        maz = max_frequency(id, posting_lists)
        print(id)
        tf = freq/maz
        if query in tf_term:
            tf_term[query] +=[(id,tf)]
        else:
            tf_term[query] = [(id,tf)]

# i want to find the max frequency of  the term in a document
print(tf_term)







# now finding the idf of a term form document
total_docs = len(key_values)

def idf(term):
    doc_freq = 0
    if term in vocabulary:
        doc_freq = vocabulary[term]['doc_frequency']
        return math.log(total_docs / doc_freq)
    else:
        return 0
idf_values = idf(query)
weight = {}
