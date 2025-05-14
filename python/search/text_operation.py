
import re
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# List of packages you want to ensure are downloaded
packages = ['punkt', 'stopwords']

for package in packages:
    try:
        nltk.data.find(f'tokenizers/{package}.zip')
    except LookupError:
        nltk.download(package, quiet=True)

class TextOperation:
    def __init__(self,document):
        self.document = document

    # a function that tokenize documents
    def tokenization(self):
        words_tokenize = word_tokenize(self.document)
        return words_tokenize

    # a function that eliminate the stop words
    def stopWord(self):
        stop_words = set(stopwords.words('english'))
        emoji_pattern = re.compile(
            "["  # Start of character class
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U00002702-\U000027B0"  # Dingbats
            "]+"
        )

        # Filter out stop words, punctuation, numbers, and emojis
        filtered_words = [
            word for word in self.tokenization()
            if (word.lower() not in stop_words and
                word not in string.punctuation and
                not re.match(r'^\d+$', word) and  # Exclude numbers
                not emoji_pattern.search(word)
                and  re.match(r'^[\w]+$', word))  # Exclude emojis

        ]

        return filtered_words

#     a function that stem word which are filtered before
    def stem_word(self ):
        stemmer = PorterStemmer()
        stemmed_word = [stemmer.stem(word) for word in self.stopWord() ]
        pure = list(stemmed_word)
        pure.sort()
        return pure

#     make lower
    def lower_case(self):
        lower = []
        st = ""
        for word in self.stem_word():
            for letter in word:
                lowerletter = letter.lower()
                st += lowerletter
            lower.append(st)
            st = ""
        return lower




