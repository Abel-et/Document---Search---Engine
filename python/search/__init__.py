from collections import defaultdict

vocabulary = defaultdict(dict)
termid = 1
term = "spacial"
docid =3
fre = 2

vocabulary[termid]['term'] =term
vocabulary[termid]['frequency'] = fre
vocabulary[termid]['documentID'] = 3
[print(key) for key in vocabulary.items()]