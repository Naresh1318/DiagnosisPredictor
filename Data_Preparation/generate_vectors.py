import pandas as pd
import numpy as np
from collections import namedtuple
from gensim.models import doc2vec
import os
import sklearn.manifold
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
size = 100  # Size of each sequence vector
window = 30  # Window for Word2Vec
min_count = 1  # min_count must be 1 for Doc2Vec
workers = 4  # Number of threads to be utilized
iterations = 10

sequence = []  # Stores the sequences of each patient
column_names = np.arange(size)  # The column names of the 100 vector sequence
diagnosis_req = {}  # Used to store the patients diagnosed with a particular disease

# Get the 80 most common diagnosis from the vocab file
with open('../Data/patient_sequences/vocab') as f:
    uniq_diag = np.array(f.read().split('\n')[1].split(' '))

# Get the sequence of each patient
with open('../Data/patient_sequences/trainv_0') as f:
    for s in f:
        sequence.append(s.split("|")[2])

# Find the actual diagnosis
for d in uniq_diag:
    diagnosis_req[d] = []  # Each entry in the dictionary is a list
    with open('../Data/patient_sequences/trainv_0') as f:
        for s in f:
            if d in s.split("|")[0]:  # append 1 if diagnosis is present in that patient's sequence
                diagnosis_req[d].append(1)
            else:
                diagnosis_req[d].append(0)

# DataFrame to store the resultant CSV file
df = pd.DataFrame(columns=np.append(column_names, uniq_diag))
docs = []  # Contains a list of tuples

# Produce a tuple containing the sequence and a tag
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(sequence):
    words = text.lower().split()
    tags = [i]
    docs.append(analyzedDocument(words, tags))

# Train the Doc2Vec model to get the vector representation for the sequences
model = doc2vec.Doc2Vec(size=size, window=window, min_count=min_count, workers=workers, iter=iterations)

# Build the vocabulary
model.build_vocab(docs)

# Train the model
model.train(docs)

# Get the vector representation for the sequences append the results for the 80 diagnosis and add a row to df
for i, text in enumerate(sequence):
    words = text.lower().split()
    pat_res = []  # Store the result of each diagnosis for a patient, gets reset for each patient
    for d in uniq_diag:
        pat_res.append(diagnosis_req[d][i])  # Get the diagnosis results for each patient

    df = df.append(pd.DataFrame([np.append(model.infer_vector(words), pat_res)],
                                columns=np.append(column_names, uniq_diag)))

# Create the Transformation_Models dir
if 'Transformation_Models' not in os.listdir():
    os.mkdir('Transformation_Models')

# Save the model
model.save('Transformation_Models/Doc2Vec_diagnosis_predictor.d2v')

# Generate a CSV file from the DataFrame
df.to_csv('../Data/mimic_diagnosis_word2vec/diagnosis_size_{0}_window_{1}_5645_pat.csv'.format(size, window))
print('Data Prepared!')
print('CSV file saved in Data/mimic_diagnosis_word2vec')

# TODO: Run this on a console
'''
model = doc2vec.Doc2Vec.load('Transformation_Models/Doc2Vec_diagnosis_predictor.d2v')
# Generate the graph for Doc2Vec
# Compress the word vectors into 2D space and plot them
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

# Generate the vector representation for all the 1391 words in the doc
all_word_vectors_matrix = model.syn0

all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

# Plot the big picture
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[model.vocab[word].index])
            for word in model.vocab
            ]
        ],
    columns=["word", "x", "y"]
)

sns.set_context("poster")


ax = points.plot.scatter("x", "y", s=35, figsize=(20, 12))
for i, point in points.iterrows():
    ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=7)
    '''
