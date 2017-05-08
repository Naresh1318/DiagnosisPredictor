import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

# Parameters
size = 1391  # Size of each sequence vector
sequence = []  # Stores the sequences of each patient
column_names = np.arange(size)  # The column names of the 1391 vector sequence
diagnosis_req = {}  # Used to store the patients diagnosed with a particular disease
# Get the 80 most common diagnosis from the vocab file
with open('../Data/patient_sequences/vocab') as f:
    uniq_diag = np.array(f.read().split('\n')[1].split(' '))

# Get the sequence of each patient
with open('../Data/patient_sequences/trainv_0') as f:
    for s in f:
        sequence.append(s.split("|")[2])

# Generate the vector representation for the input sequence
vect = TfidfVectorizer(strip_accents=None, lowercase=False)
vect.fit(sequence)
X = vect.transform(sequence)

# Find the actual diagnosis
for d in uniq_diag:
    diagnosis_req[d] = []  # Each entry in the dictionary is a list
    with open('../Data/patient_sequences/trainv_0') as f:
        for s in f:
            if d in s.split("|")[0]:  # append 1 if diagnosis is present in that patient's sequence
                diagnosis_req[d].append(1)
            else:
                diagnosis_req[d].append(0)

df = pd.DataFrame(columns=np.append(column_names, uniq_diag))

# Get the vector representation for the sequences append the results for the 80 diagnosis and add a row to df
for i in range(5645):
    pat_res = []  # Store the result of each diagnosis for a patient, gets reset for each patient
    for d in uniq_diag:
        pat_res.append(diagnosis_req[d][i])  # Get the diagnosis results for each patient
    df = df.append(pd.DataFrame([np.append(X[i].toarray(), pat_res)], columns=np.append(column_names, uniq_diag)))
    print('{}/5645'.format(i))
df.to_csv('../Data/mimic_diagnosis_tfidf/diagnosis_tfidf_5645_pat.csv')
print('Data Prepared!')
print('CSV file saved at Data/mimic_diagnosis_tfidf')

# Save the tfidf model
# Create the Transformation_Models dir
if 'Transformation_Models' not in os.listdir():
    os.mkdir('Transformation_Models')

joblib.dump(vect, 'Transformation_Models/tfidf_fitted.pkl')
print('TFIDF Model saved at Transformation_Models/tfidf_fitted.pkl')
