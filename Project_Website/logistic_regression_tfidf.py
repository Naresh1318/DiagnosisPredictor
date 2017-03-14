"""Performs Logistic Regression to find the probability of occurrence of diseases"""

import sys
import os
import time

lib_path = os.path.abspath(os.path.join('../', 'lib'))
sys.path.append(lib_path)

import numpy as np
import pandas as pd
from icd9 import ICD9
from sklearn.externals import joblib

# Parameters
diag_to_desc = {}

t1 = time.time()


def generate_icd9_lookup():
    """Generate description from ICD9 code"""
    tree = ICD9('../lib/icd9/codes.json')

    for ud in uniq_diag:
        try:
            diag_to_desc[ud] = tree.find(ud[2:]).description
        except:
            if ud[2:] == "008":
                diag_to_desc[ud] = "Intestinal infections due to other organisms"
            elif ud[2:] == "280":
                diag_to_desc[ud] = "Iron deficiency anemias"
            elif ud[2:] == "284":
                diag_to_desc[ud] = "Aplastic anemia and other bone marrow failure syndrome"
            elif ud[2:] == "285":
                diag_to_desc[ud] = "Other and unspecified anemias"
            elif ud[2:] == "286":
                diag_to_desc[ud] = "Coagulation defects"
            elif ud[2:] == "287":
                diag_to_desc[ud] = "Purpura and other hemorrhagic conditions"
            elif ud[2:] == "288":
                diag_to_desc[ud] = "Diseases of white blood cells"
            else:
                diag_to_desc[ud] = "Not Found"


# Read the CSV file and get the inputs and outputs
df = pd.read_csv('../Data/mimic_diagnosis_tfidf/diagnosis_tfidf_5645_pat.csv', header=None)
X = df.iloc[6:, 1:1392].values
Y = {}

# Get the 80 most common diagnosis from the vocab file
with open('../Data/patient_sequences/vocab') as f:
    uniq_diag = np.array(f.read().split('\n')[1].split(' '))

# Get the diagnosis results for each patient
for d, i in zip(uniq_diag, range(101, len(uniq_diag) + 101)):
    Y[d] = df[i].values[1:]


def convert_seq_to_vec(seq):
    # Input to tfidf must be a vector so covert the string to a vector
    tfidf = joblib.load('../Data_Preparation/Transformation_Models/tfidf_fitted.pkl')
    return tfidf.transform([seq])


# Perform binary classification for each of the 80 common diagnosis
Prediction_acc = {}
Prediction_for_patient = {}
Prediction_for_patient_prob = {}  # Stores the probability of a patient's diagnosis


def patient_output(vector_rep_patient):
    # The vector representation for the patient sequence
    vector_rep_patient = convert_seq_to_vec(vector_rep_patient)
    # load the sc model
    sc = joblib.load('../Predictor_Tfidf/Saved_Models/Logistic_Regression_tfidf/standard.pkl')

    generate_icd9_lookup()  # generate the lookup for each diagnosis
    for c, d in enumerate(uniq_diag):
        lr = joblib.load('../Predictor_Tfidf/Saved_Models/Logistic_Regression_tfidf/logistic_regression_{}.pkl'
                         .format(d))
        print('Completed : {0}/{1}'.format(c + 1, len(uniq_diag)))

        # Standardize the values and predict the output
        # Input to sc
        vector_rep_patient_sc = np.reshape(sc.transform(vector_rep_patient.toarray()), (1, 1391))
        Prediction_for_patient[d] = lr.predict(vector_rep_patient_sc)
        Prediction_for_patient_prob[d] = lr.predict_proba(vector_rep_patient_sc)

    print("--------------------Training Done!!!--------------------")


# Print the predictions for the patient's input
def predict(vector_rep_patient):
    output = []
    patient_output(vector_rep_patient)

    for d in Prediction_for_patient:
        if Prediction_for_patient[d] == [1.]:
            output.append('ICD9 : {0:<8} Probability : {1:<8.2f} Description : {2}'
                          .format(d, Prediction_for_patient_prob[d][0][1], diag_to_desc[d]))
            print('ICD9 : {0:<8} Probability : {1:<8.2f} Description : {2}'
                  .format(d, Prediction_for_patient_prob[d][0][1], diag_to_desc[d]))
    print('Time Taken : {}'.format(time.time() - t1))
    return output

# res = predict('l_50931 l_51221 l_51222 l_51248 l_51249 l_51277 l_51288 l_50808 l_50809 l_50811 l_50821 l_50893 l_50902 l_50960 l_51279 l_51301 l_50818 l_50970 l_51244 l_51256 l_50824 l_51237 l_51274 l_50813 l_50820 l_50912 l_51009 l_50882 l_50924 l_50953 l_50998 l_51006 l_51003 l_50862 l_51516 c_V5867 s_7885 d_008 d_041 d_250 d_250 d_272 d_285 d_311 d_324 d_349 d_357 d_362 d_401 d_458 d_511 d_567 d_584 d_721 d_722 d_722 d_730 d_999 l_50863 l_50893 l_50902 l_50912 l_50931 l_50960 l_50970 l_51003 l_51006 l_51009 l_51221 l_51222 l_51237 l_51244 l_51256 l_51265 l_51274 l_51275 l_51277 l_51279 l_51301 l_50813 l_51516 l_50818 l_50820 l_50821 l_50862 l_50963 l_50909 l_50882 l_51482 l_51493 l_51116 l_51118 l_51120 l_51123 l_51125 l_51127 l_51128 l_50971 l_50983 l_51479 l_51249 c_V5867 s_78552 s_78959 d_008 d_038 d_250 d_250 d_272 d_276 d_280 d_357 d_362 d_401 d_427 d_428 d_428 d_511 d_518 d_571 d_584 d_730 d_995')
