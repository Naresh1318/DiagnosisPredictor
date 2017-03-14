"""Use an ANN to find the probability of occurrence of diseases"""
import tflearn
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib
import os
import sys
import time
lib_path = os.path.abspath(os.path.join('../', 'lib'))
sys.path.append(lib_path)
from icd9 import ICD9

# Start time
t1 = time.time()

# Parameters
diag_to_desc = {}
n_epoch = 10


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

# Get the 80 most common diagnosis from the vocab file
with open('../Data/patient_sequences/vocab') as f:
    uniq_diag = np.array(f.read().split('\n')[1].split(' '))

input_patient = 'l_50931 l_51221 l_51222 l_51248 l_51249 l_51277 l_51288 l_50808 l_50809 l_50811 l_50821 l_50893 l_50902 l_50960 l_51279 l_51301 l_50818 l_50970 l_51244 l_51256 l_50824 l_51237 l_51274 l_50813 l_50820 l_50912 l_51009 l_50882 l_50924 l_50953 l_50998 l_51006 l_51003 l_50862 l_51516 c_V5867 s_7885 d_008 d_041 d_250 d_250 d_272 d_285 d_311 d_324 d_349 d_357 d_362 d_401 d_458 d_511 d_567 d_584 d_721 d_722 d_722 d_730 d_999 l_50863 l_50893 l_50902 l_50912 l_50931 l_50960 l_50970 l_51003 l_51006 l_51009 l_51221 l_51222 l_51237 l_51244 l_51256 l_51265 l_51274 l_51275 l_51277 l_51279 l_51301 l_50813 l_51516 l_50818 l_50820 l_50821 l_50862 l_50963 l_50909 l_50882 l_51482 l_51493 l_51116 l_51118 l_51120 l_51123 l_51125 l_51127 l_51128 l_50971 l_50983 l_51479 l_51249 c_V5867 s_78552 s_78959 d_008 d_038 d_250 d_250 d_272 d_276 d_280 d_357 d_362 d_401 d_427 d_428 d_428 d_511 d_518 d_571 d_584 d_730 d_995'

# Generate the vector representation for the input sequence
vect = joblib.load('../Data_Preparation/Transformation_Models/tfidf_fitted.pkl')
patient_seq = vect.transform([input_patient])  # Unstandardized value of patient sequence

# Standardizing the patient sequence
sc = joblib.load('../Data_Preparation/Transformation_Models/standard.pkl')
patient_seq = sc.transform(patient_seq.toarray())  # Convert the sequence to array or sc will give an error

Prediction_for_patient_prob = {}
Prediction_for_patient = {}

generate_icd9_lookup()  # generate the lookup for each diagnosis

for c, d in enumerate(uniq_diag):

    # Display the training diagnosis
    print("--------------------Training {}--------------------".format(d))

    # Run each iteration in a graph
    with tf.Graph().as_default():

        # Model
        input_layer = tflearn.input_data(shape=[None, 1391], name='input')
        dense1 = tflearn.fully_connected(input_layer, 128, activation='linear', name='dense1')
        dropout1 = tflearn.dropout(dense1, 0.8)
        dense2 = tflearn.fully_connected(dropout1, 128, activation='linear', name='dense2')
        dropout2 = tflearn.dropout(dense2, 0.8)
        output = tflearn.fully_connected(dropout2, 2, activation='softmax', name='output')
        regression = tflearn.regression(output, optimizer='adam', loss='categorical_crossentropy', learning_rate=.001)

        # Define model with checkpoint (autosave)
        model = tflearn.DNN(regression, tensorboard_verbose=3)

        # load the previously trained model
        model.load('Saved_Models/Fully_Connected_n_epochs_{0}/dense_fully_connected_dropout_5645_{1}.tfl'
                   .format(n_epoch, d))

        # Standardize the values and predict the output
        vector_rep_patient_sc = np.reshape(patient_seq, (1, 1391))
        # Find the probability of outputs
        Prediction_for_patient_prob[d] = np.array(model.predict(vector_rep_patient_sc))[:, 1]

        Prediction_for_patient[d] = np.where(Prediction_for_patient_prob[d] > 0.5, 1., 0.)

        print('\n')
        print('Completed : {0}/{1}'.format(c + 1, len(uniq_diag)))
        print('--------------------{} Complete--------------------'.format(d))
        print('\n')

# Print the final results
print('------------------------------Table for All Predictions------------------------------')
for d in uniq_diag:
    print('ICD9 : {0:<8s} Probability : {1:<8.2} Description : {2}'
          .format(d, float(Prediction_for_patient_prob[d][0]), diag_to_desc[d]))
print('------------------------------End------------------------------')

# Print the ICD9 codes of diseases with prob > 0.5

print('------------------------------Table for All Predictions with Prob > 0.5------------------------------')
for d in uniq_diag:
    if Prediction_for_patient[d] > 0.5:
        print('ICD9 : {0:<8s} Probability : {1:<8.2} Description : {2}'
              .format(d, Prediction_for_patient_prob[d][0], diag_to_desc[d]))
print('------------------------------End------------------------------')

# Calculate time
t2 = time.time()
print("Time Taken : {:.2f} s".format(t2 - t1))
