"""Use an ANN to find the probability of occurrence of diseases"""
import tflearn
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, \
    confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tflearn.data_utils import to_categorical
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
n_epoch = 5
batch_size = 32
size = 100  # Size of each sequence vector
window = 30  # Window for Word2Vec
name = 'Load_FC_n_epoch_' + str(n_epoch) + '_batch_size_' + str(batch_size) \
       + '_size_' + str(size) + '_window_' + str(window) + '_5645_'  # name of ROC Plot


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


# Load the data
df = pd.read_csv('../Data/mimic_diagnosis_word2vec/diagnosis_size_100_window_30_5645_pat.csv', header=None)
X = df.iloc[1:, 1:101].values

# Change later
# Convert label to categorical to train with tflearn
Y = {}

# Get the 80 most common diagnosis from the vocab file
with open('../Data/patient_sequences/vocab') as f:
    uniq_diag = np.array(f.read().split('\n')[1].split(' '))

# Get the diagnosis results for each patient
for d, i in zip(uniq_diag, range(101, len(uniq_diag) + 101)):
    Y[d] = df[i].values[1:]

model = {}
# Figure for ROC
plt.figure(figsize=(17, 17), dpi=400)

for c, d in enumerate(uniq_diag[:40]):

    # Display the training diagnosis
    print("--------------------Training {}--------------------".format(d))

    # Run each iteration in a graph
    with tf.Graph().as_default():
        y = Y[d].astype(np.float32)
        y = y.reshape(-1, 1)
        y = to_categorical(y, nb_classes=2)  # Convert label to categorical to train with tflearn

        # Train and test data
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=0)

        # Standardize the data
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_sd = sc.transform(X_train)
        X_test_sd = sc.transform(X_test)

        # Model
        input_layer = tflearn.input_data(shape=[None, 100], name='input')
        dense1 = tflearn.fully_connected(input_layer, 128, activation='linear', name='dense1')
        dropout1 = tflearn.dropout(dense1, 0.8)
        dense2 = tflearn.fully_connected(dropout1, 128, activation='linear', name='dense2')
        dropout2 = tflearn.dropout(dense2, 0.8)
        output = tflearn.fully_connected(dropout2, 2, activation='softmax', name='output')
        regression = tflearn.regression(output, optimizer='adam', loss='categorical_crossentropy', learning_rate=.001)

        # Define model with checkpoint (autosave)
        model = tflearn.DNN(regression, tensorboard_verbose=3)

        # load the previously trained model
        model.load('Saved_Models/Fully_Connected/dense_fully_connected_dropout_5645_{}.tfl'.format(d))

        ''''# Train model with checkpoint every epoch and every 500 steps
        model.fit(X_train_sd, Y_train, n_epoch=n_epoch, show_metric=True, snapshot_epoch=True, snapshot_step=500,
                  run_id='model_and_weights_{}'.format(c + 1),
                  validation_set=(X_test_sd, Y_test), batch_size=batch_size)'''

        # Find the probability of outputs
        y_pred_prob = np.array(model.predict(X_test_sd))[:, 1]
        # Find the predicted class
        y_pred = np.where(y_pred_prob > 0.5, 1., 0.)
        # Predicted class is the 2nd column in Y_test
        Y_test_dia = Y_test[:, 1]

        acc = accuracy_score(Y_test_dia, y_pred) * 100
        errors = (y_pred != Y_test_dia).sum()
        ps = precision_score(Y_test_dia, y_pred) * 100
        rs = recall_score(Y_test_dia, y_pred) * 100
        f1 = f1_score(Y_test_dia, y_pred) * 100
        confmat = confusion_matrix(y_true=Y_test_dia, y_pred=y_pred)

        print("Errors for %s    : %.f" % (d, errors))
        print("Accuracy for %s  : %.2f%%" % (d, acc))
        print("Precision for %s : %.2f%%" % (d, ps))
        print("Recall for %s    : %.2f%%" % (d, rs))
        print("F1 Score for %s  : %.2f%%" % (d, f1))
        print("Confusion Matrix for %s :" % d)
        print(confmat)

        # Input to roc_curve must be Target scores, can either be
        # probability estimates of the positive class, confidence values, or non-thresholded measure of decisions
        fpr, tpr, _ = roc_curve(Y_test_dia, y_pred_prob)  # Find true positive and false positive rate
        roc_auc = auc(fpr, tpr)
        roc_area = roc_auc_score(Y_test_dia, y_pred_prob)
        print("ROC AUC for %s : %.2f" % (d, roc_area))

        generate_icd9_lookup()  # generate the lookup for each diagnosis

        if d in uniq_diag[:8]:
            # Plot the results
            colors = {uniq_diag[0]: 'red', uniq_diag[1]: 'yellow', uniq_diag[2]: 'orange', uniq_diag[3]: 'pink',
                      uniq_diag[4]: 'lightblue', uniq_diag[5]: 'green', uniq_diag[6]: 'black', uniq_diag[7]: 'brown'}

            plt.plot(fpr, tpr, lw=2, label='ROC for %s (area = %.2f)' % (diag_to_desc[d], roc_auc), color=colors[d])

        print('\n')
        print('Completed : {0}/{1}'.format(c + 1, len(uniq_diag)))
        print('--------------------{} Complete--------------------'.format(d))
        print('\n')

print('ROC Plot saved at ../Results_word2vec/load_dense_fully_connected/Plots/ROC_' + name)
print("--------------------Training Done!!!--------------------")
plt.plot([0, 1], [0, 1], lw=2, linestyle='--', color=(0.6, 0.6, 0.6), label='Random guessing (area = 0.5)')
plt.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=':', label='Prefect performance (area = 1.0)')
plt.xlim([-.05, 1.05])
plt.ylim([-.05, 1.05])
plt.xlabel('false positive rate', fontsize=25)
plt.ylabel('true positive rate', fontsize=25)
plt.title('Receiver Operator Characteristics', fontsize=25)
plt.legend(loc='lower right', fontsize=16)
plt.savefig('../Results_word2vec/load_dense_fully_connected/Plots/ROC_' + name + '.png')

# Calculate time
t2 = time.time()
print("Time Taken : {:.2f} s".format(t2 - t1))
