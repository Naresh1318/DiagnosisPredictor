import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix, roc_curve, auc, \
    roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import sys
import os
lib_path = os.path.abspath(os.path.join('../', 'lib'))
sys.path.append(lib_path)
from icd9 import ICD9

# Parameters
diag_to_desc = {}
'''penalty = 'l2'
C = 1.0
max_iter = 100
size = 1391  # Size of each sequence vector
name = 'LR_pen_' + penalty + '_C_' + str(C) + '_iter_' + \
       str(max_iter) + '_size_' + str(size)  # name of ROC Plot


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
'''

# Read the CSV file and get the inputs and outputs
df = pd.read_csv('../Data/mimic_diagnosis_tfidf/diagnosis_tfidf_5645_pat.csv', header=None)
X = df.iloc[6:, 1:1392].values
Y = {}

# Get the 80 most common diagnosis from the vocab file
with open('../Data/patient_sequences/vocab') as f:
    uniq_diag = np.array(f.read().split('\n')[1].split(' '))

# Get the diagnosis results for each patient
for d, i in zip(uniq_diag, range(1392, len(uniq_diag) + 1392)):
    Y[d] = df[i].values[1:]

# Perform binary classification for each of the 80 common diagnosis
for c, d in enumerate(uniq_diag[:8]):
    y = Y[d].astype(np.float32)
    y = y[5:]  # First 5 patients are used during training and testing
    #y = y.reshape(-1, 1)
    # Get the training and te testing vectors
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    print("--------------------Training {} --------------------".format(d))

    # Standardize the data
    #sc = StandardScaler()
    #sc.fit(X_train)

    # Save the Standardizer
    #joblib.dump(sc, 'Saved_Models/Tpot_tfidf/standard.pkl')

    #X_train_sd = sc.transform(X_train)
    #X_test_sd = sc.transform(X_test)

    tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
    tpot.fit(X_train, Y_train)

    #Y_pred = tpot.predict(X_test)

    print("CV Score : {}".format(tpot.score(X_test, Y_test)))

    # Save the model
    tpot.export('Saved_Models/Tpot_tfidf/tpot_{}.py'.format(d))

    '''errors = (Y_pred_lr != Y_test).sum()
    acc = accuracy_score(Y_pred_lr, Y_test) * 100
    ps = precision_score(Y_pred_lr, Y_test) * 100
    rs = recall_score(Y_pred_lr, Y_test) * 100
    f1 = f1_score(Y_pred_lr, Y_test) * 100
    confmat = confusion_matrix(y_true=Y_test, y_pred=Y_pred_lr)
    Prediction_acc[d] = acc * 100
    print("Errors for %s    : %.f" % (d, errors))
    print("Accuracy for %s  : %.2f%%" % (d, acc))
    print("Precision for %s : %.2f%%" % (d, ps))
    print("Recall for %s    : %.2f%%" % (d, rs))
    print("F1 Score for %s  : %.2f%%" % (d, f1))
    print("Confusion Matrix for %s :" % d)
    print(confmat)

    # Input to roc_curve must be Target scores, can either be
    # probability estimates of the positive class, confidence values, or non-thresholded measure of decisions
    Y_prob = tpot.predict_proba(X_test_sd)
    Y_prob_c = Y_prob[:, 1]  # Probability of positive class
    fpr, tpr, _ = roc_curve(Y_test, Y_prob_c)  # Find true positive and false positive rate
    roc_auc = auc(fpr, tpr)
    roc_area = roc_auc_score(Y_test, Y_prob_c)
    print("ROC AUC for %s : %.2f" % (d, roc_area))
    print('Completed : {0}/{1}'.format(c + 1, len(uniq_diag)))

    generate_icd9_lookup()  # generate the lookup for each diagnosis

    if d in uniq_diag[:8]:
        # Plot the results
        colors = {uniq_diag[0]: 'red', uniq_diag[1]: 'yellow', uniq_diag[2]: 'orange', uniq_diag[3]: 'pink',
                  uniq_diag[4]: 'lightblue', uniq_diag[5]: 'green', uniq_diag[6]: 'black', uniq_diag[7]: 'brown'}

        plt.plot(fpr, tpr, lw=2, label='ROC for %s (area = %.2f)' % (diag_to_desc[d], roc_auc), color=colors[d])

print('ROC Plot saved at ../Results_tfidf/Logistic_regression/Plots/ROC_' + name)
print("--------------------Training Done!!!--------------------")

plt.plot([0, 1], [0, 1], lw=2, linestyle='--', color=(0.6, 0.6, 0.6), label='Random guessing (area = 0.5)')
plt.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=':', label='Prefect performance (area = 1.0)')
plt.xlim([-.05, 1.05])
plt.ylim([-.05, 1.05])
plt.xlabel('false positive rate', fontsize=25)
plt.ylabel('true positive rate', fontsize=25)
plt.title('Receiver Operator Characteristics', fontsize=25)
plt.legend(loc='lower right', fontsize=16)
plt.savefig('../Results_tfidf/Logistic_regression/Plots/ROC_' + name + '.png')
'''
