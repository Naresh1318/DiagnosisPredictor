# DiagnosisPredictor
Predicts chronic diseases using a patient's previous history.

**Checkout the website: http://naresh1318.pythonanywhere.com/**

## Get the data and install dependencies
1. Get access to the [MIMIC 3 Database](https://mimic.physionet.org/gettingstarted/access/).

2. Download the [CSV files](https://mimic.physionet.org/gettingstarted/dbsetup/) to a local directory.

3. Install postgres SQL and follow [these steps](https://mimic.physionet.org/tutorials/install-mimic-locally-ubuntu/) to setup the database.

4. Install the following dependencies : 

*Note : The code was written using python 3 not guaranteed to work on python 2.*

 * tensorflow
 * tflearn
 * argparse
 * numpy
 * sklearn
 * gensim
 * scipy
 * nltk
 * pandas
 * matplotlib
 * seaborn
 * psycopg2
 * Flask
 * gensim
 
## Installing the dependencies
Install virtualenv:

    pip install virtualenv
    pip install virtualenvwrapper
    export WORKON_HOME=~/Envs
    source /usr/local/bin/virtualenvwrapper.sh
   
Create a virtual environment and install the dependencies:
    
    mkvirtualenv --python=/usr/bin/python3.5 tf
    workon tf
    pip3 install -r requirements.txt
    
*Note:*

 * *All the above steps must be execute from the DiagnosisPredictor directory.*     
 * *Install tensorflow version r.10 follow [this](https://www.tensorflow.org/versions/r0.10/get_started/os_setup#virtualenv_installation) guide.*
 * ***It is recommended that you install GPU version of tensorflow if you don't want to wait for days for all the models to be trained.***  
 * *Install tflearn after installing tensorflow. `pip3 install tflearn==0.2.1`*
## Run the predictor

### 1. Data Preparation

    cd Data_Preparation
    psql -U mimic -a -f allevents.sql
    python3 generate_icd_levels.py
    python3 generate_seq_combined.py
    python3 generate_vector_tfidf.py
    
*Note : Try `sudo psql -U mimic -a -f allevents.sql` if permission is denied.*

This generate a CSV file `Data/mimic_diagnosis_tfidf/diagnosis_tfidf_5645_pat.csv` which contains 1471 columns.
The first 1391 columns contains the tfidf representation for each patient sequence. The last 80 columns contains the 80 chronic diagnosis
we are trying to predict.

### 2. Time to run the predictors

Running the decision tree predictor:

    cd ../Predictor_Tfidf
    python3 decision_tree.py
    
The results are stored at `Results_tfidf/Random_Forest`.

ROC curve for random forest predictor looks something like this :
![random forest roc](https://raw.githubusercontent.com/Naresh1318/DiagnosisPredictor/master/Results_tfidf/Random_Forest/Plots/ROC_RF_criterion_entropy_n_estimators_10_size_1391.png)
    
Similarly other algorithms such as fully connected network(Multilayer layer perceptron)
can be run as follows : 

    python3 dense_fully_connected.py
    
The results are stored at `Results_tfidf/Dense_fully_connected`.

ROC curve for random forest predictor looks something like this :
![dense fully connected](https://raw.githubusercontent.com/Naresh1318/DiagnosisPredictor/master/Results_tfidf/Dense_fully_connected/Plots/ROC_FC_n_epoch_10_batch_size_32_size_1391_5645_tfidf.png)

## Running the website server
Loads the saved models from dense fully connected model and make predictions.
    
    cd Project_Website
    python3 app.py
    
Output:
![Website](https://raw.githubusercontent.com/Naresh1318/DiagnosisPredictor/master/Results_tfidf/diagnosispredictor.png)
## Credits
* [A Predictive Model for Medical Events Based on Contextual Embedding of Temporal Sequences](http://medinform.jmir.org/2016/4/e39/)
* [Doc2Vec](https://github.com/linanqiu/word2vec-sentiments/blob/master/word2vec-sentiment.ipynb)
