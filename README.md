# DiagnosisPredictor
Predicts chronic diseases using a patients previous history.

## Get the data and install dependencies
1. Get access to the [MIMIC 3 Database](https://mimic.physionet.org/gettingstarted/access/).

2. Download the [CSV files](https://mimic.physionet.org/gettingstarted/dbsetup/) to a local directory.

3. Install postgres SQL and follow [these steps](https://mimic.physionet.org/tutorials/install-mimic-locally-ubuntu/) to setup the database.

4. Install the following dependencies : 

*Note : The code was written using python 3 not guaranteed to work on python 2.*

 * tensorflow (https://www.tensorflow.org/versions/r0.10/get_started/os_setup)
 * tflearn
 * argparse>=1.2.1
 * numpy>=1.10.4
 * sklearn
 * gensim>=0.12.4
 * scipy>=0.17.0
 * nltk
 * pandas
 * matplotlib
 * seaborn
 * psycopg2
 
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

    



