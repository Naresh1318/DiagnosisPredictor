from flask import Flask, render_template, request, Markup
from flask_mail import Mail, Message
import dense_fully_connected_tfidf
# import test
import os
import sys

app = Flask(__name__)

# Parameters for Flask-mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'email'
app.config['MAIL_PASSWORD'] = 'password'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

lib_path = os.path.abspath(os.path.join('../', 'lib'))
sys.path.append(lib_path)


@app.route('/')
def index():
    return render_template('index_form.html')


@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':
        input_seq = ""
        lab_text = request.form['labtests_text']  # Used for summary
        diagnosis_text = request.form['diagnosis_text']  # Used for summary
        name = request.form['name']

        # Convert the input string numbers to the desired format
        for i in lab_text.split(","):
            input_seq = input_seq + 'l_' + i + ' '

        for i in diagnosis_text.split(","):
            input_seq = input_seq + 'd_' + i + ' '

        input_seq = input_seq.strip()

        #length = test.delay(input_seq)
        length = dense_fully_connected_tfidf.predict(input_seq)
        seq = []

        for each_seq in length:
            seq_list = []
            each_seq = each_seq.split()
            seq_list.append(each_seq[2])
            seq_list.append(each_seq[5])
            seq_list.append(' '.join(each_seq[8:]))
            disease_info = get_disease_info(each_seq[2])
            seq_list.append(disease_info)
            seq.append(seq_list)

        # Send email
        emailID = request.form['email']
        # Email id optional
        if emailID != '':
            sendMail(emailID, name, lab_text, diagnosis_text, length)

        return render_template('results_page.html', input_seq=input_seq, length=seq, name=name,
                               lab_text=lab_text, diagnosis_text=diagnosis_text)


@app.route('/feedback', methods=['POST'])
def feedback():
    if request.method == 'POST':
        name = request.form['name']
        input_seq = request.form['input_sequence']
        lab_text = request.form['lab_text_feedback']
        diagnosis_text = request.form['diagnosis_text_feedback']
        return render_template('feedback.html', name=name, input_seq=input_seq,
                               lab_text=lab_text, diagnosis_text=diagnosis_text)


@app.route('/thankyou', methods=['POST'])
def thankyou():
    if request.method == 'POST':
        input_seq = request.form['input_seq']
        actual_diagnosis_temp = request.form['diagnosis_feedback']

        actual_diagnosis = ""
        # Add the d_ for diagnosis
        for i in actual_diagnosis_temp.split(','):
            actual_diagnosis += 'd_' + i + " "

        feedback_text = input_seq + '|' + actual_diagnosis
        feedback_text = feedback_text.strip() + '\n'

        # Save the previous sequences in the file
        with open('data/feedback.txt', 'r') as fb:
            temp_text = fb.readlines()
        with open('data/feedback.txt', 'w') as fb:
            temp_text.append(feedback_text)
            for i in temp_text:
                fb.write(i)

        return render_template('thankyou.html')


@app.route('/diagnosis.html')
def diagnosis_desc():
    return render_template('diagnosis.html')


@app.route('/all_tests.html')
def all_tests():
    return render_template('all_tests.html')


def get_info_for_mail(lab_tests, diagnoses, prediction):
    diag_codes = {'422': 'Acute myocarditis', 'V1301': 'Not Found', '070': 'Viral hepatitis', 'V453': 'Not Found',
                  '502': 'Pneumoconiosis due to other silica or silicates', '921': 'Contusion of eye and adnexa',
                  'E9502': 'Not Found', '135': 'Sarcoidosis',
                  '271': 'Disorders of carbohydrate transport and metabolism',
                  '966': 'Poisoning by anticonvulsants and anti-Parkinsonism drugs', '7964': 'Not Found',
                  'V142': 'Not Found', '435': 'Transient cerebral ischemia', '482': 'Other bacterial pneumonia',
                  '769': 'Respiratory distress syndrome', 'V1589': 'Not Found', '758': 'Not Found',
                  '862': 'Injury to other and unspecified intrathoracic organs', '7840': 'Not Found',
                  'V1072': 'Not Found', 'E882': 'Fall from or out of building or other structure',
                  '123': 'Other cestode infection', '518': 'Other diseases of lung', '286': 'Coagulation defects',
                  'V5416': 'Not Found', 'V550': 'Not Found', 'V1079': 'Not Found', '79022': 'Not Found',
                  'V632': 'Not Found', 'E888': 'Other and unspecified fall', '78551': 'Not Found',
                  '303': 'Alcohol dependence syndrome',
                  '975': 'Poisoning by agents primarily acting on the smooth and skeletal muscles and respiratory system',
                  '78009': 'Not Found', '991': 'Effects of reduced temperature',
                  '459': 'Other disorders of circulatory system', 'V153': 'Not Found', 'V1351': 'Not Found',
                  '383': 'Mastoiditis and related conditions', 'E8501': 'Not Found',
                  '285': 'Other and unspecified anemias', '78652': 'Not Found',
                  '156': 'Malignant neoplasm of gallbladder and extrahepatic bile ducts', '117': 'Other mycoses',
                  'E8705': 'Not Found', 'V0253': 'Not Found', 'V1051': 'Not Found', 'E8792': 'Not Found',
                  '310': 'Specific nonpsychotic mental disorders due to brain damage',
                  '902': 'Injury to blood vessels of abdomen and pelvis', '78001': 'Not Found', '7863': 'Not Found',
                  '619': 'Fistula involving female genital tract', '78449': 'Not Found',
                  '436': 'Acute, but ill-defined, cerebrovascular disease',
                  '806': 'Fracture of vertebral column with spinal cord injury', '78052': 'Not Found',
                  '7906': 'Not Found', '78820': 'Not Found', '7837': 'Not Found',
                  '879': 'Open wound of other and unspecified sites, except limbs',
                  '800': 'Fracture of vault of skull', 'E8161': 'Not Found', '959': 'Injury, other and unspecified',
                  '78096': 'Not Found', '621': 'Disorders of uterus, not elsewhere classified',
                  '78904': 'Not Found', 'E8580': 'Not Found', 'V8812': 'Not Found', '7907': 'Not Found',
                  'E8136': 'Not Found', 'V3100': 'Not Found', '261': 'Nutritional marasmus',
                  '420': 'Acute pericarditis', '7831': 'Not Found', '79029': 'Not Found',
                  '255': 'Disorders of adrenal glands',
                  '924': 'Contusion of lower limb and of other and unspecified sites', 'V5863': 'Not Found',
                  '272': 'Disorders of lipoid metabolism', '78194': 'Not Found', 'V1081': 'Not Found',
                  '599': 'Other disorders of urethra and urinary tract', 'V4586': 'Not Found', 'V2652': 'Not Found',
                  '882': 'Open wound of hand except finger(s) alone', '596': 'Other disorders of bladder',
                  '910': 'Not Found', 'E9413': 'Not Found', '78939': 'Not Found', 'E8799': 'Not Found',
                  '874': 'Open wound of neck', '324': 'Intracranial and intraspinal abscess', 'V860': 'Not Found',
                  '141': 'Malignant neoplasm of tongue', '78039': 'Not Found', '911': 'Not Found',
                  '745': 'Not Found', 'V596': 'Not Found', 'E9300': 'Not Found',
                  '155': 'Malignant neoplasm of liver and intrahepatic bile ducts', 'V427': 'Not Found',
                  '7821': 'Not Found', 'V1006': 'Not Found', '801': 'Fracture of base of skull',
                  '7866': 'Not Found', 'E9278': 'Not Found', '369': 'Blindness and low vision',
                  '997': 'Complications affecting specified body systems, not elsewhere classified',
                  '451': 'Phlebitis and thrombophlebitis', 'V4283': 'Not Found',
                  '242': 'Thyrotoxicosis with or without goiter', '405': 'Secondary hypertension',
                  'V1242': 'Not Found', '880': 'Open wound of shoulder and upper arm',
                  '906': 'Late effects of injuries to skin and subcutaneous tissues', 'E8230': 'Not Found',
                  'V1253': 'Not Found', 'E9288': 'Not Found', '78901': 'Not Found', 'E9507': 'Not Found',
                  '192': 'Malignant neoplasm of other and unspecified parts of nervous system',
                  'V0251': 'Not Found', '7876': 'Not Found', '7891': 'Not Found', '133': 'Acariasis',
                  '707': 'Chronic ulcer of skin', 'V1084': 'Not Found', '78442': 'Not Found', '7948': 'Not Found',
                  'E8782': 'Not Found', '319': 'Unspecified mental retardation', '836': 'Dislocation of knee',
                  '78321': 'Not Found', 'E9412': 'Not Found', '425': 'Cardiomyopathy', 'V1588': 'Not Found',
                  '312': 'Disturbance of conduct, not elsewhere classified', '295': 'Schizophrenic disorders',
                  '027': 'Other zoonotic bacterial diseases', '410': 'Acute myocardial infarction',
                  'V5861': 'Not Found', 'E8849': 'Not Found', 'V1007': 'Not Found', '281': 'Not Found',
                  '570': 'Acute and subacute necrosis of liver', 'V4509': 'Not Found',
                  '886': 'Traumatic amputation of other finger(s) (complete) (partial)', 'E9588': 'Not Found',
                  '337': 'Disorders of the autonomic nervous system',
                  'E969': 'Late effects of injury purposely inflicted by other person',
                  '161': 'Malignant neoplasm of larynx', '442': 'Other aneurysm', 'V290': 'Not Found',
                  '923': 'Contusion of upper limb',
                  '506': 'Respiratory conditions due to chemical fumes and vapors', 'V3000': 'Not Found',
                  '490': 'Bronchitis, not specified as acute or chronic', 'E9422': 'Not Found',
                  '78099': 'Not Found', 'E8538': 'Not Found', '7804': 'Not Found', '244': 'Acquired hypothyroidism',
                  '743': 'Not Found', '79389': 'Not Found', '138': 'Late effects of acute poliomyelitis',
                  'E9420': 'Not Found', '227': 'Benign neoplasm of other endocrine glands and related structures',
                  '756': 'Not Found', '434': 'Occlusion of cerebral arteries', 'E8844': 'Not Found',
                  'V160': 'Not Found', '78651': 'Not Found', 'E8542': 'Not Found', '7859': 'Not Found',
                  'E9650': 'Not Found', '933': 'Foreign body in pharynx and larynx', '78094': 'Not Found',
                  'V8801': 'Not Found', '956': 'Injury to peripheral nerve(s) of pelvic girdle and lower limb',
                  '585': 'Chronic kidney disease (CKD)', '78340': 'Not Found',
                  '291': 'Alcohol-induced mental disorders', '372': 'Disorders of conjunctiva',
                  '417': 'Other diseases of pulmonary circulation', '994': 'Effects of other external causes',
                  '378': 'Strabismus and other disorders of binocular eye movements', 'V5413': 'Not Found',
                  '7808': 'Not Found', '78863': 'Not Found', 'E8790': 'Not Found', 'V641': 'Not Found',
                  'V8523': 'Not Found', '484': 'Pneumonia in infectious diseases classified elsewhere',
                  'E8842': 'Not Found', 'V8544': 'Not Found',
                  '465': 'Acute upper respiratory infections of multiple or unspecified sites',
                  '189': 'Malignant neoplasm of kidney and other and unspecified urinary organs',
                  'V5811': 'Not Found', 'E9804': 'Not Found', 'E9290': 'Not Found', 'V6441': 'Not Found',
                  '78609': 'Not Found', 'V8545': 'Not Found', '903': 'Injury to blood vessels of upper extremity',
                  '157': 'Malignant neoplasm of pancreas', 'V551': 'Not Found', 'V6284': 'Not Found',
                  '415': 'Acute pulmonary heart disease', '403': 'Hypertensive chronic kidney disease',
                  '079': 'Viral and chlamydial infection in conditions classified elsewhere and of unspecified site',
                  '980': 'Toxic effect of alcohol', 'V0382': 'Not Found', '7823': 'Not Found', 'E8548': 'Not Found',
                  'V560': 'Not Found', '240': 'Simple and unspecified goiter', 'E8888': 'Not Found',
                  'E915': 'Foreign body accidentally entering other orifice', '78659': 'Not Found',
                  '586': 'Renal failure, unspecified', 'V1201': 'Not Found',
                  '572': 'Liver abscess and sequelae of chronic liver disease', 'V5411': 'Not Found',
                  '110': 'Dermatophytosis', '7816': 'Not Found', '038': 'Septicemia',
                  '429': 'Ill-defined descriptions and complications of heart disease', 'V8536': 'Not Found',
                  'E9384': 'Not Found', '7990': 'Not Found', 'E9393': 'Not Found', '78722': 'Not Found',
                  '907': 'Late effects of injuries to the nervous system',
                  '508': 'Respiratory conditions due to other and unspecified external agents',
                  '824': 'Fracture of ankle',
                  '200': 'Lymphosarcoma and reticulosarcoma and other specified malignant tumors of lymphatic tissue',
                  '7856': 'Not Found', '7895': 'Not Found', '581': 'Nephrotic syndrome', 'E8160': 'Not Found',
                  'V441': 'Not Found', '729': 'Other disorders of soft tissues', 'E9391': 'Not Found',
                  '872': 'Open wound of ear', '692': 'Contact dermatitis and other eczema', 'V4588': 'Not Found',
                  '049': 'Other non-arthropod-borne viral diseases of central nervous system', '78071': 'Not Found',
                  '78093': 'Not Found', '524': 'Dentofacial anomalies, including malocclusion',
                  '681': 'Cellulitis and abscess of finger and toe',
                  '900': 'Injury to blood vessels of head and neck', '357': 'Inflammatory and toxic neuropathy',
                  '530': 'Diseases of esophagus', '823': 'Fracture of tibia and fibula', 'V8741': 'Not Found',
                  'E8540': 'Not Found', '711': 'Arthropathy associated with infections', '7868': 'Not Found',
                  '78702': 'Not Found', '202': 'Other malignant neoplasms of lymphoid and histiocytic tissue',
                  '437': 'Other and ill-defined cerebrovascular disease', '774': 'Other perinatal jaundice',
                  '964': 'Poisoning by agents primarily affecting blood constituents', 'E9426': 'Not Found',
                  '884': 'Multiple and unspecified open wound of upper limb', 'V5419': 'Not Found',
                  'V430': 'Not Found', '870': 'Open wound of ocular adnexa',
                  '150': 'Malignant neoplasm of esophagus', '452': 'Portal vein thrombosis', 'V581': 'Not Found',
                  '269': 'Other nutritional deficiencies', '208': 'Leukemia of unspecified cell type',
                  '353': 'Nerve root and plexus disorders', '7836': 'Not Found', '373': 'Inflammation of eyelids',
                  'V113': 'Not Found', '908': 'Late effects of other and unspecified injuries',
                  '78199': 'Not Found', 'V4572': 'Not Found', 'E9550': 'Not Found',
                  '777': 'Perinatal disorders of digestive system', '702': 'Other dermatoses',
                  '990': 'Effects of radiation, unspecified', '241': 'Nontoxic nodular goiter',
                  'E9194': 'Not Found', 'V1053': 'Not Found', 'E9330': 'Not Found',
                  '892': 'Open wound of foot except toe(s) alone', '574': 'Cholelithiasis', 'V090': 'Not Found',
                  '735': 'Acquired deformities of toe', 'E9309': 'Not Found',
                  '713': 'Arthropathy associated with other disorders classified elsewhere', '747': 'Not Found',
                  '320': 'Bacterial meningitis', 'V1087': 'Not Found',
                  '845': 'Sprains and strains of ankle and foot', '191': 'Malignant neoplasm of brain',
                  'V556': 'Not Found', '341': 'Other demyelinating diseases of central nervous system',
                  '158': 'Malignant neoplasm of retroperitoneum and peritoneum',
                  '890': 'Open wound of hip and thigh', '438': 'Late effects of cerebrovascular disease',
                  '397': 'Diseases of other endocardial structures', '958': 'Certain early complications of trauma',
                  'V148': 'Not Found', '342': 'Hemiplegia and hemiparesis', 'V5332': 'Not Found',
                  '290': 'Dementias', '171': 'Malignant neoplasm of connective and other soft tissue',
                  '211': 'Benign neoplasm of other parts of digestive system', '78607': 'Not Found',
                  'V443': 'Not Found', '294': 'Persistent mental disorders due to conditions classified elsewhere',
                  'V140': 'Not Found', '470': 'Deviated nasal septum', '7818': 'Not Found', '78703': 'Not Found',
                  '7960': 'Not Found', '7801': 'Not Found', '912': 'Not Found',
                  '674': 'Other and unspecified complications of the puerperium, not elsewhere classified',
                  '519': 'Other diseases of respiratory system', 'V1541': 'Not Found', 'V1389': 'Not Found',
                  '394': 'Diseases of mitral valve', '576': 'Other disorders of biliary tract', '501': 'Asbestosis',
                  'V1049': 'Not Found', 'E9654': 'Not Found', 'E8788': 'Not Found',
                  'E911': 'Inhalation and ingestion of food causing obstruction of respiratory tract or suffocation',
                  'E8785': 'Not Found', 'E9688': 'Not Found', 'E0010': 'Not Found', '534': 'Gastrojejunal ulcer',
                  '256': 'Ovarian dysfunction', '322': 'Meningitis of unspecified cause',
                  '351': 'Facial nerve disorders', '926': 'Crushing injury of trunk',
                  '515': 'Postinflammatory pulmonary fibrosis',
                  '934': 'Foreign body in trachea, bronchus, and lung', 'V4561': 'Not Found', '365': 'Glaucoma',
                  '404': 'Hypertensive heart and chronic kidney disease', 'E9421': 'Not Found',
                  '205': 'Myeloid leukemia', 'E887': 'Fracture, cause unspecified',
                  '088': 'Other arthropod-borne diseases', 'V1642': 'Not Found',
                  '731': 'Osteitis deformans and osteopathies associated with other disorders classified elsewhere',
                  '802': 'Fracture of face bones', 'V8541': 'Not Found', '78057': 'Not Found',
                  '617': 'Endometriosis', '78842': 'Not Found', '494': 'Bronchiectasis', 'E9323': 'Not Found',
                  '188': 'Malignant neoplasm of bladder', '447': 'Other disorders of arteries and arterioles',
                  'V1002': 'Not Found', '920': 'Contusion of face, scalp, and neck except eye(s)',
                  'E8550': 'Not Found',
                  '237': 'Neoplasm of uncertain behavior of endocrine glands and nervous system',
                  'V444': 'Not Found', '571': 'Chronic liver disease and cirrhosis', '78606': 'Not Found',
                  '615': 'Inflammatory diseases of uterus, except cervix', 'V1271': 'Not Found',
                  '7921': 'Not Found', '7962': 'Not Found',
                  '199': 'Malignant neoplasm without specification of site', 'E9068': 'Not Found',
                  'E8789': 'Not Found', '78650': 'Not Found', '746': 'Not Found', '7852': 'Not Found',
                  '375': 'Disorders of lacrimal system', '755': 'Not Found',
                  '223': 'Benign neoplasm of kidney and other urinary organs', '79415': 'Not Found',
                  'V1011': 'Not Found', 'E8497': 'Not Found', '967': 'Poisoning by sedatives and hypnotics',
                  '270': 'Disorders of amino-acid transport and metabolism', 'E8846': 'Not Found',
                  'E8780': 'Not Found', 'E9424': 'Not Found', '423': 'Other diseases of pericardium',
                  '265': 'Thiamine and niacin deficiency states',
                  '760': 'Fetus or newborn affected by maternal conditions which may be unrelated to present pregnancy',
                  '832': 'Dislocation of elbow',
                  '955': 'Injury to peripheral nerve(s) of shoulder girdle and upper limb',
                  '804': 'Multiple fractures involving skull or face with other bones', '550': 'Inguinal hernia',
                  '041': 'Bacterial infection in conditions classified elsewhere and of unspecified site',
                  'V4589': 'Not Found', 'E9248': 'Not Found', '424': 'Other diseases of endocardium',
                  '883': 'Open wound of finger(s)',
                  '558': 'Other and unspecified noninfectious gastroenteritis and colitis', '78050': 'Not Found',
                  '722': 'Intervertebral disc disorders', '764': 'Slow fetal growth and fetal malnutrition',
                  'E9283': 'Not Found', '370': 'Keratitis', '112': 'Candidiasis',
                  '904': 'Injury to blood vessels of lower extremity and unspecified sites', 'V161': 'Not Found',
                  '541': 'Appendicitis, unqualified', '039': 'Actinomycotic infections',
                  '592': 'Calculus of kidney and ureter', '257': 'Testicular dysfunction',
                  '349': 'Other and unspecified disorders of the nervous system', 'E8162': 'Not Found',
                  '7961': 'Not Found', '666': 'Postpartum hemorrhage', 'V1584': 'Not Found', '78951': 'Not Found',
                  'V5831': 'Not Found', '127': 'Other intestinal helminthiases', 'V642': 'Not Found',
                  'V694': 'Not Found', '812': 'Fracture of humerus', '844': 'Sprains and strains of knee and leg',
                  '446': 'Polyarteritis nodosa and allied conditions', '153': 'Malignant neoplasm of colon',
                  '871': 'Open wound of eyeball', '605': 'Redundant prepuce and phimosis', '78821': 'Not Found',
                  '7810': 'Not Found', '380': 'Disorders of external ear', '209': 'NEUROENDOCRINE TUMORS ',
                  'V171': 'Not Found', '477': 'Allergic rhinitis', 'E8796': 'Not Found',
                  '183': 'Malignant neoplasm of ovary and other uterine adnexa',
                  '182': 'Malignant neoplasm of body of uterus', 'V1021': 'Not Found',
                  '315': 'Specific delays in development', '359': 'Muscular dystrophies and other myopathies',
                  'E9379': 'Not Found', '543': 'Other diseases of appendix', 'V5339': 'Not Found',
                  '398': 'Other rheumatic heart disease', '346': 'Migraine', '644': 'Early or threatened labor',
                  '512': 'Pneumothorax', 'E8794': 'Not Found', '521': 'Diseases of hard tissues of teeth',
                  '263': 'Other and unspecified protein-calorie malnutrition',
                  '608': 'Other disorders of male genital organs', 'V170': 'Not Found', '7944': 'Not Found',
                  '7919': 'Not Found', 'V1279': 'Not Found', 'V442': 'Not Found', '274': 'Gout',
                  'E8783': 'Not Found', 'V4459': 'Not Found', 'E9457': 'Not Found', '78033': 'Not Found',
                  '277': 'Other and unspecified disorders of metabolism', '600': 'Hyperplasia of prostate',
                  '595': 'Cystitis', '526': 'Diseases of the jaws', '388': 'Other disorders of ear',
                  '414': 'Other forms of chronic ischemic heart disease', '603': 'Hydrocele',
                  '922': 'Contusion of trunk', '623': 'Noninflammatory disorders of vagina', 'V1259': 'Not Found',
                  '78831': 'Not Found', 'V0481': 'Not Found',
                  'E956': 'Suicide and self-inflicted injury by cutting and piercing instrument',
                  '172': 'Malignant melanoma of skin', '78838': 'Not Found',
                  'E918': 'Caught accidentally in or between objects', '301': 'Personality disorders',
                  '860': 'Traumatic pneumothorax and hemothorax', 'E9501': 'Not Found', 'E9358': 'Not Found',
                  '7842': 'Not Found', '347': 'Cataplexy and narcolepsy',
                  '999': 'Complications of medical care, not elsewhere classified', '340': 'Multiple sclerosis',
                  '701': 'Other hypertrophic and atrophic conditions of skin', '742': 'Not Found',
                  '620': 'Noninflammatory disorders of ovary, fallopian tube, and broad ligament',
                  '426': 'Conduction disorders', '723': 'Other disorders of cervical region',
                  '690': 'Erythematosquamous dermatosis', '728': 'Disorders of muscle, ligament, and fascia',
                  'E9319': 'Not Found', '78639': 'Not Found', 'V4975': 'Not Found', '7937': 'Not Found',
                  '715': 'Osteoarthrosis and allied disorders', '79902': 'Not Found', 'E9108': 'Not Found',
                  '382': 'Suppurative and unspecified otitis media', '936': 'Foreign body in intestine and colon',
                  'E9348': 'Not Found', '709': 'Other disorders of skin and subcutaneous tissue',
                  '7847': 'Not Found', '766': 'Disorders relating to long gestation and high birthweight',
                  'E9198': 'Not Found', '748': 'Not Found',
                  '356': 'Hereditary and idiopathic peripheral neuropathy',
                  '998': 'Other complications of procedures, NEC', 'E9306': 'Not Found', 'E9398': 'Not Found',
                  '834': 'Dislocation of finger',
                  '626': 'Disorders of menstruation and other abnormal bleeding from female genital tract',
                  '78899': 'Not Found', '694': 'Bullous dermatoses', '443': 'Other peripheral vascular disease',
                  '78930': 'Not Found', 'V202': 'Not Found', '053': 'Herpes zoster', 'V850': 'Not Found',
                  '362': 'Other retinal disorders', '7935': 'Not Found',
                  '719': 'Other and unspecified disorders of joint', 'V0482': 'Not Found', 'V4511': 'Not Found',
                  'V1505': 'Not Found', 'V553': 'Not Found', 'E8798': 'Not Found', 'V298': 'Not Found',
                  '170': 'Malignant neoplasm of bone and articular cartilage', 'V151': 'Not Found',
                  'V4364': 'Not Found', 'V5862': 'Not Found', 'V4973': 'Not Found', 'E8859': 'Not Found',
                  'E8528': 'Not Found', 'V4611': 'Not Found', '416': 'Chronic pulmonary heart disease',
                  'V4361': 'Not Found', '78760': 'Not Found', '493': 'Asthma',
                  '768': 'Intrauterine hypoxia and birth asphyxia', '7955': 'Not Found', 'V1255': 'Not Found',
                  'E8529': 'Not Found', '807': 'Fracture of rib(s), sternum, larynx, and trachea',
                  'V5867': 'Not Found', '273': 'Disorders of plasma protein metabolism',
                  '486': 'Pneumonia, organism unspecified', '820': 'Fracture of neck of femur',
                  'V1504': 'Not Found', '7910': 'Not Found', '79439': 'Not Found', 'E9359': 'Not Found',
                  '402': 'Hypertensive heart disease', '302': 'Sexual and gender identity disorders',
                  '432': 'Other and unspecified intracranial hemorrhage', '250': 'Diabetes mellitus',
                  'E8502': 'Not Found', 'E8830': 'Not Found', 'V4962': 'Not Found',
                  '327': 'ORGANIC SLEEP DISORDERS ', '759': 'Not Found', 'E8889': 'Not Found', 'E8881': 'Not Found',
                  '314': 'Hyperkinetic syndrome of childhood', '78829': 'Not Found', 'V3001': 'Not Found',
                  'E8706': 'Not Found', 'E9299': 'Not Found', '364': 'Disorders of iris and ciliary body',
                  '557': 'Vascular insufficiency of intestine', 'E9344': 'Not Found', 'E9682': 'Not Found',
                  'V554': 'Not Found', 'V292': 'Not Found', 'V0991': 'Not Found',
                  '483': 'Pneumonia due to other specified organism', '865': 'Injury to spleen',
                  '296': 'Episodic mood disorders', '284': 'Aplastic anemia and other bone marrow failure syndrome',
                  '627': 'Menopausal and postmenopausal disorders', '040': 'Other bacterial diseases',
                  '972': 'Poisoning by agents primarily affecting the cardiovascular system',
                  '821': 'Fracture of other and unspecified parts of femur', 'V1552': 'Not Found',
                  '7813': 'Not Found', 'E9444': 'Not Found', 'V8542': 'Not Found', 'V643': 'Not Found',
                  'E9293': 'Not Found', 'E9353': 'Not Found', 'V5417': 'Not Found', 'V4987': 'Not Found',
                  'V1042': 'Not Found', '507': 'Pneumonitis due to solids and liquids', 'E8132': 'Not Found',
                  'E8494': 'Not Found', '776': 'Hematological disorders of newborn',
                  '582': 'Chronic glomerulonephritis', 'E9289': 'Not Found', 'V451': 'Not Found',
                  'E9504': 'Not Found', '7949': 'Not Found', '180': 'Malignant neoplasm of cervix uteri',
                  '830': 'Dislocation of jaw', '564': 'Functional digestive disorders, not elsewhere classified',
                  '7892': 'Not Found', 'V1062': 'Not Found', 'V4571': 'Not Found',
                  '323': 'Encephalitis, myelitis, and encephalomyelitis', '7873': 'Not Found',
                  '331': 'Other cerebral degenerations', '962': 'Poisoning by hormones and synthetic substitutes',
                  '708': 'Urticaria', '7963': 'Not Found', '78903': 'Not Found', '556': 'Ulcerative colitis',
                  'V4282': 'Not Found', 'V5412': 'Not Found', 'E9800': 'Not Found', 'V1202': 'Not Found',
                  '358': 'Myoneural disorders', 'V061': 'Not Found', 'V431': 'Not Found', '7861': 'Not Found',
                  'E9317': 'Not Found', 'V4976': 'Not Found',
                  '977': 'Poisoning by other and unspecified drugs and medicinal substances', 'V0254': 'Not Found',
                  '721': 'Spondylosis and allied disorders', '863': 'Injury to gastrointestinal tract',
                  '840': 'Sprains and strains of shoulder and upper arm',
                  '968': 'Poisoning by other central nervous system depressants and anesthetics',
                  '345': 'Epilepsy and recurrent seizures', 'E8496': 'Not Found', 'V1582': 'Not Found',
                  'V1241': 'Not Found', 'E8809': 'Not Found', 'V721': 'Not Found', 'V4984': 'Not Found',
                  '7843': 'Not Found', 'V1649': 'Not Found', '567': 'Peritonitis and retroperitoneal infections',
                  '7945': 'Not Found', '034': 'Streptococcal sore throat and scarlet fever',
                  'E912': 'Inhalation and ingestion of other object causing obstruction of respiratory tract or suffocation',
                  'E8791': 'Not Found', '725': 'Polymyalgia rheumatica', '78441': 'Not Found',
                  '527': 'Diseases of the salivary glands',
                  '616': 'Inflammatory disease of cervix, vagina, and vulva', '282': 'Not Found',
                  '348': 'Other conditions of brain', 'V434': 'Not Found',
                  '197': 'Secondary malignant neoplasm of respiratory and digestive systems',
                  '174': 'Malignant neoplasm of female breast', 'V1000': 'Not Found', 'E8122': 'Not Found',
                  '305': 'Nondependent abuse of drugs', 'E8843': 'Not Found',
                  '225': 'Benign neoplasm of brain and other parts of nervous system', 'V707': 'Not Found',
                  '262': 'Other severe protein-calorie malnutrition', '317': 'Mild mental retardation',
                  '7902': 'Not Found', '462': 'Acute pharyngitis', 'E8810': 'Not Found', '052': 'Chickenpox',
                  '78630': 'Not Found', '578': 'Gastrointestinal hemorrhage', '246': 'Other disorders of thyroid',
                  'V452': 'Not Found', 'E8508': 'Not Found',
                  '730': 'Osteomyelitis, periostitis, and other infections involving bone',
                  '720': 'Ankylosing spondylitis and other inflammatory spondylopathies',
                  '421': 'Acute and subacute endocarditis', '813': 'Fracture of radius and ulna',
                  '292': 'Drug-induced mental disorders', '706': 'Diseases of sebaceous glands',
                  '054': 'Herpes simplex', '396': 'Diseases of mitral and aortic valves', 'E8500': 'Not Found',
                  '763': 'Fetus or newborn affected by other complications of labor and delivery',
                  '736': 'Other acquired deformities of limbs', '485': 'Bronchopneumonia, organism unspecified',
                  'V1251': 'Not Found', 'V4986': 'Not Found', '866': 'Injury to kidney', '7824': 'Not Found',
                  '919': 'Not Found', '455': 'Hemorrhoids', 'E8190': 'Not Found', 'V1001': 'Not Found',
                  '7885': 'Not Found', 'V4502': 'Not Found', 'E9401': 'Not Found',
                  '852': 'Subarachnoid, subdural, and extradural hemorrhage, following injury',
                  '517': 'Lung involvement in conditions classified elsewhere', 'V1009': 'Not Found',
                  '591': 'Hydronephrosis', 'V1003': 'Not Found', '147': 'Malignant neoplasm of nasopharynx',
                  '935': 'Foreign body in mouth, esophagus, and stomach', 'V1047': 'Not Found', 'V163': 'Not Found',
                  '78559': 'Not Found',
                  '553': 'Other hernia of abdominal cavity without mention of obstruction or gangrene',
                  '491': 'Chronic bronchitis', '198': 'Secondary malignant neoplasm of other specified sites',
                  'V8539': 'Not Found', '496': 'Chronic airway obstruction, not elsewhere classified',
                  '78905': 'Not Found', 'V8543': 'Not Found', '7991': 'Not Found', 'E8700': 'Not Found',
                  '413': 'Angina pectoris', 'V0981': 'Not Found', '440': 'Atherosclerosis', 'E9174': 'Not Found',
                  '173': 'Other malignant neoplasm of skin', '78723': 'Not Found',
                  '839': 'Other, multiple, and ill-defined dislocations', '79311': 'Not Found',
                  '614': 'Inflammatory disease of ovary, fallopian tube, pelvic cellular tissue, and peritoneum',
                  '194': 'Malignant neoplasm of other endocrine glands and related structures', 'V183': 'Not Found',
                  '7993': 'Not Found', '811': 'Fracture of scapula', '523': 'Gingival and periodontal diseases',
                  '204': 'Lymphoid leukemia', '78059': 'Not Found', 'V8531': 'Not Found', 'V6549': 'Not Found',
                  '691': 'Atopic dermatitis and related conditions', 'V0262': 'Not Found', '458': 'Hypotension',
                  '78062': 'Not Found', 'E9378': 'Not Found', 'E8199': 'Not Found', '737': 'Curvature of spine',
                  '528': 'Diseases of the oral soft tissues, excluding lesions specific for gingiva and tongue',
                  '727': 'Other disorders of synovium, tendon, and bursa', '580': 'Acute glomerulonephritis',
                  '7802': 'Not Found', 'V433': 'Not Found', '695': 'Erythematous conditions', 'E8499': 'Not Found',
                  '607': 'Disorders of penis',
                  '778': 'Conditions involving the integument and temperature regulation of fetus and newborn',
                  'V166': 'Not Found', '808': 'Fracture of pelvis', 'V626': 'Not Found', 'V1091': 'Not Found',
                  'E8261': 'Not Found', 'E0009': 'Not Found', 'V180': 'Not Found', 'V4579': 'Not Found',
                  '79099': 'Not Found', '803': 'Other and unqualified skull fractures',
                  '520': 'Disorders of tooth development and eruption', '684': 'Impetigo', '78839': 'Not Found',
                  'V1089': 'Not Found', 'E9307': 'Not Found', 'E9385': 'Not Found', 'V0179': 'Not Found',
                  'V1052': 'Not Found', 'V1529': 'Not Found',
                  '433': 'Occlusion and stenosis of precerebral arteries', '235': 'Not Found',
                  '472': 'Chronic pharyngitis and nasopharyngitis', '395': 'Diseases of aortic valve',
                  'V462': 'Not Found', 'V8533': 'Not Found', '308': 'Acute reaction to stress', 'V066': 'Not Found',
                  'V1254': 'Not Found', '78791': 'Not Found', 'V011': 'Not Found', 'E9301': 'Not Found',
                  'V422': 'Not Found', 'E9310': 'Not Found',
                  'E927': 'Overexertion and strenuous and repetitive movements or loads',
                  '249': 'Secondary diabetes mellitus', 'E9298': 'Not Found',
                  '478': 'Other diseases of upper respiratory tract', 'E8600': 'Not Found',
                  '376': 'Disorders of the orbit', 'V1082': 'Not Found', 'E8130': 'Not Found',
                  '963': 'Poisoning by primarily systemic agents', '579': 'Intestinal malabsorption',
                  '78830': 'Not Found', 'V1272': 'Not Found', '601': 'Inflammatory diseases of prostate',
                  'V1046': 'Not Found', 'E8786': 'Not Found', '078': 'Other diseases due to viruses and Chlamydiae',
                  '867': 'Injury to pelvic organs', '814': 'Fracture of carpal bone(s)', 'E8150': 'Not Found',
                  '577': 'Diseases of pancreas', '78701': 'Not Found',
                  '868': 'Injury to other intra-abdominal organs', 'V625': 'Not Found',
                  '810': 'Fracture of clavicle', 'V8535': 'Not Found', '861': 'Injury to heart and lung',
                  '79989': 'Not Found', 'V1581': 'Not Found', 'E8781': 'Not Found', 'E9479': 'Not Found',
                  '7862': 'Not Found', '598': 'Urethral stricture', 'V3101': 'Not Found', '78065': 'Not Found',
                  '79579': 'Not Found', '535': 'Gastritis and duodenitis',
                  '775': 'Endocrine and metabolic disturbances specific to the fetus and newborn',
                  '007': 'Other protozoal intestinal diseases', '78909': 'Not Found', 'V5331': 'Not Found',
                  'E9305': 'Not Found', '411': 'Other acute and subacute forms of ischemic heart disease',
                  '243': 'Congenital hypothyroidism', '130': 'Toxoplasmosis',
                  '770': 'Other respiratory conditions of fetus and newborn',
                  '696': 'Psoriasis and similar disorders', 'V103': 'Not Found',
                  '238': 'Neoplasm of uncertain behavior of other and unspecified sites and tissues',
                  'V440': 'Not Found', '78603': 'Not Found', '78843': 'Not Found', '309': 'Adjustment reaction',
                  'V420': 'Not Found', '826': 'Fracture of one or more phalanges of foot',
                  '253': 'Disorders of the pituitary gland and its hypothalamic control',
                  '162': 'Malignant neoplasm of trachea, bronchus, and lung', 'V5869': 'Not Found',
                  '773': 'Hemolytic disease of fetus or newborn, due to isoimmunization',
                  '965': 'Poisoning by analgesics, antipyretics, and antirheumatics', '78060': 'Not Found',
                  '78604': 'Not Found', 'V653': 'Not Found', 'V620': 'Not Found', 'E9503': 'Not Found',
                  '575': 'Other disorders of gallbladder', 'V1261': 'Not Found',
                  '537': 'Other disorders of stomach and duodenum', '531': 'Gastric ulcer', 'V2651': 'Not Found',
                  '011': 'Pulmonary tuberculosis', '042': 'Human immunodeficiency virus [HIV] disease',
                  '750': 'Not Found', '336': 'Other diseases of spinal cord', 'E8784': 'Not Found',
                  '816': 'Fracture of one or more phalanges of hand',
                  '326': 'Late effects of intracranial abscess or pyogenic infection',
                  '772': 'Fetal and neonatal hemorrhage', 'E8498': 'Not Found', '7851': 'Not Found',
                  '203': 'Multiple myeloma and immunoproliferative neoplasms', '918': 'Not Found',
                  '226': 'Benign neoplasm of thyroid glands', '492': 'Emphysema',
                  '970': 'Poisoning by central nervous system stimulants', 'V8538': 'Not Found',
                  '78729': 'Not Found', '969': 'Poisoning by psychotropic agents', 'V293': 'Not Found',
                  'E9173': 'Not Found', 'E8880': 'Not Found', '835': 'Dislocation of hip', 'V3401': 'Not Found',
                  '385': 'Other disorders of middle ear and mastoid', '741': 'Not Found',
                  '228': 'Hemangioma and lymphangioma, any site', '350': 'Trigeminal nerve disorders',
                  '587': 'Renal sclerosis, unspecified', '148': 'Malignant neoplasm of hypopharynx',
                  'E8704': 'Not Found', '952': 'Spinal cord injury without evidence of spinal bone injury',
                  '584': 'Acute renal failure', 'E9443': 'Not Found', '361': 'Retinal detachments and defects',
                  'V146': 'Not Found', 'E9670': 'Not Found', '7850': 'Not Found',
                  '287': 'Purpura and other hemorrhagic conditions', '78061': 'Not Found', '754': 'Not Found',
                  'V5883': 'Not Found', 'E9411': 'Not Found', '705': 'Disorders of sweat glands',
                  '7854': 'Not Found', '338': 'PAIN ', '604': 'Orchitis and epididymitis',
                  '354': 'Mononeuritis of upper limb and mononeuritis multiplex', '79093': 'Not Found',
                  'E9600': 'Not Found', 'E9430': 'Not Found',
                  '588': 'Disorders resulting from impaired renal function', 'E9010': 'Not Found',
                  'V4971': 'Not Found', 'E9351': 'Not Found', '536': 'Disorders of function of stomach',
                  'E9352': 'Not Found', '881': 'Open wound of elbow, forearm, and wrist', 'V5865': 'Not Found',
                  '453': 'Other venous embolism and thrombosis',
                  '765': 'Disorders relating to short gestation and low birthweight', 'V1209': 'Not Found',
                  '431': 'Intracerebral hemorrhage', '464': 'Acute laryngitis and tracheitis', 'V4585': 'Not Found',
                  '008': 'Intestinal infections due to other organisms', 'E8210': 'Not Found', '7830': 'Not Found',
                  '245': 'Thyroiditis', 'E8191': 'Not Found', 'V652': 'Not Found', 'V4575': 'Not Found',
                  'V1043': 'Not Found', 'V173': 'Not Found', 'V169': 'Not Found',
                  '047': 'Meningitis due to enterovirus', 'V4512': 'Not Found', '78469': 'Not Found',
                  '236': 'Neoplasm of uncertain behavior of genitourinary organs', 'E9354': 'Not Found',
                  '457': 'Noninfectious disorders of lymphatic channels', 'V017': 'Not Found',
                  '154': 'Malignant neoplasm of rectum, rectosigmoid junction, and anus', 'V071': 'Not Found',
                  '031': 'Diseases due to other mycobacteria', '927': 'Crushing injury of upper limb',
                  '78451': 'Not Found', 'E9315': 'Not Found', 'V8530': 'Not Found', 'E9291': 'Not Found',
                  '738': 'Other acquired deformity', 'E8551': 'Not Found', '428': 'Heart failure',
                  '590': 'Infections of kidney', 'E9356': 'Not Found', '685': 'Pilonidal cyst', '7904': 'Not Found',
                  'V5489': 'Not Found', '78097': 'Not Found', 'E9201': 'Not Found', 'V5812': 'Not Found',
                  'E8556': 'Not Found', 'E8583': 'Not Found', 'E9452': 'Not Found',
                  '583': 'Nephritis and nephropathy, not specified as acute or chronic', 'V1203': 'Not Found',
                  '115': 'Histoplasmosis', '562': 'Diverticula of intestine',
                  '299': 'Pervasive developmental disorders',
                  '488': 'Influenza due to identified avian influenza virus', '445': 'Atheroembolism',
                  'V1741': 'Not Found', '905': 'Late effects of musculoskeletal and connective tissue injuries',
                  '704': 'Diseases of hair and hair follicles',
                  '854': 'Intracranial injury of other and unspecified nature', '7833': 'Not Found',
                  '318': 'Other specified mental retardation', 'V4365': 'Not Found', 'E9447': 'Not Found',
                  'V8534': 'Not Found', '78841': 'Not Found', 'E8768': 'Not Found', '218': 'Uterine leiomyoma',
                  '749': 'Not Found', 'E8495': 'Not Found', 'V8522': 'Not Found',
                  '533': 'Peptic ulcer, site unspecified', '449': 'Septic arterial embolism', '7881': 'Not Found',
                  '195': 'Malignant neoplasm of other and ill-defined sites',
                  '771': 'Infections specific to the perinatal period',
                  '714': 'Rheumatoid arthritis and other inflammatory polyarthropathies',
                  '444': 'Arterial embolism and thrombosis', '360': 'Disorders of the globe', 'E8543': 'Not Found',
                  'V425': 'Not Found', 'E9478': 'Not Found', '79319': 'Not Found', 'V1509': 'Not Found',
                  '532': 'Duodenal ulcer', '268': 'Vitamin D deficiency', '7812': 'Not Found',
                  '995': 'Certain adverse effects not elsewhere classified', '752': 'Not Found',
                  '682': 'Other cellulitis and abscess', 'E8181': 'Not Found', 'V058': 'Not Found',
                  '555': 'Regional enteritis', 'E9803': 'Not Found', 'V029': 'Not Found', '7827': 'Not Found',
                  '779': 'Other and ill-defined conditions originating in the perinatal period',
                  '648': 'Other current conditions in the mother classifiable elsewhere, but complicating pregnancy, childbirth, or the puerperium',
                  '594': 'Calculus of lower urinary tract', 'V4501': 'Not Found', '873': 'Other open wound of head',
                  'V4972': 'Not Found', 'V851': 'Not Found',
                  '853': 'Other and unspecified intracranial hemorrhage following injury', 'E8232': 'Not Found',
                  '374': 'Other disorders of eyelids', 'V4576': 'Not Found',
                  '193': 'Malignant neoplasm of thyroid gland', '78906': 'Not Found',
                  '401': 'Essential hypertension', 'V600': 'Not Found', '864': 'Injury to liver',
                  '332': "Parkinson's disease", '698': 'Pruritus and related conditions', '79431': 'Not Found',
                  '611': 'Other disorders of breast', '815': 'Fracture of metacarpal bone(s)', '916': 'Not Found',
                  '304': 'Drug dependence',
                  '149': 'Malignant neoplasm of other and ill-defined sites within the lip, oral cavity, and pharynx',
                  '363': 'Chorioretinal inflammations, scars, and other disorders of choroid',
                  '724': 'Other and unspecified disorders of back',
                  '693': 'Dermatitis due to substances taken internally', '79551': 'Not Found', 'V714': 'Not Found',
                  'V461': 'Not Found', '78550': 'Not Found', 'V1041': 'Not Found', '712': 'Crystal arthropathies',
                  'E9361': 'Not Found', 'E9809': 'Not Found',
                  '805': 'Fracture of vertebral column without mention of spinal cord injury',
                  '618': 'Genital prolapse', 'V502': 'Not Found', 'V5866': 'Not Found', 'V1005': 'Not Found',
                  '136': 'Other and unspecified infectious and parasitic diseases',
                  '276': 'Disorders of fluid, electrolyte, and acid-base balance',
                  '311': 'Depressive disorder, not elsewhere classified', 'V053': 'Not Found', 'V1749': 'Not Found',
                  'E9392': 'Not Found', '111': 'Dermatomycosis, other and unspecified',
                  'E989': 'Late effects of injury, undetermined whether accidentally or purposely inflicted',
                  '78907': 'Not Found', '094': 'Neurosyphilis', '78900': 'Not Found', 'E8147': 'Not Found',
                  '703': 'Diseases of nail', '560': 'Intestinal obstruction without mention of hernia',
                  '7931': 'Not Found', '757': 'Not Found', 'V1302': 'Not Found', '216': 'Benign neoplasm of skin',
                  '481': 'Pneumococcal pneumonia [Streptococcus pneumoniae pneumonia]',
                  '333': 'Other extrapyramidal disease and abnormal movement disorders', '751': 'Not Found',
                  '593': 'Other disorders of kidney and ureter', '151': 'Malignant neoplasm of stomach',
                  '525': 'Other diseases and conditions of the teeth and supporting structures',
                  'E9220': 'Not Found', 'V8524': 'Not Found', '7864': 'Not Found',
                  '646': 'Other complications of pregnancy, not elsewhere classified', '260': 'Kwashiorkor',
                  '487': 'Influenza', 'V168': 'Not Found', 'V854': 'Not Found', '850': 'Concussion',
                  'V1204': 'Not Found', '726': 'Peripheral enthesopathies and allied syndromes',
                  'E8490': 'Not Found', '78959': 'Not Found', '391': 'Rheumatic fever with heart involvement',
                  '480': 'Viral pneumonia', 'E9500': 'Not Found', '343': 'Infantile cerebral palsy',
                  '456': 'Varicose veins of other sites', '901': 'Injury to blood vessels of thorax',
                  '733': 'Other disorders of bone and cartilage',
                  '371': 'Corneal opacity and other disorders of cornea',
                  '139': 'Late effects of other infectious and parasitic diseases', 'V145': 'Not Found',
                  '251': 'Other disorders of pancreatic internal secretion', 'V1507': 'Not Found',
                  '7872': 'Not Found', 'V1551': 'Not Found', '280': 'Iron deficiency anemias', 'V555': 'Not Found',
                  '516': 'Other alveolar and parietoalveolar pneumonopathy', '510': 'Empyema',
                  '982': 'Toxic effect of solvents other than petroleum based',
                  '293': 'Transient mental disorders due to conditions classified elsewhere', 'V1061': 'Not Found',
                  'V1088': 'Not Found', '7845': 'Not Found', '78552': 'Not Found', 'E9429': 'Not Found',
                  '075': 'Infectious mononucleosis', '366': 'Cataract', '427': 'Cardiac dysrhythmias',
                  'V8537': 'Not Found', 'V270': 'Not Found', '78003': 'Not Found',
                  '710': 'Diffuse diseases of connective tissue',
                  '996': 'Complications peculiar to certain specified procedures', '475': 'Peritonsillar abscess',
                  '851': 'Cerebral laceration and contusion', '185': 'Malignant neoplasm of prostate',
                  '201': "Hodgkin's disease", '565': 'Anal fissure and fistula', 'E9192': 'Not Found',
                  'E9331': 'Not Found', 'E8120': 'Not Found', '7806': 'Not Found', '297': 'Delusional disorders',
                  '7908': 'Not Found', '78605': 'Not Found', '79094': 'Not Found', 'V0259': 'Not Found',
                  '377': 'Disorders of optic nerve and visual pathways',
                  '825': 'Fracture of one or more tarsal and metatarsal bones', 'V454': 'Not Found',
                  'E9347': 'Not Found', '196': 'Secondary and unspecified malignant neoplasm of lymph nodes',
                  'E9394': 'Not Found', '278': 'Overweight, obesity and other hyperalimentation',
                  '7994': 'Not Found', '78720': 'Not Found', 'V1508': 'Not Found', 'E9363': 'Not Found',
                  'V8525': 'Not Found', 'V1083': 'Not Found', '266': 'Deficiency of B-complex components',
                  'E9320': 'Not Found', 'E9060': 'Not Found', '466': 'Acute bronchitis and bronchiolitis',
                  '552': 'Other hernia of abdominal cavity, with obstruction, but without mention of gangrene',
                  'V644': 'Not Found', '386': 'Vertiginous syndromes and other disorders of vestibular system',
                  '379': 'Other disorders of eye', '355': 'Mononeuritis of lower limb',
                  '164': 'Malignant neoplasm of thymus, heart, and mediastinum', '473': 'Chronic sinusitis',
                  'V0980': 'Not Found', '753': 'Not Found', 'V1641': 'Not Found', '78721': 'Not Found',
                  '212': 'Benign neoplasm of respiratory and intrathoracic organs',
                  '412': 'Old myocardial infarction', '7905': 'Not Found',
                  '454': 'Varicose veins of lower extremities', '79092': 'Not Found', 'V4281': 'Not Found',
                  'E9229': 'Not Found', '831': 'Dislocation of shoulder', 'E8532': 'Not Found',
                  '566': 'Abscess of anal and rectal regions',
                  '847': 'Sprains and strains of other and unspecified parts of back',
                  '948': 'Burns classified according to extent of body surface involved', 'V4578': 'Not Found',
                  '573': 'Other disorders of liver', 'V6442': 'Not Found', 'E8504': 'Not Found',
                  'E9689': 'Not Found', 'E9390': 'Not Found',
                  '909': 'Late effects of other and unspecified external causes',
                  '522': 'Diseases of pulp and periapical tissues', 'V5481': 'Not Found', 'E8192': 'Not Found',
                  'E916': 'Struck accidentally by falling object', 'V1085': 'Not Found',
                  '513': 'Abscess of lung and mediastinum', 'V552': 'Not Found', 'V4577': 'Not Found',
                  '335': 'Anterior horn cell disease', 'E8801': 'Not Found', '511': 'Pleurisy',
                  '334': 'Spinocerebellar disease',
                  '647': 'Infectious and parasitic conditions in the mother classifiable elsewhere, but complicating pregnancy, childbirth, or the puerperium',
                  'V463': 'Not Found', '252': 'Disorders of parathyroid gland', 'V446': 'Not Found',
                  'V0261': 'Not Found', 'V0381': 'Not Found', '163': 'Malignant neoplasm of pleura',
                  '568': 'Other disorders of peritoneum', '300': 'Anxiety, dissociative and somatoform disorders',
                  '529': 'Diseases and other conditions of the tongue',
                  '233': 'Carcinoma in situ of breast and genitourinary system',
                  '441': 'Aortic aneurysm and dissection', '822': 'Fracture of patella',
                  '957': 'Injury to other and unspecified nerves', 'E9313': 'Not Found',
                  '951': 'Injury to other cranial nerve(s)', '009': 'Ill-defined intestinal infections',
                  'V1004': 'Not Found', '7835': 'Not Found', 'V167': 'Not Found',
                  '875': 'Open wound of chest (wall)', '7820': 'Not Found', 'V4573': 'Not Found',
                  'V4983': 'Not Found', 'V141': 'Not Found', '275': 'Disorders of mineral metabolism',
                  '239': 'NEOPLASMS OF UNSPECIFIED NATURE ', '78459': 'Not Found', '389': 'Hearing loss',
                  '625': 'Pain and other symptoms associated with female genital organs',
                  '325': 'Phlebitis and thrombophlebitis of intracranial venous sinuses',
                  '307': 'Special symptoms or syndromes, not elsewhere classified',
                  '971': 'Poisoning by drugs primarily affecting the autonomic nervous system',
                  '461': 'Acute sinusitis', 'E9571': 'Not Found', 'E9509': 'Not Found', '283': 'Not Found',
                  'E9530': 'Not Found', 'V174': 'Not Found', 'V4582': 'Not Found',
                  '569': 'Other disorders of intestine', 'V667': 'Not Found',
                  '686': 'Other local infections of skin and subcutaneous tissue', '430': 'Subarachnoid hemorrhage',
                  'E9179': 'Not Found', '298': 'Other nonorganic psychoses',
                  '279': 'Disorders involving the immune mechanism', 'E9308': 'Not Found', 'E9208': 'Not Found',
                  '767': 'Birth trauma', '716': 'Other and unspecified arthropathies', '368': 'Visual disturbances',
                  'E8708': 'Not Found', 'V1044': 'Not Found', '214': 'Lipoma',
                  '514': 'Pulmonary congestion and hypostasis',
                  'V08': 'Asymptomatic human immunodeficiency virus [HIV] infection status', '7825': 'Not Found',
                  'E8121': 'Not Found', 'V4581': 'Not Found', 'E9346': 'Not Found', 'V5864': 'Not Found',
                  'E9342': 'Not Found', 'E9383': 'Not Found', '220': 'Benign neoplasm of ovary',
                  '152': 'Malignant neoplasm of small intestine, including duodenum',
                  '259': 'Other endocrine disorders', '540': 'Acute appendicitis', 'E8493': 'Not Found',
                  'V1252': 'Not Found', '495': 'Extrinsic allergic alveolitis', '78799': 'Not Found',
                  '913': 'Not Found', 'V4574': 'Not Found', '289': 'Not Found', '78079': 'Not Found',
                  'E966': 'Assault by cutting and piercing instrument',
                  '891': 'Open wound of knee, leg [except thigh], and ankle', '344': 'Other paralytic syndromes',
                  '288': 'Diseases of white blood cells', 'V8532': 'Not Found', '79001': 'Not Found'}
    lab_codes = {'51447': 'Macrophages', '51425': 'Kappa', '51397': 'CD16/56', '51264': 'Platelet Clumps',
                 '51275': 'PTT', '51501': 'Transitional Epithelial Cells', '51499': 'Sperm', '51189': 'CD57',
                 '50866': 'Ammonia', '51476': 'Epithelial Cells', '51194': 'CD8 Cells, Percent', '50825': 'Temperature',
                 '51123': 'Other', '51067': '24 hr Creatinine', '51119': 'Metamyelocytes', '51446': 'Lymphocytes',
                 '51217': 'Glyco A', '51274': 'PT', '51268': 'Polychromasia', '51511': 'Urine Fat Bodies',
                 '51482': 'Hyaline Casts', '51149': 'Bleeding Time', '51262': 'Pencil Cells', '51035': 'LD, Body Fluid',
                 '50898': 'Cancer Antigen 27.29', '51057': 'Potassium, Pleural', '51409': 'CD4/CD8 Ratio',
                 '51124': 'Plasma', '51111': 'Bands', '51530': 'AF-AFP', '51010': 'Vitamin B12',
                 '51068': '24 hr Protein', '51340': 'Kappa', '51347': 'Eosinophils', '50828': 'Ventilator',
                 '51143': 'Atypical Lymphocytes', '51051': 'Cholesterol, Pleural', '51551': 'VOIDED SPECIMEN',
                 '50998': 'Transferrin', '50941': 'Hepatitis B Surface Antigen', '51396': 'CD16', '51478': 'Glucose',
                 '51097': 'Potassium, Urine', '50800': 'SPECIMEN TYPE', '51332': 'CD7',
                 '50938': 'Hepatitis A Virus IgM Antibody', '51169': 'CD20 %', '51376': 'Macrophage',
                 '50919': 'EDTA Hold', '51180': 'CD4 Cells, Percent', '51216': 'Fragmented Cells',
                 '50856': 'Acetaminophen', '51372': 'Joint Crystals, Location', '51185': 'CD5 %', '51307': 'CD13',
                 '51454': 'Plasma Cells', '51341': 'Lambda', '51253': 'Monocyte Count', '51375': 'Lymphocytes',
                 '51078': 'Chloride, Urine', '50962': 'N-Acetylprocainamide (NAPA)', '51467': 'Broad Casts',
                 '51211': 'Factor XIII', '51400': 'CD20', '51070': 'Albumin/Creatinine, Urine', '51113': 'Blasts',
                 '51219': 'H/O Smear', '51311': 'CD16', '50876': 'Anti-Smooth Muscle Antibody',
                 '50905': 'Cholesterol, LDL, Calculated', '51151': 'Burr Cells', '51401': 'CD22', '51431': 'Monos',
                 '51508': 'Urine Color', '51128': 'WBC, Ascites', '50947': 'I',
                 '51299': 'Von Willebrand Factor Antigen',
                 '51316': 'CD22', '51017': 'PEP, CSF', '51550': 'VOIDED SPECIMEN', '50968': 'Phenytoin, Free',
                 '50864': 'Alpha-Fetoprotein', '51540': 'PROBLEM SPECIMEN', '50824': 'Sodium, Whole Blood',
                 '51018': 'Total Protein, CSF', '51343': 'Atypical Lymphocytes', '51291': 'Sickle Cells',
                 '51238': 'Kappa', '51512': 'Urine Mucous', '51545': 'VOIDED SPECIMEN', '50904': 'Cholesterol, HDL',
                 '51421': 'Glyco A', '50860': 'AFP, Maternal Screen', '50985': 'Study Tubes', '51486': 'Leukocytes',
                 '51308': 'CD138', '50881': 'Beta-2 Microglobulin', '50930': 'Globulin', '50833': 'Potassium',
                 '51318': 'CD25', '50970': 'Phosphate', '50988': 'Testosterone', '51389': 'CD103', '50807': 'Comments',
                 '50890': 'C3', '51445': 'Hematocrit, Pleural', '51380': 'NRBC', '51064': 'Potassium, Stool',
                 '51371': 'Joint Crystals, Comment', '50816': 'Oxygen', '50861': 'Alanine Aminotransferase (ALT)',
                 '51394': 'CD14', '51469': 'Calcium Oxalate Crystals', '51434': 'Other Cell',
                 '51261': 'Pappenheimer Bodies', '50869': 'Anti-DGP (IgA/IgG)', '51276': 'Quantitative G6PD',
                 '51096': 'Porphobilinogen Screen', '50865': 'Amikacin', '51426': 'Lambda', '51259': 'Other Cells',
                 '51368': 'Eosinophils', '51083': 'Ethanol, Urine', '51304': 'CD103',
                 '50894': 'Calculated Free Testosterone', '51306': 'CD11c', '51273': 'Protein S, Functional',
                 '51484': 'Ketone', '51227': 'Hemogloblin S', '51414': 'CD57', '51289': 'Serum Viscosity',
                 '51346': 'Blasts', '50913': 'Cryoglobulin', '51348': 'Hematocrit, CSF', '51167': 'CD2',
                 '51336': 'Glyco A', '51505': 'Uric Acid Crystals', '51472': 'Cholesterol Crystals', '51135': 'ADP',
                 '51390': 'CD117', '50831': 'pH', '50873': 'Anti-Nuclear Antibody', '51203': 'Factor IX',
                 '51481': 'Hemosiderin', '51324': 'CD41', '51015': 'Lactate Dehydrogenase, CSF',
                 '51281': 'Reptilase Time Control', '51435': 'Plasma', '50924': 'Ferritin', '51355': 'Monocytes',
                 '50826': 'Tidal Volume', '50811': 'Hemoglobin', '50857': 'Acetone',
                 '51349': 'Hypersegmented Neutrophils', '51313': 'CD19', '51356': 'Myelocytes',
                 '51032': 'Creatinine, Body Fluid', '51497': 'Renal Epithelial Cells', '51526': 'FRUCAMN+',
                 '50837': 'Bicarbonate, Ascites', '51129': 'Young', '51197': 'Elliptocytes', '50984': 'Stat',
                 '51407': 'CD38', '51251': 'Metamyelocytes', '50912': 'Creatinine', '51019': 'Albumin, Joint Fluid',
                 '51080': 'Creatinine Clearance', '51365': 'Atypical Lymphocytes', '51528': 'STDY HOLD',
                 '50867': 'Amylase', '50830': 'pCO2, Body Fluid', '51104': 'Urea Nitrogen, Urine', '51405': 'CD33',
                 '51182': 'CD41', '51471': 'Cellular Cast', '51399': 'CD2', '51179': 'CD38',
                 '51542': 'PROBLEM SPECIMEN',
                 '51029': 'Calcium, Body Fluid', '51430': 'Metamyelocytes', '51206': 'Factor VIII',
                 '51000': 'Triglycerides', '50953': 'Iron Binding Capacity, Total', '51452': 'NRBC',
                 '50829': 'Fluid Type', '51456': 'Promyelocytes', '51242': 'LUC', '51408': 'CD4', '50931': 'Glucose',
                 '51345': 'Basophils', '50879': 'Barbiturate Screen', '51121': 'Myelocytes', '51335': 'FMC-7',
                 '51157': 'CD138', '50982': 'Sex Hormone Binding Globulin', '50979': 'Red Top Hold', '51491': 'pH',
                 '51393': 'CD138', '51050': 'Chloride, Pleural', '51323': 'CD4', '51535': 'CD55',
                 '51552': 'VOIDED SPECIMEN', '50877': 'Anti-Thyroglobulin Antibodies', '51465': 'Bilirubin Crystals',
                 '50902': 'Chloride', '51244': 'Lymphocytes', '51046': 'Albumin, Pleural',
                 '51049': 'Bilirubin, Total, Pleural', '51320': 'CD33', '50992': 'Thyroid Peroxidase Antibodies',
                 '51117': 'Macrophage', '51462': 'Amorphous Crystals', '50896': 'Calculated Thyroxine (T4) Index',
                 '50847': 'Potassium, Ascites', '51213': 'Fibrin Degradation Products', '51315': 'CD20',
                 '51378': 'Metamyelocytes', '51549': 'VOIDED SPECIMEN', '51288': 'Sedimentation Rate',
                 '51233': 'Hypochromia', '51257': 'Nucleated Red Cells', '51475': 'Epithelial Casts',
                 '51120': 'Monocytes', '51379': 'Monocytes', '51022': 'Glucose, Joint Fluid', '51112': 'Basophils',
                 '50883': 'Bilirubin, Direct', '51477': 'Free Fat', '51295': 'TdT', '50850': 'Triglycerides, Ascites',
                 '51110': 'Atypical Lymphocytes', '50845': 'Miscellaneous, Ascites', '51047': 'Amylase, Pleural',
                 '51230': 'HLA-DR', '51279': 'Red Blood Cells', '50960': 'Magnesium', '51333': 'CD71',
                 '51139': 'Anticardiolipin Antibody IgM', '51448': 'Mesothelial Cells', '51460': 'Blood, Occult',
                 '51207': 'Factor VIII Inhibitor', '51079': 'Cocaine, Urine', '51082': 'Creatinine, Urine',
                 '51191': 'CD64', '51105': 'Uric Acid, Urine', '50910': 'Creatine Kinase (CK)', '51009': 'Vancomycin',
                 '51042': 'Sodium, Body Fluid', '51366': 'Bands', '51065': 'Sodium, Stool',
                 '50942': 'Hepatitis B Virus Core Antibody', '50842': 'Glucose, Ascites', '51367': 'Basophils',
                 '51248': 'MCH', '50823': 'Required O2', '51058': 'Sodium, Pleural', '50853': '25-OH Vitamin D',
                 '51069': 'Albumin, Urine', '50803': 'Calculated Bicarbonate, Whole Blood',
                 '51231': 'Howell-Jolly Bodies', '50888': 'Blue Top Hold Frozen', '51359': 'Plasma',
                 '51073': 'Amylase/Creatinine Ratio, Urine', '51547': 'VOIDED SPECIMEN', '50944': 'HIV Antibody',
                 '51294': 'Target Cells', '50897': 'Call', '51360': 'Polys', '51386': 'Bands', '50916': 'DHEA-Sulfate',
                 '51256': 'Neutrophils', '50899': 'Carbamazepine', '50972': 'Procainamide',
                 '51377': 'Mesothelial Cells',
                 '50950': 'Immunoglobulin G', '51493': 'RBC', '51188': 'CD56', '51466': 'Blood',
                 '51140': 'Antithrombin',
                 '51076': 'Bicarbonate, Urine', '51325': 'CD45', '50909': 'Cortisol', '51330': 'CD59',
                 '51272': 'Protein S, Antigen', '51144': 'Bands', '50934': 'H', '50882': 'Bicarbonate',
                 '50955': 'Light Green Top Hold', '51319': 'CD3', '51532': 'PLASMGN', '50855': 'Absolute Hemoglobin',
                 '51458': 'WBC, Pleural', '51006': 'Urea Nitrogen', '51150': 'Blood Parasite Smear', '51159': 'CD15',
                 '51554': 'VOIDED SPECIMEN', '51090': 'Methadone, Urine', '51382': 'Polys',
                 '51176': 'CD3 Cells, Percent', '51171': 'CD22', '51329': 'CD57', '51522': 'PG', '51249': 'MCHC',
                 '50895': 'Calculated TBG', '51305': 'CD117', '51284': 'Reticulocyte Count, Manual',
                 '51222': 'Hemoglobin', '51388': 'CD10', '51212': 'Fetal Hemoglobin', '51518': 'WBC Clumps',
                 '51093': 'Osmolality, Urine', '51513': 'Urine Specimen Type', '51142': 'Arachadonic Acid',
                 '50935': 'Haptoglobin', '50937': 'Hepatitis A Virus Antibody', '50976': 'Protein, Total',
                 '51141': 'APT Test', '51395': 'CD15', '51158': 'CD14', '51161': 'CD16', '51523': 'GR HOLD',
                 '51500': 'Sulfonamides', '51473': 'Cysteine Crystals', '51384': 'WBC, Joint Fluid',
                 '51496': 'Reducing Substances, Urine', '51234': 'Immunophenotyping', '51302': 'Young Cells',
                 '50952': 'Iron', '51344': 'Bands', '51327': 'CD55', '51170': 'CD20 Absolute Count',
                 '51352': 'Macrophage', '51240': 'Large Platelets', '51250': 'MCV', '50964': 'Osmolality, Measured',
                 '51544': 'VOIDED SPECIMEN', '51495': 'RBC Clumps', '50858': 'Acid Phosphatase', '50914': 'Cyclosporin',
                 '51470': 'Calcium Phosphate Crystals', '50929': 'Gentamicin', '51503': 'Triple Phosphate Crystals',
                 '51088': 'Magnesium, Urine', '51398': 'CD19', '51247': 'MacroOvalocytes', '51062': 'Chloride, Stool',
                 '50875': 'Anti-Parietal Cell Antibody', '51108': 'Urine Volume',
                 '50911': 'Creatine Kinase, MB Isoenzyme', '51106': 'Urine Creatinine',
                 '51373': 'Joint Crystals, Number', '51184': 'CD5', '51148': 'Blasts',
                 '50849': 'Total Protein, Ascites',
                 '51282': 'Reticulocyte Count, Absolute', '51298': 'Von Willebrand Factor Activity', '51364': 'Young',
                 '51038': 'Miscellaneous, Body Fluid', '51048': 'Bicarbonate, Pleural',
                 '51370': 'Joint Crystals, Birefringence', '50844': 'Lipase, Ascites',
                 '50870': 'Anti-Gliadin Antibody, IgA', '51515': 'Waxy Casts', '51519': 'Yeast',
                 '51553': 'VOIDED SPECIMEN', '51533': 'WBCP', '51423': 'HLA-DR', '50974': 'Prostate Specific Antigen',
                 '51195': 'Collagen', '51255': 'Myelocytes', '50975': 'Protein Electrophoresis',
                 '50940': 'Hepatitis B Surface Antibody', '50977': 'Quinidine', '51026': 'Amylase, Body Fluid',
                 '51229': 'Heparin, LMW', '51392': 'CD13', '51331': 'CD64', '51228': 'Heparin',
                 '50817': 'Oxygen Saturation', '51258': 'Osmotic Fragility', '51524': 'FATTY', '51450': 'Monos',
                 '51208': 'Factor X', '50813': 'Lactate', '51056': 'Miscellaneous, Pleural', '50945': 'Homocysteine',
                 '51317': 'CD23', '51514': 'Urobilinogen', '51433': 'NRBC', '51427': 'Lymphocytes', '51172': 'CD23',
                 '51439': 'WBC, Other Fluid', '50812': 'Intubated', '50863': 'Alkaline Phosphatase',
                 '51115': 'Hematocrit, Ascites', '51210': 'Factor XII', '50819': 'PEEP', '51334': 'CD8',
                 '51224': 'Hemoglobin C', '50893': 'Calcium, Total', '51152': 'CD10', '51134': 'Acanthocytes',
                 '50958': 'Luteinizing Hormone', '50840': 'Cholesterol, Ascites', '51290': 'Sickle Cell Preparation',
                 '51509': 'Urine Comments', '51102': 'Total Protein, Urine', '51415': 'CD64', '51252': 'Microcytes',
                 '50989': 'Testosterone, Free', '51103': 'Uhold', '50862': 'Albumin', '51548': 'VOIDED SPECIMEN',
                 '51449': 'Metamyelocytes', '50978': 'Rapamycin', '51385': 'Atypical Lymphocytes',
                 '51280': 'Reptilase Time', '51468': 'Calcium Carbonate Crystals', '51297': 'Thrombin',
                 '51285': 'Reticulocyte, Cellular Hemoglobin', '51031': 'Cholesterol, Body Fluid',
                 '51014': 'Glucose, CSF', '50841': 'Creatinine, Ascites', '51361': 'Promyelocytes', '51520': 'ANTI-MC',
                 '50922': 'Ethanol', '51127': 'RBC, Ascites', '51488': 'Non-squamous Epithelial Cells',
                 '51489': 'NonSquamous Epithelial Cell', '50886': 'Blood Culture Hold',
                 '51071': 'Amphetamine Screen, Urine', '51516': 'WBC', '51074': 'Barbiturate Screen, Urine',
                 '50927': 'Gamma Glutamyltransferase', '51541': 'PROBLEM SPECIMEN', '51118': 'Mesothelial Cell',
                 '50994': 'Thyroxine (T4)', '51153': 'CD103', '50868': 'Anion Gap', '50951': 'Immunoglobulin M',
                 '51441': 'Bands', '50880': 'Benzodiazepine Screen', '50980': 'Rheumatoid Factor',
                 '51353': 'Mesothelial cells', '51537': 'TDT', '51043': 'Total Protein, Body Fluid',
                 '50836': 'Amylase, Ascites', '51527': 'MS-DIA', '51225': 'Hemoglobin F', '50839': 'Chloride, Ascites',
                 '51193': 'CD71', '50949': 'Immunoglobulin A', '51293': 'Sugar Water Test', '51008': 'Valproic Acid',
                 '50999': 'Tricyclic Antidepressant Screen', '51339': 'Iron Stain', '51100': 'Sodium, Urine',
                 '51086': 'Immunofixation, Urine', '51136': 'Alpha Antiplasmin', '51164': 'CD19', '51089': 'Marijuana',
                 '50995': 'Thyroxine (T4), Free', '51453': 'Other', '50838': 'Bilirubin, Total, Ascites',
                 '51417': 'CD71', '51403': 'CD25', '50917': 'Digoxin', '51517': 'WBC Casts', '51237': 'INR(PT)',
                 '50936': 'HCG, Maternal Screen', '51357': 'NRBC', '50966': 'Phenobarbital',
                 '51054': 'Lactate Dehydrogenase, Pleural', '50852': '% Hemoglobin A1c', '51461': 'Ammonium Biurate',
                 '51487': 'Nitrite', '51098': 'Prot. Electrophoresis, Urine', '51278': 'Red Blood Cell Fragments',
                 '51536': 'CD59', '51092': 'Opiate Screen, Urine', '51413': 'CD56', '51016': 'Miscellaneous, CSF',
                 '51202': 'Factor II', '51168': 'CD20', '51270': 'Protein C, Antigen', '51442': 'Basophils',
                 '51174': 'CD3 %', '51410': 'CD41', '51095': 'Phosphate, Urine', '50956': 'Lipase',
                 '51199': 'Eosinophil Count', '51177': 'CD33', '51480': 'Hematocrit', '51131': 'Absolute CD4 Count',
                 '50906': 'Cholesterol, LDL, Measured', '51246': 'Macrocytes', '50804': 'Calculated Total CO2',
                 '50821': 'pO2', '51263': 'Plasma Cells', '50809': 'Glucose', '50918': 'Double Stranded DNA',
                 '51429': 'Mesothelial cells', '51374': 'Joint Crystals, Shape', '51215': 'FMC-7',
                 '51292': 'Spherocytes', '50892': 'CA-125', '51063': 'Osmolality, Stool', '51002': 'Troponin I',
                 '50901': 'Centromere', '50933': 'Green Top Hold (plasma)', '50871': 'Anti-Mitochondrial Antibody',
                 '50903': 'Cholesterol Ratio (Total/HDL)', '51192': 'CD7', '50884': 'Bilirubin, Indirect',
                 '51525': 'Billed', '50932': 'Gray Top Hold (plasma)', '51221': 'Hematocrit', '50983': 'Sodium',
                 '51044': 'Triglycer', '51338': 'Immunophenotyping', '51424': 'Immunophenotyping',
                 '50822': 'Potassium, Whole Blood', '50851': 'Urea Nitrogen, Ascites', '50990': 'Theophylline',
                 '51045': 'Urea Nitrogen, Body Fluid', '50802': 'Base Excess', '51283': 'Reticulocyte Count, Automated',
                 '51235': 'Inhibitor Screen', '50808': 'Free Calcium', '51369': 'Hematocrit, Joint Fluid',
                 '51266': 'Platelet Smear', '50959': 'Macro Prolactin', '50925': 'Folate', '51309': 'CD14',
                 '50872': 'Anti-Neutrophil Cytoplasmic Antibody', '50887': 'Blue Top Hold', '50820': 'pH',
                 '51084': 'Glucose, Urine', '50846': 'Osmolality, Ascites', '51432': 'Myelocytes',
                 '50926': 'Follicle Stimulating Hormone', '51166': 'CD19 Absolute Count', '51428': 'Macrophage',
                 '51391': 'CD11c', '51485': 'Leucine Crystals', '51072': 'Amylase, Urine', '51483': 'Hyphenated Yeast',
                 '51241': 'Leukocyte Alkaline Phosphatase', '50907': 'Cholesterol, Total',
                 '51245': 'Lymphocytes, Percent', '51165': 'CD19 %', '51464': 'Bilirubin', '50981': 'Salicylate',
                 '50969': 'Phenytoin, Percent Free', '51200': 'Eosinophils', '51147': 'Bite Cells', '51183': 'CD45',
                 '51260': 'Ovalocytes', '51322': 'CD38', '50993': 'Thyroid Stimulating Hormone',
                 '51269': 'Promyelocytes', '51226': 'Hemogloblin A', '51160': 'CD16/56',
                 '50996': 'Tissue Transglutaminase Ab, IgA', '51201': 'Epinepherine', '50848': 'Sodium, Ascites',
                 '50891': 'C4', '51363': 'WBC, CSF', '51130': 'Absolute CD3 Count', '51267': 'Poikilocytosis',
                 '51271': 'Protein C, Functional', '51531': 'STDYURINE', '51040': 'Phosphate, Body Fluid',
                 '50946': 'Human Chorionic Gonadotropin', '51510': 'Urine Crystals, Other', '51296': 'Teardrop Cells',
                 '51196': 'D-Dimer', '51001': 'Triiodothyronine (T3)', '51406': 'CD34',
                 '51138': 'Anticardiolipin Antibody IgG', '51007': 'Uric Acid', '51024': 'Total Protein, Joint Fluid',
                 '51507': 'Urine Casts, Other', '50997': 'Tobramycin', '50948': 'Immunofixation',
                 '50908': 'CK-MB Index',
                 '51387': 'Basophils', '51155': 'CD11c', '51094': 'pH', '50915': 'D-Dimer', '51358': 'Other',
                 '51021': 'Creatinine, Joint Fluid', '51178': 'CD34', '50954': 'Lactate Dehydrogenase (LD)',
                 '51041': 'Potassium, Body Fluid', '50854': 'Absolute A1c', '51342': 'Wright Giemsa',
                 '50939': 'Hepatitis B Core Antibody, IgM', '51085': 'HCG, Urine, Qualitative',
                 '51498': 'Specific Gravity', '51443': 'Blasts', '50900': 'Carcinoembyronic Antigen (CEA)',
                 '50961': 'Methotrexate', '50859': 'Acid Phosphatase, Non-Prostatic', '51101': 'Total Collection Time',
                 '51404': 'CD3', '51402': 'CD23', '50973': 'Prolactin', '51075': 'Benzodiazepine Screen, Urine',
                 '51381': 'Other', '51186': 'CD5 Absolute Count', '51154': 'CD117', '51312': 'CD16/56',
                 '51543': 'VOIDED SPECIMEN', '51034': 'Glucose, Body Fluid', '50805': 'Carboxyhemoglobin',
                 '51114': 'Eosinophils', '51116': 'Lymphocytes', '51506': 'Urine Appearance',
                 '51218': 'Granulocyte Count', '51087': 'Length of Urine Collection', '51020': 'Amylase, Joint Fluid',
                 '51232': 'Hypersegmented Neutrophils', '51438': 'RBC, Other Fluid', '50928': 'Gastrin',
                 '51190': 'CD59',
                 '51534': 'MYELOS', '50889': 'C-Reactive Protein', '50801': 'Alveolar-arterial Gradient',
                 '50965': 'Parathyroid Hormone', '50814': 'Methemoglobin', '51025': 'Albumin, Body Fluid',
                 '51146': 'Basophils', '50963': 'NTproBNP', '51137': 'Anisocytosis', '51239': 'Lambda',
                 '51463': 'Bacteria', '51436': 'Polys', '50835': 'Albumin, Ascites', '51109': 'Urine Volume, Total',
                 '51555': 'SURFACTANT ALBUMIN RATIO', '51122': 'Nucleated RBC', '51033': 'FetalFN',
                 '51457': 'RBC, Pleural', '51411': 'CD45', '51162': 'CD16/56 Absolute Count', '51077': 'Calcium, Urine',
                 '51494': 'RBC Casts', '51030': 'Chloride, Body Fluid', '51081': 'Creatinine, Serum',
                 '51214': 'Fibrinogen, Functional', '50921': 'Estradiol', '50957': 'Lithium', '50986': 'tacroFK',
                 '51175': 'CD3 Absolute Count', '51060': 'Triglycerides, Pleural', '50967': 'Phenytoin', '51412': 'CD5',
                 '51061': 'Bicarbonate, Stool', '51444': 'Eosinophils', '51133': 'Absolute Lymphocyte Count',
                 '50806': 'Chloride, Whole Blood', '50885': 'Bilirubin, Total', '51416': 'CD7', '51310': 'CD15',
                 '50810': 'Hematocrit, Calculated', '51321': 'CD34', '51362': 'RBC, CSF', '51326': 'CD5',
                 '50834': 'Sodium, Body Fluid', '51243': 'Lupus Anticoagulant', '50874': 'Anti-Nuclear Antibody, Titer',
                 '51479': 'Granular Casts', '51053': 'Glucose, Pleural', '51011': 'Albumin',
                 '51052': 'Creatinine, Pleural', '50832': 'pO2, Body Fluid', '50943': 'Hepatitis C Virus Antibody',
                 '50843': 'Lactate Dehydrogenase, Ascites', '51125': 'Polys', '51004': 'UE3, Maternal Screen',
                 '51419': 'Eosinophils', '50991': 'Thyroglobulin', '51039': 'Osmolality, Body Fluid',
                 '50971': 'Potassium', '51300': 'WBC Count', '51012': 'Bilirubin, Total, CSF', '51420': 'FMC-7',
                 '51055': 'Lipase, Pleural', '50818': 'pCO2', '51254': 'Monocytes', '51265': 'Platelet Count',
                 '51474': 'Eosinophils', '51277': 'RDW', '51337': 'HLA-DR', '51220': 'Heinz Body Prep',
                 '51013': 'Chloride, CSF', '50827': 'Ventilation Rate', '51005': 'Uptake Ratio',
                 '51036': 'Lipase, Body Fluid', '51286': 'Ristocetin', '51459': 'Young Cells',
                 '51132': 'Absolute CD8 Count', '51287': 'Schistocytes', '51187': 'CD55', '51492': 'Protein',
                 '51303': 'CD10', '50923': 'Fax', '51066': '24 hr Calcium', '50815': 'O2 Flow', '51209': 'Factor XI',
                 '51451': 'Myelocytes', '51422': 'Hematocrit, Other Fluid', '51328': 'CD56', '51204': 'Factor V',
                 '51351': 'Lymphs', '51301': 'White Blood Cells', '51181': 'CD4/CD8 Ratio',
                 '50878': 'Asparate Aminotransferase (AST)', '51502': 'Trichomonas', '51418': 'CD8', '51156': 'CD13',
                 '51173': 'CD25', '51099': 'Protein/Creatinine Ratio', '51314': 'CD2',
                 '51028': 'Bilirubin, Total, Body Fluid', '51236': 'Inpatient Hematology/Oncology Smear',
                 '51023': 'LD, Joint Fluid', '51504': 'Tyrosine Crystals', '51059': 'Total Protein, Pleural',
                 '51163': 'CD16/56%', '51107': 'Urine tube, held', '51205': 'Factor VII',
                 '51529': 'Estimated Actual Glucose', '51354': 'Metamyelocytes', '51437': 'Promyelocytes',
                 '51126': 'Promyelocytes', '50920': 'Estimated GFR (MDRD equation)', '51037': 'Magnesium, Body Fluid',
                 '51027': 'Bicarbonate, Other Fluid', '51350': 'Immunophenotyping', '51383': 'RBC, Joint Fluid',
                 '51198': 'Envelope Cells', '51223': 'Hemoglobin A2', '51455': 'Polys', '51440': 'Atypical Lymphocytes',
                 '51490': 'Oval Fat Body', '51145': 'Basophilic Stippling', '51091': 'Myoglobin, Urine',
                 '51521': 'ACID PHOSPHATASE, PROSTATIC', '51546': 'VOIDED SPECIMEN', '51003': 'Troponin T'}

    # create dict for lab tests
    lab_dict = {}
    for lab_test in lab_tests:
        lab_dict[lab_test] = lab_codes[lab_test]

    # create dict for current diagnoses
    diag_dict = {}
    for diagnosis in diagnoses:
        diag_dict[diagnosis] = diag_codes[diagnosis]

    # create dict for predicted diagnoses
    predicted_diag_dict = {}
    for i in prediction:
        i = i.split()
        predicted_diag_dict[i[2]] = (i[5], ' '.join(i[8:]))

    return [lab_dict, diag_dict, predicted_diag_dict]


def sendMail(emailID, name, lab_tests, diagnoses, seq):
    msg = Message('Your Predicted Diagnosis', sender='compvisionnn@gmail.com', recipients=[emailID])
    info = get_info_for_mail(lab_tests.split(','), diagnoses.split(','), seq)
    html_message = render_template('send_email.html', name=name, lab_test=info[0], diag=info[1], pred=info[2])
    msg.html = html_message
    mail.send(msg)


def get_disease_info(ICD):
    # 1
    if ICD == 'd_428':
        return ["https://shawglobalnews.files.wordpress.com/2016/02/heart-failure.jpg?quality=70&strip=all&strip=all",
                Markup("""

        <b>Description</b> <br>Congestive heart failure (CHF) is a chronic progressive condition that affects the pumping power of your heart muscles. While 	often referred to simply as “heart failure,” CHF specifically refers to the stage in which fluid builds up around the heart and causes it to pump inefficiently. <br><br><br>
        <b>Symptoms</b> <br><u1>
                                 <li>Fatigue</li>
                                 <li>Irregular heart beat</li>
                                 <li>Swelling in the ankles, feet, and legs.</li>
                                 <li>Chest pain that radiates through the upper body</li>
                                 <li>Increased need to urinate, especially at night.</li>
                                 <li>Shortness of breath, which may indicate pulmonary edem</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>Diagnostic tests may include an electrocardiogram (ECG or EKG), an echocardiogram (cardiac echo), and cardiac catheterization. The purpose of these tests is to evaluate heart function (e.g., assess ejection fraction), and to detect coronary artery disease, heart attack, and valve dysfunction. <br><br><br>
        <b>Potential Complications</b> <br><u1>
                                        <li>Arrhythmias - Atrial fibrillation; ventricular arrhythmias (ventricular tachycardia, ventricular fibrillation); bradyarrhythmias</li>
                                        <li>Thromboembolism - Stroke</li>
                                        <li>Peripheral embolism</li>
                                        <li>Deep venous  thrombosis</li>
                                        <li>Pulmonary embolism</li>
                                        <li>Gastrointestinal - Hepatic congestion and hepatic dysfunction; malabsorption</li>
                                        <li>Musculoskeletal - Muscle wasting</li>
                                        <li>Respiratory - Pulmonary congestion; respiratory muscle weakness; pulmonary hypertension (rare)</li>
                                  </u1> <br><br>
        <b>Prevention</b> <br>Smoking or using tobacco of any kind is one of the most significant risk factors for developing heart disease. Chemicals in tobacco can damage your heart and blood vessels, leading to narrowing of the arteries due to plaque buildup (atherosclerosis). Atherosclerosis can ultimately lead to a heart attack. In addition to that, eating a healthy diet can reduce your risk of heart disease. Two examples of heart-healthy food plans include the Dietary Approaches to Stop Hypertension (DASH) eating plan and the Mediterranean diet.<br><br><br>
        """)]


    elif ICD == 'd_560':
        # 2
        return ["https://www.epainassist.com/images/Article-Images/bowel-obstruction.jpg",
                Markup("""
        <b>Description</b> <br>An intestinal obstruction is a potentially serious condition in which the intestines are blocked. The blockage may be either partial or complete, occurring at one or more locations. \n Both the small intestine and large intestine, called the colon, can be affected. When a blockage occurs, food and drink cannot pass through the body. Obstructions are serious and need to be treated immediately. They may even require surgery. <br><br>
        <b>Symptoms</b> <u1>
                                 <li>Severe abdominal pain, cramps that come in waves</li>
                                 <li>Bloating, nausea and vomiting</li>
                                 <li>Diarrhea, constipation or inability to have a bowel movement</li>
                                 <li>Inability to pass gas, distention or swelling of the abdomen</li>
                                 <li>Loud noises from the abdomen, foul breath</li>
                         </u1><br>
        <b>Diagnosis</b> <br>To diagnose a bowel obstruction, your doctor will need to feel and listen to your abdomen and feel inside your rectum. A blockage in the intestine is confirmed by X-rays of your abdomen, which show gas and liquid bowel contents above the area of the blockage, but no gas below the blockage. <br><br>
        <b>Potential Complications</b> <br><u1>
                                            <li>Infection, tissue death</li>
                                            <li>Intestinal Perforation, sepsis</li>
                                            <li>Multi system organ failure, death</li>
                                  </u1> <br>
        <b>Prevention</b> <br>Eat a balanced diet low in fat with plenty of vegetables and fruits, don't smoke, and see your doctor for colorectal cancer screening once a year after age 50.<br><br>
        """)]


    elif ICD == 'd_276':
        # 3
        return ["http://www.healthcareinnovationtransfer.org/images/mission.jpg",
                Markup("""
        <b>Description</b> <br>Total body water is a function of age, body mass, and body fat. Body water declines throughout life, ultimately comprising about 45% of total body mass in old age. <br>
        <b>Symptoms</b> <br><u1>
                                 <li>Lethargy, Convulsions or seizures</li>
                                 <li>Vomiting, Diarrhea or constipation</li>
                                 <li>Abdominal cramping, Muscle weakness</li>
                                 <li>Muscle cramping, Irritability</li>
                                 <li>Confusion, Headaches</li>
                         </u1>
        <b>Diagnosis</b> <br>The most common tests conducted for the diagnosis of Disorders of fluid electrolyte and acid-base balance are Bicarbonate test, Chloride test, Electrolyte test and Osmality test <br>
        <b>Potential Complications</b> <br><u1>
                                         <li>Calcium: hypercalcemia and hypocalcemia</li>
                                         <li>Chloride: hyperchloremia and hypochloremia</li>
                                         <li>Magnesium: hypermagnesemia and hypomagnesemia</li>
                                  </u1>
        <b>Prevention</b> <br>Stay hydrated. Drink at least 8 glasses of water each day. Drink additional glasses of water after exercising heavily, vomiting or experiencing diarrhea. Get enough vitamin D. Spend time outdoors in the sunlight and eat foods that contain vitamin D. Eat a balanced diet. <br>
        """)]


    elif ICD == 'd_427':
        # 4
        return ["http://www.sunshinedrugs.com/wp-content/uploads/2015/02/Heart-Infographic.png",
                Markup("""
        <b>Description</b> <br>Cardiac arrhythmia, also known as cardiac dysrhythmia or irregular heartbeat, is a group of conditions in which the heartbeat is irregular, too fast, or too slow<br><br>
        <b>Symptoms</b> <br><u1>
                                 <li>Pain areas: in the chest</li>
                                 <li>Whole body: collapse, dizziness, fainting, or light-headedness</li>
                                 <li>Heart: cardiac arrest, sensation of an abnormal heartbeat, or slow heart rate</li>
                                 <li>Also common: shortness of breath</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>To diagnose a heart arrhythmia, your doctor will review your symptoms and your medical history and conduct a physical examination. Your doctor may ask about — or test for — conditions that may trigger your arrhythmia, such asheart disease or a problem with your thyroid gland. <br><br>
        <b>Potential Complications</b> <br><u1>
                                             <li>A racing heartbeat (tachycardia)</li>
                                             <li>A slow heartbeat (bradycardia)</li>
                                             <li>Chest pain</li>
                                  </u1> <br>
        <b>Prevention</b> <br>Lifestyle choices such as smoking, drinking, and use of illegal drugs can increase your chances of developing an arrhythmia. An arrhythmia can also occur due to another condition, such as coronary artery disease, congestive heart failure, or diabetes. You may not be able to prevent the development of anarrhythmia<br><br>
        """)]


    elif ICD == 'd_250':
        # 5
        return ["http://sagememorial.com/wp-content/uploads/2014/08/Hispanic_ENG-1024x806.jpg",
                Markup("""
        <b>Description</b> <br>Diabetes mellitus (DM), commonly referred to as diabetes, is a group of metabolic diseases in which there are high blood sugar levels over a prolonged period.<br><br><br>
        <b>Symptoms</b> <br><u1>
                                 <li>Type 2 diabetes-  Symptoms include increased thirst, frequent urination, hunger, fatigue and blurred vision. In some cases, there may be no symptoms</li>
                                 <li>Type 1 diabetes-  Symptoms include increased thirst, frequent urination, hunger, fatigue and blurred vision</li>
                                 <li>Prediabetes- Many people with prediabetes have no symptoms</li>
                                 <li>Gestational diabetes- In most cases, there are no symptoms. A blood sugar test during pregnancy is used for diagnosis</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>Diabetes may be diagnosed based on A1C criteria or plasma glucose criteria, either the fasting plasma glucose (FPG) or the 2-h plasma glucose (2-h PG) value after a 75-g oral glucose tolerance test (OGTT)<br><br><br>
        <b>Potential Complications</b> <br><u1>
                                            <li>Cardiovascular disease</li>
                                            <li>Nerve damage (neuropathy)</li>
                                            <li>Kidney damage (nephropathy)</li>
                                            <li>Eye damage (retinopathy)</li>
                                            <li>Foot damage</li>
                                            <li>Skin conditions</li>
                                            <li>Hearing impairment</li>
                                            <li>Alzheimer's disease</li>
                                  </u1> <br><br>
        <b>Prevention</b> <br>Physical activity is one of the main pillars in the prevention of diabetes. Increased physical activity is important in maintaining weight loss and is linked to reduced blood pressure, reduced resting heart rate, increased insulin sensitivity, improved body composition and psychological well-being. All smokers should be encouraged to quit smoking. However, weight gain is common when quitting smoking and therefore dietary advice on avoiding weight gain should also be given (e.g. managing cravings and withdrawal symptoms by using short bouts of physical activity as a stress-relief activity,<br><br><br>
        """)]


    elif ICD == 'd_401':
        # 6
        return ["http://dqw-6ac9.kxcdn.com/wp-content/uploads/2016/05/World-Hypertension-Day-Infographic.png",
                Markup("""
        <b>Description</b> <br>Hypertension (HTN or HT), also known as high blood pressure (HBP), is a long term medical condition in which the blood pressure in the arteries is persistently elevated.High blood pressure usually does not cause symptoms.Long term high blood pressure, however, is a major risk factor for coronary artery disease, stroke, heart failure, peripheral vascular disease, vision loss, and chronic kidney disease.<br><br>
        <b>Symptoms</b> <br><u1>
                                  <li>High blood pressure is generally a chronic condition and is often associated with few or no symptoms</li>
                                  <li>When symptoms do occur, it is usually when blood pressure spikes suddenly and extremely enough to be considered a medical emergency</li>
                                  <li>Rare symptoms include dizzy spells, headaches, and nosebleeds</li>
                         </u1> <br>
        <b>Diagnosis</b> <br>Blood pressure is measured in millimeters of mercury (mm Hg) and is written systolic over diastolic (for example, 120/80 mm Hg, or "120 over 80"). According to the most recent guidelines, a normal blood pressure is less than 120/80 mm Hg.Hypertension is blood pressure that is greater than 140/90.<br><br>
        <b>Potential Complications</b> <br><u1>
                                            <li>Heart attack or stroke- High blood pressure can cause hardening and thickening of the arteries (atherosclerosis), which can lead to a heart attack, stroke or other complications</li>
                                            <li>Aneurysm- Increased blood pressure can cause your blood vessels to weaken and bulge, forming an aneurysm. If an aneurysm ruptures, it can be life-threatening</li>
                                            <li>Heart failure- To pump blood against the higher pressure in your vessels, your heart muscle thickens. Eventually, the thickened muscle may have a hard time pumping enough blood to meet your body's needs, which can lead to heart failure</li>
                                            <li>Weakened and narrowed blood vessels in your kidneys- This can prevent these organs from functioning normally</li>
                                            <li>Thickened, narrowed or torn blood vessels in the eyes- This can result in vision loss</li>
                                            <li>Metabolic syndrome- This syndrome is a cluster of disorders of your body's metabolism, including increased waist circumference; high triglycerides; low high-density lipoprotein (HDL) cholesterol, the "good" cholesterol; high blood pressure; and high insulin levels. These conditions make you more likely to develop diabetes, heart disease and stroke</li>
                                            <li>Trouble with memory or understanding. Uncontrolled high blood pressure may also affect your ability to think, remember and learn. Trouble with memory or understanding concepts is more common in people with high blood pressure</li>
                                  </u1> <br>
        <b>Prevention</b> <br>Be sure to eat plenty of fresh fruits and vegetables. Eating foods low in salt (sodium) and high in potassium can lower your blood pressure. The DASH (Dietary Approaches to Stop Hypertension) eating plan is one healthy diet that is proven to help people lower their blood pressure. <br><br>
        """)]

    elif ICD == 'd_518':
        # 7
        return ["https://s-media-cache-ak0.pinimg.com/originals/54/4c/df/544cdf2776c67e3cf1dbcfbe147f8f7e.png",
                Markup("""
        <b>Description</b> <br>This category of diseases usually include the following: Asthma, chronic bronchitis, and emphysema, Infections, such as influenza and pneumonia, Lung cancer, Sarcoidosis (sar-KOY-doh-sis) and pulmonary fibrosis.<br><br>
        <b>Symptoms</b> <br><u1>
                                 <li>Cough: can be dry, with phlegm, can occur due to smoking, during exercise, or severe</li>
                                 <li>Respiratory: difficulty breathing, expiratory wheezing, fast breathing, frequent respiratory infections, rapid breathing, shortness of breath, shortness of breath at night, shortness of breath on exercise, shortness of breath on lying down, or wheezing</li>
                                 <li>Whole body: fatigue or inability to exercise</li>
                                 <li>Weight: underweight or weight loss</li>
                                 <li>Also common: phlegm, anxiety, bulging chest, chest tightness, depression, fast heart rate, flare, high carbon dioxide levels in blood, limping, muscle weakness, phlegm with pus, or sleeping difficulty</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>Your doctor may use blood tests, pulmonary function testing (spirometry), pulse oximetry, chest x-ray, chest CT, bronchoscopy and biopsy, or surgical biopsy to helpdiagnose your condition. Treatment depends on the underlying cause of the disease and your health status.<br><br>
        <b>Potential Complications</b> <br><u1>
                                            <li>Heart failure</li>
                                            <li>Osteoporosis</li>
                                            <li>Depression are possible complications of COPD</li>
                                            <li>Chronic obstructive pulmonary disease, or COPD, is a progressive lung condition commonly causes various symptoms, including coughing, wheezing, and shortness of breath</li>
                                            <li>It can also cause numerous complications to develop over time</li>
                                  </u1> <br><br>
        <b>Prevention</b> <br> There's no sure way to prevent lung cancer, but you can reduce your risk if you: Don't smoke. If you've never smoked, don't start. Talk to your children about not smoking so that they can understand how to avoid this major risk factor for lungcancer.<br><br>
        """)]

    elif ICD == 'd_414':
        # 8
        return [
            "https://lh6.googleusercontent.com/-QpabjwjhUL0/VON57Y4FA6I/AAAAAAAAAU8/niuD3Xv0Mq8/w1200-h900/Heart-Disease-Infographic_FINAL%2B%25282%2529.JPG",
            Markup("""
        <b>Description</b> <br>Ischemic heart disease is a condition of recurring chest pain or discomfort that occurs when a part of the heart does not receive enough blood. This condition occurs most often during exertion or excitement, when the heart requires greater blood flow. <br><br>
        <b>Symptoms</b> <br><u1>
                                 <li>Extreme fatigue</li>
                                 <li>Shortness of breath</li>
                                 <li>Dizziness, lightheadedness, or fainting</li>
                                 <li>Chest pain and pressure, known as angina</li>
                                 <li>Heart palpitations</li>
                                 <li>Swelling in your legs and feet, known as edema</li>
                                 <li>Swelling in your abdomen</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>Your doctor will diagnose coronary heart disease (CHD) based on your medical and family histories, your risk factors for CHD, a physical exam, and the results from tests and procedures. No single test can diagnose CHD. If your doctor thinks you have CHD, he or she may recommend one or more of the following tests- EKG (Electrocardiogram), Stress Testing, Echocardiography, Chest X Ray, Blood Tests,  Coronary Angiography and Cardiac Catheterization. <br><br>
        <b>Potential Complications</b> <br><u1>
                                        <li>Chest pain (angina)- You may feel pressure or tightness in your chest, as if someone were standing on your chest. This pain, referred to as angina, usually occurs on the middle or left side of the chest. Angina is generally triggered by physical or emotional stress.</li>
                                        <li>The pain usually goes away within minutes after stopping the stressful activity. In some people, especially women, this pain may be fleeting or sharp and felt in the neck, arm or back.</li>
                                        <li>Shortness of breath. If your heart can't pump enough blood to meet your body's needs, you may develop shortness of breath or extreme fatigue with exertion.</li>
                                        <li>Heart attack. A completely blocked coronary artery may cause a heart attack. The classic signs and symptoms of a heart attack include crushing pressure in your chest and pain in your shoulder or arm, sometimes with shortness of breath and sweating.</li>

                                  </u1> <br><br>
        <b>Prevention</b> <br>A heart-healthy lifestyle can lower the risk of CHD. If you already have CHD, a heart-healthy lifestyle may prevent it from getting worse. Heart-healthy lifestyle changes include: Heart-healthy eating, Aiming for a healthy weight, Managing stress, Physical activity, Quitting smoking. Many lifestyle habits begin during childhood. <br><br>
        """)]

    elif ICD == 'd_285':
        # 9
        return ["http://www.who.int/mediacentre/infographic/nutrition/anemia-jpg-large.jpg",
                Markup("""
        <b>Description</b> <br>A condition in which the blood doesn't have enough healthy red blood cells.<br><br>
        <b>Symptoms</b> <br><u1>
                                 <li>Fatigue</li>
                                 <li>Skin pallor</li>
                                 <li>Shortness of breath</li>
                                 <li>Light-headedness</li>
                                 <li>Dizziness or a fast heartbeat</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>Your doctor will do a physical exam to find out how severe your anemia is and to check for possible causes. He or she may: Listen to your heart for a rapid or irregular heartbeat; Listen to your lungs for rapid or uneven breathing; Feel your abdomen to check the size of your liver and spleen; Your doctor also may do a pelvic or rectal exam to check for common sources of blood loss.<br><br>
        <b>Potential Complications</b> <br><u1>
                                        <li>Intestinal disorders- Having an intestinal disorder that affects the absorption of nutrients in your small intestine — such as Crohn's disease and celiac disease — puts you at risk of anemia</li>
                                        <li>Menstruation- In general, women who haven't experienced menopause have a greater risk of iron deficiency anemia than do men and postmenopausal women. That's because menstruation causes the loss of red blood cells</li>
                                        <li>Pregnancy- If you're pregnant and aren't taking a multivitamin with folic acid, you're at an increased risk of anemia</li>
                                        <li>Chronic conditions- If you have cancer, kidney failure or another chronic condition, you may be at risk of anemia of chronic disease. These conditions can lead to a shortage of red blood cells</li>
                                  </u1> <br><br>
        <b>Prevention</b> <br>In order to prevent anemia, it is important to: Eat a balanced healthy diet rich in iron. Reduce tea and coffee intake as they make it harder for your body to absorb iron. Increase vitamin C intake as it may help iron absorption.<br><br>
        """)]

    elif ICD == 'd_272':
        # 10
        return ["http://muscularstrength.com/uploads/froala/091eae1a0ab03ed8672b3e055f20f31247f9a559.jpg",
                Markup("""
        <b>Description</b> <br>Familial hypercholesterolemia (FH), also known as pure hypercholesterolemia, refers to a condition in which people have a genetic tendency for high cholesterol or high lipid levels. High cholesterol can boost a person's chances for developing heart disease and other conditions.<br><br>
        <b>Symptoms</b> <br><u1>
                                 <li>Fatty skin deposits called xanthomas over parts of the hands, elbows, knees, ankles, and around the cornea of the eye</li>
                                 <li>Cholesterol deposits in the eyelids (xanthelasmas)</li>
                                 <li>Chest pain (angina) or other signs of coronary artery disease; may be present at a young age</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>Diagnosis of familial hypercholesterolemia is based on physical examination and laboratory testing. Physical examination may find xanthomas and xanthelasmas (skin lesions caused by cholesterol rich lipoprotein deposits), and cholesterol deposits in the eye called corneal arcus.<br><br>
        <b>Potential Complications</b> <br><u1>
                                            <li>Cramping of one or both calves when walking</li>
                                            <li>Sores on the toes that do not heal</li>
                                            <li>Sudden stroke-like symptoms such as trouble speaking, drooping on one side of the face, weakness of an arm or leg, and loss of balance</li>
                                  </u1> <br><br>
        <b>Prevention</b> <br>Healthy and well balanced diet, regular on exercising, avoiding processed foods (containing saturated fats), take good quantity of vegetables, take lot of fruits, take whole-grain breads and cereals, take low-fat dairy products, maintain a healthy weight.<br><br>
        """)]

    elif ICD == 'd_584':
        # 11
        return ["http://alternative-doctor.com/wp-content/uploads/2015/03/symptoms-of-kidney-failure-ig.jpg",
                Markup("""
        <b>Description</b> <br>A condition in which the kidneys suddenly can't filter waste from the blood.<br><br>
        <b>Symptoms</b> <br><u1>
                                 <li>Whole body: fatigue or water-electrolyte imbalance</li>
                                 <li>Urinary: insufficient urine production or urinary retention</li>
                                 <li>Also common: shortness of breath, swelling, too much acid in blood and tissues, or vomiting</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>Analyzing a sample of your urine, a procedure called urinalysis, may reveal abnormalities that suggest kidney failure. Blood tests. A sample of your blood may reveal rapidly rising levels of urea and creatinine — two substances used to measure kidney function. Imaging tests.<br><br>
        <b>Potential Complications</b> <br><u1>
                                            <li>High levels of potassium in the blood – in severe cases, this can lead to muscle weakness, paralysis and heart rhythm problems</li>
                                            <li>Fluid in the lungs (pulmonary oedema)</li>
                                            <li>Acidic blood (metabolic acidosis) – which can cause nausea, vomiting, drowsiness and breathlessness</li>
                                  </u1><br><br>
        <b>Prevention</b> <br>If you have kidney disease or another condition that increases your risk of acute kidney failure, such as diabetes or high blood pressure, stay on track with treatment goals and follow your doctor's recommendations to manage your condition. Make a healthy lifestyle a priority.<br><br>
        """)]

    elif ICD == 'd_585':
        # 12
        return ["http://alternative-doctor.com/wp-content/uploads/2015/03/symptoms-of-kidney-failure-ig.jpg",
                Markup("""
        <b>Description</b> <br>The kidneys filter waste and excess fluid from the blood. As kidneys fail, waste builds up.<br><br>
        <b>Symptoms</b> <br><u1>
                                 <li>Whole body: fatigue, high blood pressure, loss of appetite, malaise, or water-electrolyte imbalance</li>
                                 <li>Gastrointestinal: nausea or vomiting</li>
                                 <li>Also common: kidney damage, abnormal heart rhythm, failure to thrive, fluid in the lungs, hyperphosphatemia, hypocalcemia, insufficient urine production, itching, kidney failure, sensation of pins and needles, severe unintentional weight loss, swelling, or uremia</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>A blood creatinine test helps to estimate the glomerular filtration rate (GFR) by measuring the level of creatinine in your blood. The doctor can use the GFR to regularly check how well the kidneys are working and to stage your kidney disease.  High blood sugar levels damage blood vessels in the kidneys.<br><br>
        <b>Potential Complications</b> <br><u1>
                                            <li>Anemia</li>
                                            <li>Bone disease and high phosphorus (hyperphosphatemia)</li>
                                            <li>Heart disease</li>
                                            <li>High potassium (hyperkalemia)</li>
                                            <li>Fluid buildup</li>

                                  </u1><br><br>
        <b>Prevention</b> <br>Cut back on protein, especially animal products such as meat. Damaged kidneys may fail to remove protein waste products from your blood; Avoid a high-fat diet. High-fat diets are high in cholesterol; Avoid high-sodium foods; Ask your doctor about the amount of potassium you need.<br><br>
        """)]

    elif ICD == 'd_403':
        # 13
        return ["http://i.imgur.com/IjV4Mqb.jpg",
                Markup("""
        <b>Description</b> <br>Blood pressure makes the heart work harder and, over time, can damage blood vessels throughout the body. If the blood vessels in the kidneys are damaged, they may stop removing wastes and extra fluid from the body. The extra fluid in the blood vessels may then raise blood pressure even more.<br><br>
        <b>Symptoms</b> <br><u1>
                                  <li>Fatigue, nausea</li>
                                  <li>Pruritus</li>
                                  <li>Nocturia</li>
                                  <li>Hypertension</li>
                         </u1> <br>
        <b>Diagnosis</b> <br>Blood pressure is the force of blood pushing against blood vessel walls as the heart pumps out blood, and high blood pressure, also called hypertension, is an increase in the amount of force that blood places on blood vessels as it moves through the body. Kidney disease is diagnosed with urine and blood tests.<br><br>
        <b>Potential Complications</b> <br><u1>
                                            <li>Severe anemia, hyperphosphatemia</li>
                                            <li>Hypocalcemia</li>
                                            <li>Subperiosteal erosions on radiography</li>
                                  </u1><br>
        <b>Prevention</b> <br>The NHLBI recommends the following lifestyle changes that help control blood pressure. Maintain normal weight, eat fresh fruits and vegetables, limit the consumption of frozen foods and trips to fast food restaurants, read nutrition labels on packaged foods to learn how much sodium is in one serving.<br><br>
        """)]

    elif ICD == 'd_599':
        # 14
        return [
            "https://image.slidesharecdn.com/urinarydisordersfinal-120910013646-phpapp02/95/urinary-disorders-final-18-728.jpg?cb=1347241176",
            Markup("""
        <b>Description</b> <br>A urinary tract infection (UTI) is an infection in any part of your urinary system — your kidneys, ureters, bladder and urethra. Most infections involve the lowerurinary tract — the bladder and the urethra. However, serious consequences can occur if a UTI spreads to your kidneys.<br><br>
        <b>Symptoms</b> <br><u1>
                                  <li>Strong and frequent urge to urinate.</li>
                                  <li>Cloudy, bloody, or strong smelling urine</li>
                                  <li>Pain or burning sensation when urinating</li>
                                  <li>Nausea and vomiting</li>
                                  <li>Muscle aches and abdominal pains</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>A diagnosis is usually made when the obvious causes for symptoms are ruled out. These causes include infections caused by viruses and bacteria. Your doctor will first want to review your symptoms and medical history. They may also perform a physical examination and take a urine sample. Your doctor may decide to take a blood sample or perform an ultrasound on the pelvic region. Your doctor may need to use a scope to get a view of the inside of your urethra if the first few treatments don’t work.<br><br>
        <b>Potential Complications</b> <br><u1>
                                            <li>Urethral cancer - a rare cancer that happens more often in men</li>
                                            <li>Urethral stricture - a narrowing of the opening of the urethra</li>
                                            <li>Urethritis - inflammation of the urethra, sometimes caused by infection</li>
                                  </u1><br><br>
        <b>Prevention</b> <br>Drink lots of water and urinate frequently; Avoid fluids such as alcohol and caffeine that can irritate the bladder; Urinate shortly after sex; Wipe from front to back after urinating and bowel movement; Keep the genital area clean; Showers are preferred to baths and avoid using oils.<br><br>
        """)]

    elif ICD == 'd_530':
        # 15
        return ["http://timesofindia.indiatimes.com/photo/49644114.cms",
                Markup("""
        <b>Description</b> <br>Many people experience a burning sensation in their chest occasionally, caused by stomach acids refluxing into the esophagus, normally called heartburn. The following are additional diseases and conditions that affect the esophagus: Acuteesophageal necrosis. Achalasia. Barrett's esophagus.<br><br>
        <b>Symptoms</b> <br><u1>
                                  <li>Achalasia</li>
                                  <li>Barrett's Esophagus</li>
                                  <li>Esophageal Cancer</li>
                                  <li>Stomach (Gastric) Cancer</li>
                                  <li>Gastroesophageal Reflux Disease (GERD)</li>
                                  <li>Gastroparesis</li>
                                  <li>Peptic Ulcer Disease</li>
                                  <li>Swallowing Disorders</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>The diagnosis of Barrett's esophagus rests upon seeing (at endoscopy) a pinkesophageal lining that extends a short distance (usually less than 2.5 inches) up theesophagus from the gastroesophageal junction and finding intestinal type cells (goblet cells) on biopsy of the lining.<br><br>
        <b>Potential Complications</b> <br><u1>
                                            <li>Gastrointestinal: belching, heartburn, indigestion, nausea, regurgitation, vomiting, or vomiting blood</li>
                                            <li>Throat: difficulty swallowing, irritation, or lumps</li>
                                            <li>Also common: coughing, loss of appetite, pulmonary aspiration, or weight loss</li>
                                  </u1><br><br>
        <b>Prevention</b> <br>It is advised to maintain a healthy weight, quit smoking, reduce food that cause heart burn also reduce the amount of food intake, avoid lying down after a meal.<br><br>
        """)]

    elif ICD == 'd_038':
        # 16
        return ["http://sepsistrust.org/wp-content/uploads/2015/08/New-paeds-sepsis-card-side2-3.jpg",
                Markup("""
        <b>Description</b> <br>Septicemia is a serious bloodstream infection. It's also known as bacteremia, or blood poisoning. Septicemia occurs when a bacterial infection elsewhere in the body, such as in the lungs or skin, enters the bloodstream.<br><br>
        <b>Symptoms</b> <br><u1>
                                  <li>Delirium</li>
                                  <li>Fast heart rate</li>
                                  <li>Insufficient urine production</li>
                                  <li>Organ dysfunction</li>
                                  <li>Skin discoloration</li>
                                  <li>Sleepiness</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>Your doctor may look for your cell and platelet counts and also order tests to analyze your blood clotting. Your doctor may also look at the oxygen and carbon dioxide levels in your blood if septicemia is causing you to have breathing issues.<br><br>
        <b>Potential Complications</b> <br><u1>
                                            <li>Whole body: chills, dizziness, fatigue, fever, flushing, low blood pressure, low body temperature, or shivering</li>
                                            <li>Respiratory: fast breathing, rapid breathing, respiratory distress, or shortness of breath</li>
                                            <li>Cognitive: altered level of consciousness or mental confusion</li>
                                  </u1><br><br>
        <b>Prevention</b> <br>Get vaccinated, thirty-five percent of sepsis cases in the CDC study stemmed from pneumonia. Treat urinary tract infections promptly. A quarter of sepsis cases resulted from urinary tract infections. Clean skin wounds properly. Avoid infections in hospitals.<br><br>
        """)]

    elif ICD == 'd_707':
        # 17
        return ["http://oi66.tinypic.com/euonwi.jpg",
                Markup("""
        <b>Description</b> <br>A skin disease characterized by dark wartlike patches in the body folds; can be benign or malignant. acne. an inflammatory disease involving the sebaceous glands of the skin; characterized by papules or pustules or comedones.<br><br>
        <b>Symptoms</b> <br><u1>
                                  <li>Measles</li>
                                  <li>Warts</li>
                                  <li>Acne</li>
                                  <li>Fifth disease</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>Diagnosis is usually done using either Skin scraping or Combing of the hair coat (also called flea combing) or Examination of microscopic hairs on the skin.<br><br>
        <b>Potential Complications</b> <br><u1>
                                            <li>eczema</li>
                                            <li>Diaper rash</li>
                                            <li>Seborrheic dermatitis</li>
                                            <li>Chickenpox</li>
                                  </u1><br><br>
        <b>Prevention</b> <br>Avoiding risk factors and increasing protective factors may help prevent cancer. Avoiding cancer risk factors may help prevent certain cancers. Risk factors include smoking, being overweight, and not getting enough exercise. Increasing protective factors such as quitting smoking, eating a healthy diet, and exercising may also help prevent some cancers. Talk to your doctor or other health care professional about how you might lower your risk of cancer. Being exposed to ultraviolet radiation is a risk factor for skin cancer. Some studies suggest that being exposed to ultraviolet (UV) radiation and the sensitivity of a person's skin to UV radiation are risk factors for skin cancer. UV radiation is the name for the invisible rays that are part of the energy that comes from the sun. Sunlamps and tanning beds also give off UV radiation.<br><br>
        """)]

    elif ICD == 'd_995':
        # 18
        return ["http://oi67.tinypic.com/2hga1ch.jpg",
                Markup("""
        <b>Description</b> <br>Anaphylaxis is a serious allergic reaction that is rapid in onset and may cause death.<br><br>
        <b>Symptoms</b> <br><u1>
                                  <li>Gastrointestinal: nausea or vomiting</li>
                                  <li>Speech: vocal cord spasm or impaired voice</li>
                                  <li>Also common: coughing, difficulty swallowing, facial swelling, fast heart rate, feeling of impending doom, itching, mental confusion, nasal congestion, or tongue swelling</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>The diagnosis of anaphylaxis is based on symptoms that occur within minutes to a few hours after exposure to a potential trigger, such as a food, medication, or insect sting.<br><br>
        <b>Potential Complications</b> <br><u1>
                                            <li>Pain areas: in the abdomen or chest</li>
                                            <li>Whole body: dizziness, fainting, flushing, light-headedness, or low blood pressure</li>
                                            <li>Respiratory: difficulty breathing, noisy breathing, rapid breathing, shortness of breath, or wheezing</li>
                                            <li>Skin: blue skin from poor circulation, hives, rashes, or swelling under the skin</li>
                                  </u1><br><br>
        <b>Prevention</b> <br>The best way to prevent anaphylaxis is to avoid substances that you know cause this severe reaction. Follow these steps: Wear a medical alert necklace or bracelet to indicate if you have an allergy to specific drugs or other substances. Alert your doctor to your drug allergies before having any medical treatment.<br><br>
        """)]

    elif ICD == 'd_998':
        # 19
        return [
            "http://www.carecor.com/sites/default/files/styles/article_banner/public/assets/news/185/coverimage/aa043385-2.jpg?itok=fxq03A3Y",
            Markup("""
        <b>Description</b> <br>Surgical shock is a condition of shock that may occur during or after surgery, with signs of profound hypotension, decreased urine, increased heart rate, restlessness, and cyanosis of the extremities. Hemoglobin for blood volume may be low, or patient may be bleeding or have a severe infection.<br><br>
        <b>Symptoms</b> <br><u1>
                                  <li>Pain</li>
                                  <li>Acute confusion: exclude dehydration and sepsis</li>
                                  <li>Nausea and vomiting: analgesia or anaesthetic-related</li>
                                  <li>Paralytic ileus</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>Postoperative complications may either be general or specific to the type of surgery undertaken and should be managed with the patient's history in mind. Common general postoperative complications include postoperative fever, atelectasis, wound infection, embolism and deep vein thrombosis (DVT). The highest incidence of postoperative complications is between one and three days after the operation. However, specific complications occur in the following distinct temporal patterns: early postoperative, several days after the operation, throughout the postoperative period and in the late postoperative period.<br><br>
        <b>Potential Complications</b> <br><u1>
                                            <li>Fever</li>
                                            <li>Secondary haemorrhage: often as a result of infection</li>
                                            <li>Pneumonia</li>
                                            <li>Wound or anastomosis dehiscence</li>
                                  </u1><br><br>
        <b>Prevention</b> <br>The prevention of postoperative shock and postanesthesia hypotension can be done by the use of reverse Trendelenburg position during surgery under light, etherless, general anesthesia.<br><br>
        """)]

    elif ICD == 'd_041':
        # 20
        return [
            "https://edc2.healthtap.com/ht-staging/user_answer/reference_image/11208/large/Bacterial_Infection.jpeg?1386670561",
            Markup("""
        <b>Description</b> <br>Streptococcal infections are any type of infection caused by the streptococcus("strep") group of bacteria. There are many different types of Streptococci bacteria, and infections vary in severity from mild throat infections to life-threateninginfections of the blood or organs.<br><br>
        <b>Symptoms</b> <br><u1>
                                  <li>Enlarged neck lymph nodes</li>
                                  <li>Swollen tonsils</li>
                                  <li>Tender lymph nodes</li>
                                  <li>Bad breath</li>
                                  <li>Headache</li>
                                  <li>Respiratory tract infection</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>If you complain of a persistent sore throat, your doctor will examine your throat and check for signs of inflammation. ... This test determines whether your sore throat is caused by astrep infection or another type of bacteria or germ.<br><br>
        <b>Potential Complications</b> <br><u1>
                                            <li>Pain circumstances: can occur while swallowing</li>
                                            <li>Whole body: fever, chills, fatigue syndrome, or malaise</li>
                                            <li>Throat: soreness, difficulty swallowing, or irritation</li>
                                            <li>Gastrointestinal: nausea or vomiting</li>
                                  </u1><br><br>
        <b>Prevention</b> <br>To avoid getting strep throat, it is a good idea to avoid contact with anyone who has a strep infection. Wash your hands often when you are around people with colds or viral or bacterial illnesses. Do not share toothbrushes or eating and drinking utensils.<br><br>
        """)]

    elif ICD == 'd_244':
        # 21
        return [
            "https://share.baptisthealth.com//wp-content/uploads/2016/03/infographic-foods-to-avoid-hypothyroidism.jpg",
            Markup("""
        <b>Description</b> <br>Acquired hypothyroidism is a condition that develops when your child's thyroidgland makes little or no thyroid hormone. Thyroid hormones help control body temperature, heart rate, and how your child gains or loses weight. Thyroidhormones play an important role in normal growth and development of children.<br><br>
        <b>Symptoms</b> <br><u1>
                                  <liFatigue</li>
                                  <li>Weakness</li>
                                  <li>Intolerance to cold</li>
                                  <li>Muscle aching and cramps</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>Thyroid stimulating hormone (TSH). The TSH assay is the most sensitive test for the diagnosis of hypothyroidism. Free T4 (free thyroxine) and T3 thyroidhormones. Thyroid hormone levels in the blood will be low, but with mild, or "subclinical" hypothyroidism, the thyroid hormone levels can be in the low normal range.<br><br>
        <b>Potential Complications</b> <br><u1>
                                            <li>Constipation</li>
                                            <li>Weight gain or difficulty losing weight</li>
                                            <li>Poor appetite</li>
                                            <li>Goiter (enlarged thyroid gland)</li>
                                  </u1><br><br>
        <b>Prevention</b> <br>Hypothyroidism may be prevented in a population by adding iodine to commonly used foods. Pregnant and breastfeeding women, require 66% more daily iodine requirement than non-pregnant women.<br><br>
        """)]

    elif ICD == 'd_486':
        # 22
        return [
            "https://www.ipsos-mori.com/Assets/Images/Infographics/ipsos-healthcare-pneu-vue-infographic_lightbox.jpg",
            Markup("""
        <b>Description</b> <br>Pneumonia, organism unspecified. Also called: Bronchopneumonia. Pneumoniais an infection in one or both of the lungs. Many germs, such as bacteria, viruses, and fungi, can cause pneumonia. You can also get pneumonia by inhaling a liquid or chemical.<br><br><br>
        <b>Symptoms</b> <br><u1>
                                  <li>Pain types: can be sharp in the chest</li>
                                  <li>Cough: can be chronic, dry, with phlegm, mild, or severe</li>
                                  <li>Fever, which may be mild or high</li>
                                  <li>Shaking chills</li>
                                  <li>Shortness of breath, which may only occur when you climb stairs</li>
                                  <li>Sharp or stabbing chest pain that gets worse when you breathe deeply or cough</li>
                                  <li>Excessive sweating and clammy skin</li>
                         </u1> <br><br>
        <b>What Causes Pneumonia?</b> <u1>
                                            <li>Bacteria</li>
                                            <li>Viruses</li>
                                            <li>Mycoplasmas</li>
                                            <li>Other infectious agents, such as fungi</li>
                                            <li>Various chemicals</li>
                                     </u1> <br><br>
        <b>Diagnosis</b> <br>Chest x ray to look for inflammation in your lungs. A chest x ray is the best test fordiagnosing pneumonia. However, this test won't tell your doctor what kind of germ is causing the pneumonia. Blood tests such as a complete blood count (CBC) to see if your immune system is actively fighting an infection.<br><br><br>
        <b>Prevention</b> <br>Get a flu shot every year to prevent seasonal influenza. The flu is a common cause of pneumonia, so preventing the flu is a good way to prevent pneumonia. Children younger than 5 and adults 65 and older should get vaccinated against pneumococcal pneumonia, a common form of bacterial pneumonia.<br><br><br>
        """)]

    elif ICD == 'd_458':
        # 23
        return ["https://www.todayhealthtips.com/wp-content/uploads/2016/10/Symptoms-of-Postural-Hypotension.jpg",
                Markup("""
        <b>Description</b> <br>Hypotension is the opposite of hypertension (abnormally high blood pressure). Hypotension is a relative term because the blood pressure normally varies greatly with activity, age, medications, and underlying medical conditions.<br><br>
        <b>Symptoms</b> <br><u1>
                                <li>Dizziness or lightheadedness</li>
                                <li>Fainting (syncope)</li>
                                <li>Lack of concentration</li>
                                <li>Blurred vision</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>Diagnosing low blood pressure. Low blood pressure (hypotension) can be easilydiagnosed by measuring your blood pressure. You may need further tests, such as blood tests or an electrocardiogram (ECG), to determine the underlying cause.<br><br>
        <b>Prevention</b> <br>If your blood pressure drops after eating, your doctor may recommend small, low-carbohydrate meals. Get plenty of fluids. Keeping hydrated helps prevent symptoms of low blood pressure. But avoid or limit the amount of alcohol you drink, because alcohol can worsen orthostatic hypotension.<br><br>
        """)]


    elif ICD == 'd_424':
        # 24
        return [
            "https://aos.iacpublishinglabs.com/question/aq/1400px-788px/many-times-heart-beat-day_55e78b7fbeb0bff6.jpg?domain=cx.aos.ask.com",
            Markup("""
        <b>Description</b> <br>A disorder characterized by a defect in mitral valve function or structure. A heart disorder characterized by a defect in mitral valve structure or function.<br><br>
        <b>Symptoms</b> <br><u1>
                                  <li>Fluttering or rapid heartbeat called palpitations</li>
                                  <li>Shortness of breath, especially with exercise</li>
                                  <li>Dizziness</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>Echocardiography (echo) is the main test for diagnosing heart valve disease. But an EKG (electrocardiogram) or chest x ray commonly is used to reveal certain signs of the condition. If these signs are present, echo usually is done to confirm the diagnosis.<br><br>
        <b>Prevention</b> <br>Heart-healthy eating, physical activity, other heart-healthy lifestyle changes, and medicines aimed at preventing a heart attack, high blood pressure, or heart failure helps in preventing heart valve disease.<br><br>
        """)]

    elif ICD == 'd_496':
        # 25
        return ["https://airwayjedi.files.wordpress.com/2014/05/signsobstruct.png",
                Markup("""
        <b>Description</b> <br>Acute exacerbation of COPD also known as acute exacerbations of chronic bronchitis (AECB) is a sudden worsening of COPD symptoms (shortness of breath, quantity and color of phlegm) that typically lasts for several days. It may be triggered by an infection with bacteria or viruses or by environmental pollutants.<br><br>
        <b>Symptoms</b> <br><u1>
                                  <li>Shortness of breath</li>
                                  <li>Weakness</li>
                                  <li>Fever</li>
                                  <li>Coughing</li>
                                  <li>Fatigue</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>Pulmonary function tests measure the amount of air you can inhale and exhale, and if your lungs are delivering enough oxygen to your blood. Spirometry is the most common lung function test. Spirometry can detect COPD even before you have symptoms of the disease.<br><br>
        <b>Potential Complications</b> <br><u1>
                                            <li>More frequent lung infections, such as pneumonia</li>
                                            <li>An increased risk of thinning bones (osteoporosis), especially if you use oral corticosteroids</li>
                                            <li>Problems with weight</li>
                                            <li>Heart failure affecting the right side of the heart (cor pulmonale)</li>
                                            <li>A collapsed lung (pneumothorax)</li>
                                  </u1><br><br>
        <b>Prevention</b> <br>Try to avoid lung irritants that can contribute to COPD. Examples include secondhand smoke, air pollution, chemical fumes, and dust. (Secondhand smoke is smoke in the air from other people smoking.)<br><br>
        """)]

    elif ICD == 'd_997':
        # 26
        return ["http://www.brainline.org/images/uploads/orig/2012/TBIStats_Causes.jpg",
                Markup("""
        <b>Description</b> <br>This class of diseases usually includes conditions like blood supply problems (vascular disorders), injuries (trauma), especially injuries to the head and spinal cord,  problems that are present at birth (congenital).<br><br>
        <b>Symptoms</b> <br><u1>
                                  <li>Persistent or sudden onset of a headache</li>
                                  <li>A headache that changes or is different</li>
                                  <li>Loss of feeling or tingling</li>
                                  <li>Weakness or loss of muscle strength</li>
                         </u1> <br><br>
        <b>Diagnosis</b> <br>To further complicate the diagnostic process, many disorders do not have definitive causes, markers, or tests. In addition to a complete medical history and physical exam, diagnostic procedures for nervous system disorders may include the following: Computed tomography scan (also called a CT or CAT scan).<br><br>
        <b>Potential Complications</b> <br><u1>
                                            <li>Sudden loss of sight or double vision</li>
                                            <li>Memory loss</li>
                                            <li>Impaired mental ability</li>
                                            <li>Lack of coordination</li>
                                  </u1><br><br>
        <b>Prevention</b> <br>Exercise regularly, do not smoke or use other tobacco products, get plenty of rest, take care of health conditions.<br><br>
        """)]

    elif ICD == 'd_305':
        # 27
        return ["https://paradigmmalibu.com/wp-content/uploads/2014/06/alcohol.jpg",
                Markup("""
        <b>Description</b> <br>Nondependent alcohol abuse means that their drinking causes distress and harm. It includes alcoholism and alcohol abuse. When you abuse alcohol, you continue to drink even though you know your drinking is causing problems. If you continue to abuse alcohol, it can lead toalcohol dependence. Alcohol dependence is also called alcoholism. You are physically or mentally addicted to alcohol<br><br>
        <b>Symptoms</b> <br><ul>
                                    <li>Anxiety</li>
                                    <li>Tremors</li>
                                    <li>Sweating</li>
                                    <li>Insomnia</li>
                                    <li>Nausea</li>
                                    <li>Depression</li>
                                    <li>Fatigue </li>
                                    <li>Headache </li>
                                    <li>Irritability.</li>
                        </ul> <br><br>
        <b>Potential Complications</b> <ul>
                                            <li>Cardiomyopathy, atrial fibrillation</li>
                                            <li>Hypertension</li>
                                            <li>Peptic ulcer disease/gastritis</li>
                                            <li>Cirrhosis, fatty liver, cholelithiasis</li>
                                            <li>Hepatitis</li>
                                            <li>Diabetes mellitus</li>
                                            <li>Pancreatitis</li>
                                            <li>Malnutrition</li>
                                            <li>Upper GI malignancies</li>
                                            <li>Peripheral neuropathy, seizures</li>
                                            <li>Abuse and violence</li>
                                            <li>Trauma (falls, motor vehicle accidents [MVAs])</li>
                                    </ul>><br><br>
        <b>Prevention</b> <br>Raising children is no easy task. With so many variables—parenting style, temperament of your child, as well as the peer group that they find themselves surrounded by—it’s impossible to control for everything. Furthermore, it might be counterproductive to attempt to raise a child in a completely protected environment. Still, a parent with little insight as to the whereabouts and activities of their child runs the risk of encountering problems. Even trustworthy teens are bound to find themselves in some questionable situations over the course of their development towards adulthood. Being vigilant isn’t necessarily being overprotective. Knowing the location, company and activity of your family members throughout the course of the day can help keep them safe. The advent of cell phones makes it easier than ever to check up on your teen. Additionally, straying from a perceived routine in terms of when you are at the house yourself might go far in preventing a teen from feeling complacent that they are alone and able to do whatever they please. For parents who are unable to be at home after school has let out due to work obligations, having other family members or neighbors keep a helpful eye on a child—even if from a distance—can make a big difference in preventing <br><br>
        """)]

    elif ICD == 'd_410':
        # 28
        return ["https://s-media-cache-ak0.pinimg.com/originals/74/95/ed/7495ed1656e817bc505ffc0386ca5550.jpg",
                Markup("""
        <b>Description</b> <br>Myocardial infarction (MI) or acute myocardial infarction (AMI), commonly known as a heart attack, occurs when blood flow stops to a part of the heart causing damage to the heart muscle. The most common symptom is chest pain or discomfort which may travel into the shoulder, arm, back, neck, or jaw. <br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Pain areas: in the area between shoulder blades, arm, chest, chest, jaw, left arm, or upper abdomen </li>
                                <li>Pain types: can be crushing, like a clenched fist in the chest, radiating from the chest, sudden in the chest, or mild </li>
                                <li>Pain circumstances: can occur during rest </li>
                                <li>Whole body: dizziness, fatigue, light-headedness, clammy skin, cold sweat, or sweating </li>
                                <li>Gastrointestinal: heartburn, indigestion, nausea, or vomiting </li>
                                <li>Chest: discomfort, fullness, or tightness </li>
                                <li>Neck: discomfort or tightness </li>
                                <li>Arm: discomfort or tightness </li>
                                <li>Also common: anxiety, feeling of impending doom, sensation of an abnormal heartbeat, shortness of breath, or shoulder discomfort</li>
                            </ul> <br><br>
        <b>Diagnosis</b> <br>A diagnosis of myocardial infarction is created by integrating the history of the presenting illness and physical examination with electrocardiogram findings and cardiac markers (blood tests for heart muscle cell damage).<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Ischemic: Angina, reinfarction, infarct extension</li>
                                                <li>Mechanical: Heart failure, cardiogenic shock, mitral valve dysfunction, aneurysms, cardiac rupture</li>
                                                <li>Arrhythmic: Atrial or ventricular arrhythmias, sinus or atrioventricular node dysfunction</li>
                                                <li>Embolic: Central nervous system or peripheral embolization</li>
                                                <li>Inflammatory: Pericarditis</li>
                                        </ul><br><br>
        <b>Prevention</b> <br>The use of diuretics to lower blood pressure reduces strokes. In contrast, calcium antagonists do not appear to consistently reduce mortality or prevent vascular events when used for primary or secondary prevention of either myocardial infarction or strokes.<br><br>
        """)]

    elif ICD == 'd_287':
        # 29
        return ["https://s-media-cache-ak0.pinimg.com/736x/1b/61/fe/1b61fe64d8490ac2e829d90332f738aa.jpg",
                Markup("""
        <b>Description</b> <br>Allergic purpura (AP) is an allergic reaction of unknown origin causing red patches on the skin and other symptoms. AP is also called Henoch-Schonlein purpura<br><br>
        <b>Symptoms</b> <br><ul>
                                    <li>Skin: red spots or rash of small purplish spots </li>
                                    <li>Also common: bleeding, bruising, heavy or prolonged periods, or nosebleed</li>
                            </ul> <br><br>
        <b>Diagnosis</b> <br>Henoch-Schonlein purpura (HSP) is a disease involving inflammation of small blood vessels. ... The inflammation causes blood vessels in the skin, intestines, kidneys, and joints to start leaking. The main symptom is a rash with numerous small bruises, which have a raised appearance, over the legs or buttocks<br><br>
        <b>Potential Complication-</b> <br> Kidney Damage. <br>
        """)]

    elif ICD == 'd_571':
        # 30
        return [
            "http://thumbnails-visually.netdna-ssl.com/how-alcohol-travels-through-the-body_50ca3deb65aab_w1500.jpeg",
            Markup("""
        <b>Description</b> <br>Alcoholic fatty liver disease results from the deposition of fat in liver cells.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Loss of appetite</li>
                                <li>Nausea</li>
                                <li>Vomiting</li>
                                <li>Abdominal pain</li>
                            </ul> <br>
        <b>Diagnosis</b> <br>Ultrasonography, computed tomography (CT), or magnetic resonance imaging (MRI) of the abdomen can detect excess fat in the liver but cannot always determine whether inflammation or fibrosis is present (see Imaging Tests of the Liver and Gallbladder). Liver biopsy may be necessary to confirm the diagnosis.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Ascites (fluid buildup in the abdomen)</li>
                                                <li>Variceal hemorrhage (bleeding in the upper stomach and esophagus from ruptured blood vessels)</li>
                                                <li>Spontaneous bacterial peritonitis is a form of peritonitis (inflammation of the membrane that lines the abdomen), which is associated with ascites. Other bacterial infections are also a common complication of cirrhosis.</li>
                                            </ul><br>
        <b>Prevention</b> <br>Foods that need to be restricted include bread, pasta, rice, breakfast cereals, cakes, pastry, donuts, biscuits, fries, chips, pretzels (and other similar snack foods) and any food made of flour. Excess alcohol consumption is the second biggest cause of fatty liver.<br><br>
        """)]

    elif ICD == 'd_493':
        # 31
        return ["https://s-media-cache-ak0.pinimg.com/originals/0d/c6/22/0dc622792a21fd0d6ee2d50f9b579500.png",
                Markup("""
        <b>Description</b> <br>Allergic or atopic asthma (sometimes called extrinsic asthma) is due to an allergy to antigens; usually the offending allergens are suspended in the air in the form of pollen, dust, smoke, automobile exhaust, or animal dander.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Many of the symptoms of allergic and non-allergic asthma are the same</li>
                                <li>Coughing, wheezing, shortness of breath or rapid breathing, and chest tightness.</li>
                                <li>However, allergic asthma is triggered by inhaled allergens such as dust mite allergen, pet dander, pollen, mold, etc. resulting in asthma symptoms.</li>
                            </ul> <br>
        <b>Diagnosis</b> <br>The two most common lung function tests used to diagnose asthma are spirometry and methacholine challenge tests. Spirometry is a simple breathing test that measures how much and how fast you can blow air out of your lungs. It is often used to determine the amount of airway obstruction you have.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Permanent damage to the airways</li>
                                                <li>Death - uncommon</li>
                                                <li>Severe breathing difficulty (seeBreathing difficulties)</li>
                                                <li>School absenteeism (seeAbsenteeism)</li>
                                                <li>Home confinement</li>
                                            </ul><br>
        <b>Prevention</b> <br>It is advised to avoid the following: Dust mites, pollen, mould, animal dander, cockroach faeces.<br><br>
        """)]

    elif ICD == 'd_311':
        # 32
        return ["https://s-media-cache-ak0.pinimg.com/originals/e5/0e/9e/e50e9ebbe21117f84b6bd479c595f008.jpg",
                Markup("""
        <b>Description</b> <br>A depressive disorder is not a passing blue mood but rather persistent feelings of sadness and worthlessness and a lack of desire to engage in formerly pleasurable activities.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Pain areas: in the back </li>
                                <li>Mood: anxiety, apathy, general discontent, guilt, hopelessness, loneliness, loss of interest, loss of interest or pleasure in activities, mood swings, panic attack, sadness, or emotional distress </li>
                                <li>Behavioural: agitation, excessive crying, irritability, restlessness, self-harm, or social isolation </li>
                                <li>Sleep: early awakening, excess sleepiness, insomnia, or restless sleep </li>
                                <li>Whole body: excessive hunger, fatigue, or loss of appetite </li>
                                <li>Cognitive: lack of concentration, slowness in activity, or thoughts of suicide </li>
                                <li>Psychological: depression or repeatedly going over thoughts </li>
                                <li>Also common: constipation, headache, poor appetite, substance abuse, or weight loss</li>
                            </ul><br>
        <b>Diagnosis</b> <br>A major depressive episode is characterized by the presence of a severely depressed mood that persists for at least two weeks. Episodes may be isolated or recurrent and are categorized as mild (few symptoms in excess of minimum criteria), moderate, or severe (marked impact on social or occupational functioning).<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Excess weight or obesity, which can lead to heart disease and diabetes.</li>
                                                <li>Pain and physical illness.</li>
                                                <li>Alcohol or substance misuse.</li>
                                                <li>Anxiety, panic disorder or social phobia.</li>
                                                <li>Family conflicts, relationship difficulties, and work or school problems.</li>
                                                <li>Social isolation.</li>
                                            </ul><br>
        <b>Prevention</b><br>Get enough sleep, exercise, regulate your blood sugar, eat healthy fats, find passion in life.<br><br>
        """)]

    elif ICD == 'd_412':
        # 33
        return ["https://s-media-cache-ak0.pinimg.com/originals/74/95/ed/7495ed1656e817bc505ffc0386ca5550.jpg",
                Markup("""
        <b>Description</b> <br>Chest pain is the most common symptom of acute MI and is often described as a sensation of tightness, pressure, or squeezing. Chest pain due to ischemia (a lack of blood and hence oxygen supply) of the heart muscle is termed angina pectoris.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Myocardial infarction (MI) or acute myocardial infarction (AMI), commonly known as a heart attack, occurs when blood flow stops to a part of the heart causing damage to the heart muscle.</li>
                                <li> The most common symptom is chest pain or discomfort which may travel into the shoulder, arm, back, neck, or jaw.</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>The diagnosis of myocardial infarction requires two out of three components (history, ECG, and enzymes). When damage to the heart occurs, levels of cardiac markers rise over time, which is why blood tests for them are taken over a 24-hour period.<br><br>
        <b>Potential Complications</b> <br><ul>
                                            <li>Ischaemic (including failure of reperfusion): angina, re-infarction, infarct extension.</li>
                                            <li>Mechanical: heart failure, cardiogenic shock, mitral valve dysfunction, aneurysms, cardiac rupture.</li>
                                            <li>Arrhythmic: atrial or ventricular arrhythmias, sinus or atrioventricular (AV) node dysfunction.</li>
                                            <li>Thrombosis and embolic: central nervous system or peripheral embolisation.</li>
                                            <li>Inflammatory: pericarditis.</li>
                                            <li>Psychosocial complications (including depression).</li>
                                          </ul><br><br>
        <b>Prevention</b><br>A number of lifestyle recommendations are available to those who have experienced myocardial infarction. This includes the adoption of a Mediterranean-type diet, maintaining alcohol intake within recommended limits, exercising to the point of mild breathlessness for 20–30 minutes every day, stopping smoking, and trying to achieve a healthy weight.Exercise is both safe and effective even if people have had stents or heart failure.<br>
                            People are usually started on several long-term medications after an MI, with the aim of preventing further cardiovascular events such as MIs, congestive heart failure, or strokes.<br>
                            Aspirin as well as another antiplatelet agent such as clopidogrel or ticagrelor ("dual antiplatelet therapy" or DAPT), is continued for up to twelve months, followed by aspirin indefinitely. If someone has another medical condition that requires anticoagulation (e.g. with warfarin) this may need to be adjusted based on risk of further cardiac events as well as bleeding risk. In those who have had a stent, more than 12 months of clopidogrel plus aspirin does not affect the risk of death.<br>
                            Beta blocker therapy such as metoprolol or carvedilol is recommended to be started within 24 hours, provided there is no acute heart failure or heart block. The dose should be increased to the highest tolerated. Contrary to what was long believed, the use of beta blockers does not appear to affect the risk of death, possibly because other treatments for MI have improved. They should not be used in those who have recently taken cocaine.<br>
                            ACE inhibitor therapy should be started when stable and continued indefinitely at the highest tolerated dose. Those who cannot tolerate ACE inhibitors may be treated with an angiotensin II receptor antagonist.<br>
                            Statin therapy has been shown to reduce mortality and morbidity.[120] The protective effects of statins may be due to more than their LDL lowering effects. The general consensus is that statins have the ability to stabilize plaques and multiple other ("pleiotropic") effects that may prevent myocardial infarction in addition to their effects on blood lipids.<br>
                            Aldosterone antagonists (spironolactone or eplerenone) may be used if there is evidence of left ventricular dysfunction after an MI, ideally after beginning treatment with an ACE inhibitor.<br><br>
        """)]

    elif ICD == 'd_070':
        # 34
        return [
            "http://www.fs-researchcenter.com/admin/Pub_Events/121130_FSRC-Infographic_Burden-of-Viral-Hepatitis.jpg",
            Markup("""
        <b>Description</b> <br>Hepatitis A is preventable by vaccine. It spreads from contaminated food or water or contact with someone who is infected.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Fatigue,</li>
                                <li>Nausea, </li>
                                <li>Abdominal pain,</li>
                                <li>Loss of appetite </li>
                                <li>Low-grade fever. </li>
                                <li>The condition clears up on its own in one or two months.</li>
                                <li>Rest and adequate hydration can help.</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>When a patient reports symptoms of fatigue, nausea, abdominal pain, darkening of urine, and then develops jaundice, the diagnosis of acute viral hepatitis is likely and can be confirmed by blood tests. ... Typically, these patients do not have jaundice until the liver damage is far advanced.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Scarring of the liver (cirrhosis). </li>
                                                <li>The inflammation associated with a hepatitis B infection can lead to extensive liver scarring (cirrhosis), which may impair the liver's ability to function.</li>
                                                <li> Liver cancer.</li>
                                            </ul><br><br>
        <b>Prevention</b><br>Prevention, Diagnosis, Treatment of Hepatitis B and C. The hepatitis B virus(HBV) is transmitted between people through contact with the blood or other body fluids, including semen and vaginal fluid of an infected person.<br><br>
        """)]

    elif ICD == 'd_511':
        # 35
        return ["https://upload.wikimedia.org/wikipedia/commons/0/06/Pleurisy.png",
                Markup("""
        <b>Description</b> <br>Inflammation of the pleurae, which impairs their lubricating function and causes pain when breathing. It is caused by pneumonia and other diseases of the chest or abdomen.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Pain areas: in the chest or side part of the body </li>
                                <li>Cough: can be dry </li>
                                <li>Respiratory: fast breathing, shallow breathing, or shortness of breath </li>
                                <li>Also common: chest pain worsened by breathing or fever</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>Physical Exam. Your doctor will listen to your breathing with a stethoscope to find out whether your lungs are making any abnormal sounds. If you have pleurisy, the inflamed layers of the pleura make a rough, scratchy sound as they rub against each other when you breathe. Doctors call this a pleural friction rub.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>A viral infection, such as the flu. </li>
                                                <li>A bacterial infection, such as pneumonia.</li>
                                                <li>a blood clot that blocks the flow of blood into the lungs (a pulmonary embolism)</li>
                                                <li>Lung cancer</li>
                                            </ul><br><br>
        <b>Prevention</b><br>Pleurisy cannot be prevented in all cases. Receiving early treatment for bacterial respiratory infections (e.g., pneumonia) and managing underlying conditions effectively can help reduce the risk for developing pleurisy.<br><br>
        """)]

    elif ICD == 'd_507':
        # 36
        return ["https://www.healthunits.com/wp-content/uploads/2017/03/signs-dignosis-of-Pneumonia.jpg",
                Markup("""
        <b>Description</b> <br>Pneumonia is a breathing condition in which there is swelling or an infection of the lungs or large airways. Aspiration pneumonia occurs when food, saliva, liquids, orvomit is breathed into the lungs or airways leading to the lungs, instead of being swallowed into the esophagus and stomach.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Symptoms of aspiration pneumonitis from inhaling food/vomitus can include difficulty breathing </li>
                                <li>A wheezing sound in the chest, along with chest pain </li>
                                <li>Coughing upphlegm</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>Before diagnosing hypersensitivity pneumonitis, your doctor must rule out: unintentional effects of medicines such as bleomycin, methotrexate, or nitrofurantoin; lung infections such as pneumonia or the flu (influenza); smoking-related lung disease; connective tissue disease; bleeding in the lungs; idiopathic pulmonary.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>A viral infection, such as the flu. </li>
                                                <li>A bacterial infection, such as pneumonia.</li>
                                                <li>a blood clot that blocks the flow of blood into the lungs (a pulmonary embolism)</li>
                                                <li>Lung cancer</li>
                                            </ul><br><br>
        <b>Prevention</b><br>Avoid exposure to the cause - eg, control of occupational hazards, routine maintenance of heating, ventilation and air-conditioning equipment.<br><br>
        """)]

    elif ICD == 'd_416':
        # 37
        return ["https://www.curascriptsd.com/Assets/Pulmonary%20Hypertension_December%202015.jpg",
                Markup("""
        <b>Description</b> <br>Idiopathic pulmonary arterial hypertension (IPAH) is a rare disease characterized by elevated pulmonary artery pressure with no apparent cause. IPAH is also termed precapillary pulmonary hypertension and was previously termed primary pulmonary hypertension.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Pulmonary hypertension affects arteries in the lungs and the right side of the heart.</li>
                                <li>Shortness of breath, dizziness and chest pressure are symptoms.</li>
                                <li>The condition worsens over time, but medication and oxygen Therapy can help lessen symptoms and improve quality of life.</li>
                            </ul><br>
        <b>Diagnosis</b> <br>Background. Idiopathic pulmonary arterial hypertension (IPAH) is a rare disease characterized by elevated pulmonary artery pressure with no apparent cause. IPAH is also termed precapillary pulmonary hypertension and was previously termedprimary pulmonary hypertension.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Shortness of breath (dyspnea), initially while exercising and eventually while at rest.</li>
                                                <li>Fatigue.</li>
                                                <li>Dizziness or fainting spells (syncope)</li>
                                                <li>Swelling (edema) in your ankles, legs and eventually in your abdomen (ascites)</li>
                                            </ul><br>
        <b>Prevention</b><br>Pulmonary hypertension (PH) has no cure. However, treatment may help relieve symptoms and slow the progress of the disease.<br><br>
        """)]

    elif ICD == 'd_327':
        # 38
        return ["http://www.pindex.com/uploads/post_images/original/image_502.png",
                Markup("""
        <b>Description</b> <br>If the insomnia is caused by a physical change in the structure of an organ, such as the brain or thyroid, it is called organic insomnia.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Difficulty falling asleep, including difficulty finding a Domfortable sleeping position. </li>
                                <li>Waking during the night and being unable to return to sleep. </li>
                                <li>Feeling unrefreshed upon waking. </li>
                                <li>Daytime sleepiness, irritability or anxiety.</li>
                            </ul><br><br>
        <b>Prevention</b><br>Make sleep a priority, avoid caffeine, avoid looking at bright screens before going to bed. Make a sleep journal and develop sleep hygiene.<br><br>
        """)]

    elif ICD == 'd_733':
        # 39
        return ["https://s-media-cache-ak0.pinimg.com/originals/46/e0/01/46e0019883bd92d5d300713196c256c1.jpg",
                Markup("""
        <b>Description</b> <br> Cartilage is the tough but flexible tissue that covers the ends of yourbones at a joint. It also protects bones by preventing them from rubbing against each other. Injured, inflamed, or damaged cartilage can cause symptoms such as pain and limited movement. It can also lead to joint damage and deformity.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Arthritis is inflammation of the joints. </li>
                                <li>Hip Pain There are many causes of hip pain, such as arthritis, Trauma, strains, sprains, and other conditions that cause referred hip pain.  </li>
                                <li>Knee Pain Overview Knee pain has a wide variety of causes and treatments.</li>
                            </ul><br>
        <b>Diagnosis</b> <br>Telling the difference between cartilage damage in the knee and a sprain, or ligament damage, is not easy because the symptoms can be similar. However, modern non-invasive tests make the job much easier than it used to be. Usually Magnetic Resonance Imaging and Arthroscopy are used in diagnosis.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>If left untreated, the joint, especially if it is a weight-bearing one, such as the knee, can eventually become so damaged that the person cannot walk.</li>
                                                <li>Apart from immobility, the pain may slowly get worse.</li>
                                                <li>All small articular cartilage defects can eventually progress to osteoarthritis if given enough time.</li>
                                            </ul><br><br>
        <b>Prevention</b><br><ul>
                                    <li>Maintain your ideal body weight</li>
                                    <li>Control blood sugar</li>
                                    <li>Be active every day</li>
                                    <li>Avoid injury to joints</li>
                                    <li>Pay attention to pain</li>
                                </ul><br><br>
        """)]

    elif ICD == 'd_300':
        # 40
        return ["http://naomigoodlet.com/wp-content/uploads/2014/01/Anxiety-Infographic-www.naomigoodlet.com_1.jpg",
                Markup("""
        <b>Description</b> <br>Anxiety states are defined as the presence of anxiety in a situation or to a degree where it becomes maladaptive. Simple states are agoraphobia, a fear of crowds, claustrophobia, fear of enclosed spaces eg lifts, and stage fright.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Pounding heart, sweating. </li>
                                <li>Headaches, stomach upset, or dizziness. </li>
                                <li>Frequent urination or diarrhea. </li>
                                <li>Shortness of breath. </li>
                                <li>Muscle tension, tremors, and twitches. </li>
                                <li>Fatigue or insomnia.</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>You must have three features to be diagnosed with social anxiety disorder: Your symptoms must not be the result of some other mental health condition (for example, a delusion). You feel anxious entirely or mostly in social situations. One of your main symptoms will be the avoidance of social situations.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Depression (which often occurs with generalized anxiety disorder)</li>
                                                <li>Substance abuse.</li>
                                                <li>Trouble sleeping (insomnia)</li>
                                                <li>Digestive or bowel problems.</li>
                                                <li>Headaches.</li>
                                                <li>Heart-health issues.</li>
                                            </ul><br><br>
        <b>Prevention</b><br>Breathe deeply. When you get anxious, your breathing quickens, which reduces the amount of oxygen your brain gets. Take some time to exercise.  Meditate or pray.  Keep a healthy diet. Take a magnesium supplement.  Try an herbal remedy.  Visit a therapist.<br><br>
        """)]

    elif ICD == 'd_278':
        # 41
        return [
            "https://infographiclist.files.wordpress.com/2014/04/vicious-circle-of-the-os-obesity-and-osteoarthritis_5252bc9086287.jpg",
            Markup("""
        <b>Description</b> <br>Overweight and obesity are defined as abnormal or excessive fat accumulation that may impair health. Body mass index (BMI) is a simple index of weight-for-height that is commonly used to classify overweight and obesity in adults.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Breathing disorders (e.g., sleep apnea, chronic obstructive Pulmonary disease) </li>
                                <li>Certain types of cancers (e.g., prostate and bowel cancer in men, Breast and uterine cancer in women) </li>
                                <li>Coronary artery (heart) disease. </li>
                                <li>Depression. </li>
                                <li>Diabetes. </li>
                                <li>Gallbladder or liver disease.</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>Body mass index (BMI) is widely used as a simple and reliable way of finding out whether a person is a healthy weight for their height. For most adults, having a BMI of 18.5 to 24.9 means you're considered to be a healthy weight.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Coronary heart disease.</li>
                                                <li>High blood pressure.</li>
                                                <li>Stroke.</li>
                                                <li>Type 2 diabetes.</li>
                                                <li>Cancer.</li>
                                                <li>Sleep apnea.</li>
                                                <li>Gallstones.</li>
                                                <li>Osteoarthritis.</li>
                                            </ul><br><br>
        <b>Prevention</b><br><ul>
                                <li>Exercise regularly. You need to get 150 to 300 minutes of moderate-intensity activity a week to prevent weight gain.</li>
                                <li>Follow a healthy eating plan. </li>
                                <li>Know and avoid the food traps that cause you to eat. </li>
                                <li>Monitor your weight regularly. </li>
                                <li>Be consistent.</li>
                            </ul><br><br>
        """)]

    elif ICD == 'd_348':
        # 42
        return [
            "http://zliving.azureedge.net/zliving/zliving/media/zliving/health/prevention-healing/articles/1076x616/70lesser-known-symptoms-of-brain-cancer-760x428.jpg?ext=.jpg",
            Markup("""
        <b>Description</b> <br>Arachnoid cysts are cerebrospinal fluid covered by arachnoidal cells and collagen that may develop between the surface of thebrain and the cranial base or on the arachnoid membrane, one of the 3 meningeal layers that cover the brain and the spinal cord.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Headache. </li>
                                <li>Nausea and vomiting. </li>
                                <li>Hydrocephalus (accumulation of fluid in the brain, which can change the shape of an infant's head) </li>
                                <li>Developmental delay. </li>
                                <li>Behavioral changes. </li>
                                <li>Seizures. </li>
                                <li>Hearing and visual disturbances. </li>
                                <li>Vertigo</li>
                            </ul>
        <b>Diagnosis</b> <br>In many cases, arachnoid cysts do not cause symptoms (asymptomatic). In cases in which symptoms occur, headaches, seizures and abnormal accumulation of excessive cerebrospinal fluid in the brain (hydrocephalus) are common. The exact cause of arachnoid cysts is unknown.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Brain damage</li>
                                                <li>Failure to thrive in infants and children</li>
                                                <li>Hydrocephalus (fluid buildup in the skull)</li>
                                                <li>Permanent nerve damage including paralysis</li>
                                                <li>Seizures and tremors</li>
                                            </ul>
        <b>Prevention</b><br>Unfortunately there is no prevention for the arachnoid cyst. The best thing you can do is stay fit and healthy and maintain a healthy diet.<br><br>
        """)]

    elif ICD == 'd_578':
        # 43
        return ["https://www.unitedwaynems.org/wp-content/uploads/2014/03/Flag_of_the_Red_Cross.png",
                Markup("""
        <b>Description</b> <br>Hematemesis or haematemesis is the vomiting of blood.  The source is generally the upper gastrointestinal tract, typically above the suspensory muscle of duodenum.  Patients can easily confuse it with hemoptysis (coughing up blood), although the latter is more common.<br><br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Mallory-Weiss syndrome: bleeding tears in the esophagal mucosa, usually caused by prolonged and vigorous retching. </li>
                                <li>Irritation or erosion of the lining of the esophagus or stomach. </li>
                                <li>Vomiting of ingested blood after hemorrhage in the oral cavity, nose or throat</li>
                            </ul><br><br>
        <b>Prevention</b><br><ul>
                                <li>Avoid alcohol</li>
                                <li>Stop smoking</li>
                                <li>Speak with your doctor about limiting aspirin intake</li>
                                <li>Stay vigilant after surgery</li>
                                <li>Avoid foods that irritate the stomach or cause acid reflux</li>
                            </ul><br><br>
        """)]

    elif ICD == 'd_572':
        # 44
        return ["https://www.unitedwaynems.org/wp-content/uploads/2014/03/Flag_of_the_Red_Cross.png",
                Markup("""
        <b>Description</b> <br>A liver abscess is a pus-filled mass inside the liver. Common causes are abdominal infections such as appendicitis or diverticulitis due to haematogenous spread through the portal vein.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Abdominal pain, particularly in the right, upper part of the abdomen; pain is intense, continuous or stabbing. </li>
                                <li>Cough. </li>
                                <li>Fever and chills. </li>
                                <li>Diarrhea (in only one-third of patients) </li>
                                <li>General discomfort, uneasiness, or ill feeling (malaise)</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br><ul>
                                <li>A CT scan to locate the abscess</li>
                                <li>A CT scan with intravenous contrast, or injected dye, to find and measure the abscess</li>
                                <li>Blood tests to look for signs of an infection, such as an increased white blood count and high neutrophil level</li>
                                <li>Blood cultures for bacteria to identify the bacteria and determine which antibiotic you need</li>
                                <li>An abdominal ultrasound to check for an abscess in the right upper quadrant</li>
                            </ul><br><br>
        <b>Potential Complications</b> <br>Complications of Pyogenic Liver Abscess- The main complication of PLA is sepsis, which is a body-wide bacterial infection that causes inflammation and a dangerous drop in blood pressure<br><br>
        <b>Prevention</b><br>Primary prevention. Appropriate and timely treatment of intra-abdominal infections can prevent the complication of liver abscess. Antibiotic prophylaxis for chemoembolisation and, in selected cases, at endoscopic retrograde cholangiography is used as a primary preventative strategy.<br><br>
        """)]

    elif ICD == 'd_482':
        # 45
        return ["https://s-media-cache-ak0.pinimg.com/originals/c9/e3/fd/c9e3fdf6993f611790dad8564b8522eb.jpg",
                Markup("""
        <b>Description</b> <br>Inflammation of the lung (pneumonia) caused by the GRAM NEGATIVE organism Klebsiella pneumoniae .One or more lobes of the lungs become solidified and partially destroyed. Large amounts of purulent, brownish sputum are produced. The organisms is resistant to many antibiotics.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Bloodstream infections (bacteremia and sepsis) from Klebsiella cause fever, chills, rash, light-headedness, and altered mental states. </li>
                                <li>Pneumonia from K.pneumoniae can result in: Fevers and chills. </li>
                                <li>Flu-like symptoms</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>Chest x ray to look for inflammation in your lungs. A chest x ray is the best test fordiagnosing pneumonia. However, this test won't tell your doctor what kind of germ is causing the pneumonia. Blood tests such as a complete blood count (CBC) to see if your immune system is actively fighting an infection.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Bacteria in the bloodstream (bacteremia). Bacteria that enter the bloodstream from your lungs can spread the infection to other organs, potentially causing organ failure.</li>
                                                <li>Difficulty breathing. If your pneumonia is severe or you have chronic underlying lung diseases, you may have trouble breathing in enough oxygen. You may need to be hospitalized and use a breathing machine (ventilator) while your lung heals.</li>
                                                <li>Fluid accumulation around the lungs (pleural effusion). Pneumonia may cause fluid to build up in the thin space between layers of tissue that line the lungs and chest cavity (pleura). If the fluid becomes infected, you may need to have it drained through a chest tube or removed with surgery.</li>
                                                <li>Lung abscess. An abscess occurs if pus forms in a cavity in the lung. An abscess is usually treated with antibiotics. Sometimes, surgery or drainage with a long needle or tube placed into the abscess is needed to remove the pus.</li>
                                            </ul><br><br>
        <b>Prevention</b><br>Get a flu shot every year to prevent seasonal influenza. The flu is a common cause of pneumonia, so preventing the flu is a good way to prevent pneumonia. Children younger than 5 and adults 65 and older should get vaccinated against pneumococcal pneumonia, a common form of bacterial pneumonia.<br><br>
        """)]

    elif ICD == 'd_425':
        # 46
        return ["https://s-media-cache-ak0.pinimg.com/originals/29/59/e8/2959e89c4f6f939850d84ba38ccbf988.jpg",
                Markup("""
        <b>Description</b> <br>An acquired or hereditary disease of heart muscle, this condition makes it hard for the heart to deliver blood to the body, and can lead to heart failure.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Breathlessness</li>
                                <li>Swollen legs and feet and a bloated stomach.</li>
                            </ul>
        <b>Diagnosis</b> <br><ul>
                                <li>Blood tests. These tests give your doctor information about your heart. They also may reveal if you have an infection, a metabolic disorder or toxins in your blood that can cause dilated cardiomyopathy.</li>
                                <li>Chest X-ray. Your doctor may order a chest X-ray to check your heart and lungs for abnormalities in the heart's structure and size and for fluid in or around your lungs.</li>
                                <li>Electrocardiogram (ECG). An electrocardiogram — also called an ECG or EKG — records electrical signals as they travel through your heart. Your doctor can look for patterns that show abnormal heart rhythm or problems with the left ventricle. Your doctor may ask you to wear a portable ECG device known as a Holter monitor to record your heart rhythm for a day or two.</li>
                                <li>Echocardiogram. This primary tool for diagnosing dilated cardiomyopathy uses sound waves to produce images of the heart, allowing your doctor to see whether your left ventricle is enlarged. This test can also reveal how much blood is ejected from the heart with each beat and whether blood is flowing in the right direction.</li>
                                <li>Exercise stress test. Your doctor may have you perform an exercise test, either walking on a treadmill or riding a stationary bike. Electrodes attached to you during the test help your doctor measure your heart rate and oxygen use.</li>
                                <li>This type of test can show the severity of problems caused by dilated cardiomyopathy. If you're unable to exercise, you may be given medication to create the stress.</li>
                                <li>CT or MRI scan. In some situations, your doctor might order one of these tests to check the size and function of your heart's pumping chambers.</li>
                                <li>Cardiac catheterization. For this invasive procedure, a long, narrow tube is threaded through a blood vessel in your arm, groin or neck into the heart. The test enables your doctor to see your coronary arteries on X-ray, measure pressure in your heart and collect a sample of muscle tissue to check for damage that indicates dilated cardiomyopathy.</li>
                                <li>This procedure may involve having a dye injected into your coronary arteries to help your doctor study your coronary arteries (coronary angiography).</li>
                                <li>Genetic screening or counseling. If your doctor can't identify the cause of dilated cardiomyopathy, he or she may suggest screening of other family members to see if the disease is inherited in your family.</li>
                            </ul>
        <b>Potential Complications</b> <br><ul>
                                                <li>Heart failure.</li>
                                                <li>Poor blood flow from the left ventricle can lead to heart failure.</li>
                                                <li>Heart valve regurgitation.</li>
                                            </ul>
        <b>Prevention</b><br>You cannot prevent inherited types of cardiomyopathy. But you can take steps to lower your risk for diseases or conditions that may lead to or complicatecardiomyopathy. Examples include coronary heart disease, high blood pressure, and heart attack. Cardiomyopathy may be due to an underlying disease or condition.<br><br>
        """)]

    elif ICD == 'd_280':
        # 47
        return ["http://www.who.int/mediacentre/infographic/nutrition/anemia-jpg-large.jpg?ua=1",
                Markup("""
        <b>Description</b> <br>Blood contains iron within red blood cells. Women with heavy periods are at risk of iron deficiency anemia because they lose blood during menstruation. Slow,chronic blood loss within the body — such as from a peptic ulcer, a hiatal hernia, a colon polyp or colorectal cancer — can cause iron deficiency anemia.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Extreme fatigue. </li>
                                <li>Weakness. </li>
                                <li>Pale skin. </li>
                                <li>Chest pain, fast heartbeat or shortness of breath. </li>
                                <li>Headache, dizziness or lightheadedness. </li>
                                <li>Cold hands and feet. </li>
                                <li>Inflammation or soreness of your tongue. </li>
                                <li>Brittle nails.</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br><ul>
                                <li>Complete blood count (CBC). A CBC is used to count the number of blood cells in a sample of your blood. </li>
                                <li>A test to determine the size and shape of your red blood cells.</li>
                            </ul><br><br>
        <b>Potential Complications</b> <br> Any long-term medical condition can lead to anemia. The exact mechanism of this process in unknown, but any long-standing and ongoing medical condition such as a chronic infection or a cancer may cause this type of anemia.<br><br>
        <b>Prevention</b><br><ul>
                                <li>Treat underlying conditions. </li>
                                <li>Ingest iron supplements. </li>
                                <li>Eat a diet rich in iron. </li>
                                <li>Increase your Vitamin C and folate intake. </li>
                                <li>Consume foods containing vitamin B12.</li>
                                <li>Take B12 and folate supplements. </li>
                                <li>Get a B12 prescription.</li>
                                <li>Cook using iron pots and pans.</li>
                            </ul><br><br>
        """)]

    elif ICD == 'd_274':
        # 48
        return [
            "http://orangecountyfootandankle.com/wp-content/uploads/2014/05/Gout-by-Orange-County-Foot-and-Ankle1.jpg",
            Markup("""
        <b>Description</b> <br>Gout is a kind of arthritis. It can cause an attack of sudden burning pain, stiffness, and swelling in a joint, usually a big toe. These attacks can happen over and over unless gout is treated. Over time, they can harm your joints, tendons, and other tissues.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Pain areas: in the joints, ankle, foot, knee, or toe </li>
                                <li>Joints: swelling, tenderness, lumps, stiffness, or swelling </li>
                                <li>Also common: physical deformity or redness</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>Joint arthrocentesis is the definitive diagnostic study for gouty arthritis. Since urate crystals precipitate within the affected joint, their presence is viewable under polarized light microscopy of synovial fluid. ... Serum uric acid is elevated in 95% of patients during a flare.<br><br>
        <b>Potential Complications</b> <br> Gout is the painful and acute onset of an inflammatory arthritis. It’s caused by the buildup of uric acid in the blood. Many people who experience one gout attack never have a second attack. Others develop chronic gout or repeated attacks that happen more often over time. Chronic gout can lead to more severe problems, especially if left untreated.<br><br>
        <b>Prevention</b><br>Weight gain is a significant risk factor for gout in men, whereas weight loss reduces the risk.Intake of high-fructose corn syrup should be restricted because the fructose contributes to increased uric acid production as a byproduct of adenosine triphosphate catabolism.Patients with gout should limit their intake of purine-rich animal protein (e.g., organ meats, beef, lamb, pork, shellfish) and avoid alcohol (especially beer). Purine-rich vegetables do not increase the risk of gout.Consumption of vegetables and low-fat or nonfat dairy products9 should be encouraged.<br><br>
        """)]

    elif ICD == 'd_440':
        # 49
        return [
            "https://static1.squarespace.com/static/54a7706de4b039f26ff67092/54f3e43de4b0b75ae55d6095/54f3e9a1e4b0a87b1adf97d0/1429028152607/Inflammation_Milner-02.jpg?format=1500w",
            Markup("""
        <b>Description</b> <br>The build-up of fats, cholesterol and other substances in and on the artery walls.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Plaques may rupture, causing acute occlusion of the artery by clot. </li>
                                <li>Atherosclerosis often has no symptoms until a plaque ruptures or the build-up is severe enough to block blood flow.</li>
                            </ul>
        <b>Diagnosis</b> <br>These tests can include: a blood test to check your cholesterol levels. a Doppler ultrasound, which uses sound waves to create a picture of the artery that shows if there's a blockage. ankle-brachial index test, which looks for a blockage in your arms or legs by comparing the blood pressure in each limb<br><br>
        <b>Potential Complications</b> <br> <ul>
                                                <li>If you have atherosclerosis in your heart arteries, you may have symptoms, such as chest pain or pressure (angina).</li>
                                                <li>If you have atherosclerosis in the arteries leading to your brain, you may have signs and symptoms such as sudden numbness or weakness in your arms or legs, difficulty speaking or slurred speech, temporary loss of vision in one eye, or drooping muscles in your face. These signal a transient ischemic attack (TIA), which, if left untreated, may progress to a stroke.</li>
                                                <li>If you have atherosclerosis in the arteries in your arms and legs, you may have symptoms of peripheral artery disease, such as leg pain when walking (claudication).</li>
                                            </ul>
        <b>Prevention</b><br>A healthy diet and exercise can help.<br><br>
        """)]

    elif ICD == 'd_357':
        # 50
        return ["https://s-media-cache-ak0.pinimg.com/originals/d5/88/59/d5885994fa6222a779bd5813beb726c6.png",
                Markup("""
        <b>Description</b> <br>Peripheral neuropathy. Any syndrome in which muscle weakness, paresthesias, impaired reflexes, and autonomic symptoms in the hands and feet are common. This syndrome occurs in patients with diabetes mellitus, renal or hepatic failure, alcoholism, or in those who take certain medications such as phenytoin and isoniazid.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Sensory nerves that receive sensation, such as temperature, pain, vibration or touch, from the skin</li>
                                <li>Motor nerves that control muscle movement</li>
                                <li>Autonomic nerves that control functions such as blood pressure, heart rate, digestion and bladder</li>
                                <li>Gradual onset of numbness, prickling or tingling in your feet or hands, which can spread upward into your legs and arms</li>
                                <li>Sharp, jabbing, throbbing, freezing or burning pain</li>
                                <li>Extreme sensitivity to touch</li>
                                <li>Lack of coordination and falling</li>
                                <li>Muscle weakness or paralysis if motor nerves are affected</li>
                            </ul><br><br>
        <b>Potential Complications</b> <br> <ul>
                                                <li>Heat intolerance and altered sweating</li>
                                                <li>Bowel, bladder or digestive problems</li>
                                                <li>Changes in blood pressure, causing dizziness or lightheadedness</li>
                                                <li>Burns and skin trauma. You might not feel temperature changes or pain on parts of your body that are numb.</li>
                                                <li>Infection. Your feet and other areas lacking sensation can become injured without your knowing. Check these areas regularly and treat minor injuries before they become infected, especially if you have diabetes mellitus.</li>
                                                <li>Falls. Weakness and loss of sensation may be associated with lack of balance and falling.</li>
                                            </ul><br><br>
        """)]

    elif ICD == 'd_198':
        # 51
        return ["http://www.wcisu.wales.nhs.uk/sitesplus/contentimages/1111/Kidney%20Cancer%20English.jpg",
                Markup("""
        <b>Description</b> <br>Kidney cancer is a disease in which the cells in certain tissues of the kidney start to grow uncontrollably and form tumors. Renal cell carcinoma, which occurs in the cells lining the kidneys (epithelial cells), is the most common type ofkidney cancer.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Blood in your urine. </li>
                                <li>A lump in your side or abdomen. </li>
                                <li>A loss of appetite. </li>
                                <li>A pain in your side that doesn't go away. </li>
                                <li>Weight loss that occurs for no known reason. </li>
                                <li>Fever that lasts for weeks and isn't caused by a cold or other infection. </li>
                                <li>Extreme fatigue. </li>
                                <li>Anemia</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br><ul>
                                <li>Blood and urine tests. Tests of your blood and your urine may give your doctor clues about what's causing your signs and symptoms.</li>
                                <li>Imaging tests. Imaging tests allow your doctor to visualize a kidney tumor or abnormality. Imaging tests might include ultrasound, computerized tomography (CT) scan or magnetic resonance imaging (MRI).</li>
                                <li>Removing a sample of kidney tissue (biopsy). In rare cases, your doctor may recommend a procedure to remove a small sample of cells (biopsy) from a suspicious area of your kidney. The sample is tested in a lab to look for signs of cancer.</li>
                            </ul><br><br>
        <b>Potential Complications</b> <br> <ul>
                                                <li>High blood pressure (hypertension)</li>
                                                <li>Too much calcium in the blood</li>
                                                <li>High red blood cell count</li>
                                                <li>Liver problems</li>
                                                <li>Spread of the cancer</li>
                                            </ul><br><br>
        <b>Prevention</b><br>Obesity and high blood pressure are also risk factors . Maintaining a healthy weight by exercising and choosing a diet high in fruits and vegetables, and getting treatment for high blood pressure may also reduce your chance of getting this disease.<br><br>
        """)]

    elif ICD == 'd_443':
        # 52
        return ["https://s-media-cache-ak0.pinimg.com/736x/54/98/a0/5498a07772960fc2a9c9a0a4c298fc86.jpg",
                Markup("""
        <b>Description</b> <br>Kidney cancer is a disease in which the cells in certain tissues of the kidney start to grow uncontrollably and form tumors. Renal cell carcinoma, which occurs in the cells lining the kidneys (epithelial cells), is the most common type of kidney cancer.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Painful cramping in your hip, thigh or calf muscles after certain activities, such as walking or climbing stairs (claudication) </li>
                                <li>Leg numbness or weakness. </li>
                                <li>Coldness in your lower leg or foot, especially when compared with the other side.</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>Ankle-brachial index (ABI). This is a common test used to diagnose PAD. It compares the blood pressure in your ankle with the blood pressure in your arm. To get a blood pressure reading, your doctor uses a regular blood pressure cuff and a special ultrasound device to evaluate blood pressure and flow.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Amputation (loss of a limb)</li>
                                                <li>Poor wound healing.</li>
                                                <li>Restricted mobility due to pain or discomfort with exertion.</li>
                                                <li>Severe pain in the affected extremity.</li>
                                                <li>Stroke (three times more likely in people with PVD)</li>
                                            </ul><br><br>
        <b>Prevention</b><br><ul>
                                <li>Be physically active.</li>
                                <li>Be screened for P.A.D. A simple office test, called an ankle-brachial index or ABI, can help determine whether you have P.A.D.</li>
                                <li>Follow heart-healthy eating.</li>
                                <li>If you smoke, quit. Talk with your doctor about programs and products that can help you quit smoking.</li>
                                <li>If you’re overweight or obese, work with your doctor to create a reasonable weight-loss plan.</li>
                            </ul><br><br>
        """)]

    elif ICD == 'd_197':
        # 53
        return ["https://www.nationaljewish.org/NJH/media/img/stock/lung-infographic-full.JPG",
                Markup("""
        <b>Description</b> <br>When cancer cells travel to other organs in the body, it's called metastasis. Metastatic lung cancer is a life-threatening condition that develops when cancer in another area of the body metastasizes, or spreads, to the lung. Cancer that develops at any primary site can form metastatic tumors.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>a persistent cough. </li>
                                <li>coughing up blood or bloody phlegm. </li>
                                <li>chest pain. </li>
                                <li>shortness of breath. </li>
                                <li>wheezing. </li>
                                <li>weakness. </li>
                                <li>sudden weight loss.</li>
                            </ul>
        <b>Diagnosis</b> <br><ul>
                                <li>Imaging tests. An X-ray image of your lungs may reveal an abnormal mass or nodule. A CT scan can reveal small lesions in your lungs that might not be detected on an X-ray.</li>
                                <li>Sputum cytology. If you have a cough and are producing sputum, looking at the sputum under the microscope can sometimes reveal the presence of lung cancer cells.</li>
                                <li>Tissue sample (biopsy). A sample of abnormal cells may be removed in a procedure called a biopsy.</li>
                            </ul>
        <b>Potential Complications</b> <br><ul>
                                                <li>Coughing up blood. Lung cancer can cause bleeding in the airway, which can cause you to cough up blood (hemoptysis). Sometimes bleeding can become severe. Treatments are available to control bleeding.</li>
                                                <li>Pain. Advanced lung cancer that spreads to the lining of a lung or to another area of the body, such as a bone, can cause pain.</li>
                                                <li>Tell your doctor if you experience pain. Pain may initially be mild and intermittent, but can become constant. Medications, radiation therapy and other treatments may help make you more comfortable.</li>
                                                <li>Cancer that spreads to other parts of the body (metastasis). Lung cancer often spreads (metastasizes) to other parts of the body, such as the brain and the bones.</li>
                                                <li>Cancer that spreads can cause pain, nausea, headaches, or other signs and symptoms depending on what organ is affected. Once lung cancer has spread to other organs, it's generally not curable. Treatments are available to decrease signs and symptoms and to help you live longer</li>
                                            </ul>
        <b>Prevention</b><br>There's no sure way to prevent lung cancer, but you can reduce your risk if you: Don't smoke. If you've never smoked, don't start. Talk to your children about not smoking so that they can understand how to avoid this major risk factor for lung cancer.<br><br>
        """)]

    elif ICD == 'd_008':
        # 54
        return ["http://www.lumibyte.eu/wp-content/uploads/E.coli-Infographic.png",
                Markup("""
        <b>Description</b> <br>E. coli (Escherichia coli) is the name of a germ, or bacterium, that lives in the digestive tracts of humans and animals. Some strains of E. coli bacteria may also cause severe anemia or kidney failure, which can lead to death. Other strains of E. coli can cause urinary tract infections or other infections.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Bloody diarrhea. </li>
                                <li>Stomach cramps. </li>
                                <li>Nausea and vomiting</li>
                            </ul>
        <b>Diagnosis</b> <br><ul>
                                <li>You have had diarrhea that isn’t getting better after four days, or two days for an infant or child.</li>
                                <li>You have a fever with diarrhea.</li>
                                <li>Abdominal pain doesn’t get better after a bowel movement.</li>
                                <li>There is pus or blood in your stool.</li>
                                <li>You have trouble keeping liquids down.</li>
                                </ul>
        <b>Potential Complications</b> <br><ul>
                                                <li>Coughing up blood. Lung cancer can cause bleeding in the airway, which can cause you to cough up blood (hemoptysis). Sometimes bleeding can become severe. Treatments are available to control bleeding.</li>
                                                <li>Pain. Advanced lung cancer that spreads to the lining of a lung or to another area of the body, such as a bone, can cause pain.</li>
                                                <li>Tell your doctor if you experience pain. Pain may initially be mild and intermittent, but can become constant. Medications, radiation therapy and other treatments may help make you more comfortable.</li>
                                                </ul>
        <b>Prevention</b><br><ul>
                                <li>washing fruits and vegetables thoroughly</li>
                                <li>avoiding cross-contamination by using clean utensils, pans, and serving platters</li>
                                <li>keeping raw meats away from other foods and away from other clean items</li>
                                <li>not defrosting meat on the counter</li>
                                <li>always defrosting meat in the refrigerator or microwave</li>
                                </ul>
        """)]

    elif ICD == 'd_438':
        # 55
        return [
            "http://thumbnails-visually.netdna-ssl.com/mental-health-cause-and-prevention-of-mental-disorder_51f852af6a55c.jpg",
            Markup("""
        <b>Description</b> <br>Cognitive deficit is an inclusive term used to describe impairment in an individual's mental processes that lead to the acquisition of information and knowledge, and drive how an individual understands and acts in the world. <br>
        <b>Symptoms</b> <br><ul>
                                <li>There's no single cause of mild cognitive impairment (MCI), just as there's no single outcome for the disorder. </li>
                                <li>Symptoms of MCI may remain stable for years, progress to Alzheimer's disease or another type of dementia, or improve over time.</li>
                                <li>Difficulty speaking or understanding</li>
                                <li>Severe onset of headache</li>
                                 </ul>
        <b>Diagnosis</b> <br>A doctor commonly diagnoses a stroke through a physical examination of the person affected as well as a description of the symptoms they are experiencing. A doctor attempts to find the location in the person's brain that has experienced damage through testing involving a CT or MRI scans, which may also help to rule out brain hemorrhage or tumors. <br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Vision impairment, particularly blindness or vision field problems in one eye</li>
                                                <li>Depression, mood and behavioral disturbances</li>
                                                <li>Impaired speech and verbal comprehension</li>
                                                <li>Paralysis affecting only one side of the body</li>
                                                </ul>
        <b>Prevention</b><br><ul>
                                <li>Improving blood cholesterol levels (lowering LDL and raising HDL)</li>
                                <li>Eating a heart-healthy diet, low in trans and saturated fat</li>
                                </ul>
        """)]

    elif ICD == 'd_303':
        # 56
        return ["https://www.cdc.gov/vitalsigns/alcohol-poisoning-deaths/images/graphic1_970px.jpg",
                Markup("""
        <b>Description</b> <br>A disturbance in behaviour or mental function during or after alcohol consumption.Acute alcohol poisoning is a related medical term used to indicate a dangerously high concentration of alcohol in the blood, high enough to induce coma, respiratory depression, or even death.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Depression</li>
                                <li>Euphoria problems with coordination</li>
                                <li>rapid involuntary eye movement</li>
                                <li>slow breathing</li>
                            </ul><br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Whole body: blackout, dehydration, fainting, flushing, or low blood sugar </li>
                                                <li>Cognitive: amnesia, mental confusion, or unresponsiveness </li>
                                                <li>Gastrointestinal: nausea or vomiting </li>
                                                <li>Behavioural: aggression or lack of restraint </li>
                                                <li>Speech: slurred speech or impaired voice</li>
                                            </ul><br><br>
        """)]

    elif ICD == 'd_453':
        # 57
        return ["https://www.unitedwaynems.org/wp-content/uploads/2014/03/Flag_of_the_Red_Cross.png",
                Markup("""
        <b>Description</b> <br>Budd–Chiari syndrome is a very rare condition, affecting 1 in a million adults. The condition is caused by occlusion of the hepatic veins that drain the liver. It presents with the classical triad of abdominal pain, ascites, and liver enlargement. <br><br>
        <b>Symptoms</b> <br><ul>
                                <li>The formation of a blood clot within the hepatic veins can lead to Budd–Chiari syndrome. </li>
                                <li>The syndrome can be fulminant, acute, chronic, or asymptomatic.</li>
                            </ul><br><br>
        """)]


    elif ICD == 'd_345':
        # 58
        return ["https://stroke.nih.gov/images/NINDS_LYR_infographic.jpeg",
                Markup("""
        <b>Description</b> <br>A transient ischemic attack (TIA) is a transient episode of neurologic dysfunction caused by ischemia (loss of blood flow) – either focal brain, spinal cord, or retinal – without acute infarction (tissue death)<br>
        <b>Symptoms</b> <br><ul>
                                <li>Whole body: feeling faint, light-headedness, or vertigo </li>
                                <li>Muscular: muscle weakness, problems with coordination, or weakness of one side of the body </li>
                                <li>Speech: slurred speech, speech disorder, or impaired voice </li>
                                <li>Sensory: pins and needles or reduced sensation of touch </li>
                                <li>Visual: blurred vision or vision loss </li>
                                <li>Facial: muscle weakness or numbness </li>
                                <li>Also common: arm weakness, difficulty swallowing, facial nerve paralysis, limping, or mental confusion</li>
                            </ul>
        <b>Diagnosis</b> <br>TIA will usually be diagnosed after a doctor performs a history and a physical exam. There are several radiological tests that are done to evaluate patients who have had a TIA. These include a CT scan or an MRI of the brain, ultrasound of the neck, or an echocardiogram of the heart. In most cases, the source of atherosclerosis is usually identified with an ultrasound.<br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Muscle weakness</li>
                                                <li>Amaurosis fugax</li>
                                                <li>Dizzyness</li>
                                                <li>Visual impairment</li>
                                            </ul>
        <b>Prevention</b><br><ul>
                                <li>Avoiding smoking</li>
                                <li>Cutting down on fats to help reduce the amount of plaque build up</li>
                                <li>Eating a healthy diet including plenty of fruits and vegetables</li>
                                <li>Limiting sodium in the diet, thereby reducing blood pressure</li>
                                <li>Exercising regularly</li>
                                </ul>
        """)]

    elif ICD == 'd_682':
        # 59
        return ["https://www.unitedwaynems.org/wp-content/uploads/2014/03/Flag_of_the_Red_Cross.png",
                Markup("""
        <b>Description</b> <br>Cellulitis is a bacterial infection involving the inner layers of the skin. It specifically affects the dermis and subcutaneous fat. ... For facial infections, a break in the skin beforehand is not usually the case. The bacteria most commonly involved are streptococci and Staphylococcus aureus.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Redness. </li>
                                <li>Red streaking. </li>
                                <li>Swelling. </li>
                                <li>Warmth. </li>
                                <li>Pain or tenderness. </li>
                                <li>Leaking of yellow, clear fluid or pus.</li>
                            </ul>
        <b>Diagnosis</b> <br><ul>
                                <li>A blood test if the infection is suspected to have spread to your blood</li>
                                <li>An X-ray if there’s a foreign object in the skin or the bone underneath is possibly infected</li>
                                <li>A culture. Your doctor will use a needle to draw fluid from the affected area and send it to the lab.</li>
                            </ul>
        <b>Potential Complications</b> <br><ul>
                                                <li>Complications of cellulitis can include blood poisoning, abscesses, and meningitis.</li>
                                                <li> If the bacteria that infect your skin and tissue enter your bloodstream, they can cause blood poisoning (septicaemia).</li>
                                            </ul>
        <b>Prevention</b><br><ul>
                                    <li>Practice good personal hygiene and keep your skin clean.</li>
                                    <li>Wear sturdy, well-fitting shoes or slippers with loose-fitting cotton socks. Avoid walking barefoot outdoors.</li>
                                    <li>Wash injured skin with soap and water. Make sure it heals over the next few days.</li>
                                </ul>
        """)]

    elif ICD == 'd_286':
        # 60
        return ["http://www.phoenix-cardiology.com/images/medium/862829.png",
                Markup("""
        <b>Description</b> <br>Coagulopathy (also called a bleeding disorder) is a condition in which the blood's ability to coagulate (form clots) is impaired. This condition can cause a tendency toward prolonged or excessive bleeding (bleeding diathesis), which may occur spontaneously or following an injury or medical and dental procedures.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Blood in the urine or stool. </li>
                                <li>Bruising easily and excessively. </li>
                                <li>Extreme fatigue. </li>
                                <li>An injury that will not stop bleeding. </li>
                                <li>Joint pain caused by internal bleeding. </li>
                                <li>Nosebleeds that seem to have no cause. </li>
                                <li>A painful headache that will not go away.</li>
                            </ul>
        <b>Diagnosis</b> <br><ul>
                                <li>a complete blood count (CBC), which measures the amount of red and white blood cells in your body</li>
                                <li>a platelet aggregation test, which checks how well your platelets clump together</li>
                                <li>a bleeding time, which determines how quickly your blood clots to prevent bleeding</li>
                            </ul>
        <b>Potential Complications</b> <br><ul>
                                                <li>Hemophilia</li>
                                                <li>Von Willebrand disease</li>
                                                <li>Other clotting factor deficiencies</li>
                                                <li>Disseminated intravascular coagulation</li>
                                                <li>Liver Disease</li>
                                                <li>Overdevelopment of circulating anticoagulants</li>
                                                <li>Vitamin K deficiency</li>
                                                <li>Platelet dysfunction</li>
                                            </ul>
        <b>Prevention</b><br>Prevention depends on the specific disorder.<br><br>
        """)]


    elif ICD == 'd_112':
        # 61
        return ["https://www.unitedwaynems.org/wp-content/uploads/2014/03/Flag_of_the_Red_Cross.png",
                Markup("""
        <b>Description</b> <br>Candidiasis is an infection caused by a species of the yeast Candida, usually Candida albicans. This is a common cause of vaginal infections in women. Also, Candida may cause mouth infections in people with reduced immune function, or in patients taking certain antibiotics.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Vaginal yeast infection </li>
                                <li>This condition can cause inflammation, intense itchiness and a thick, white discharge from the vagina. </li>
                                <li>Candidiasis of skin and nails </li>
                                <li>Symptoms include a red rash. Common areas affected include the skin between the fingers and toes, around the fingernails and toenails and the groin. </li>
                                <li>Oral thrush </li>
                                <li>Oral thrush causes white lesions on the tongue or inner cheeks</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>Healthcare providers rely on your medical history, symptoms, physical examinations, and laboratory tests to diagnose invasive candidiasis. The most common way that healthcare providers test for invasive candidiasis is by taking a blood sample and sending it to a laboratory to see if it will grow Candida in a culture.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Immune-suppressing diseases, including HIV.</li>
                                                <li>Diabetes.</li>
                                                <li>Obesity.</li>
                                            </ul><br><br>
        <b>Prevention</b><br>Garlic is believed to have some natural antifungal properties and may help to prevent candidiasis. Drink milk or eat yogurt that contains acidophilus bacteria. Apply yogurt containing "friendly" bacteria directly into the vagina (such as Lactobacillus bifidus or Lactobacillus acidophilus).<br><br>
        """)]

    elif ICD == 'd_491':
        # 62
        return ["http://www.fabhow.com/wp-content/uploads/2017/02/intro-bronchitis-1.jpg",
                Markup("""
        <b>Description</b> <br>Chronic bronchitis is one type of COPD (chronic obstructive pulmonary disease). The inflamed bronchial tubes produce a lot of mucus. This leads to coughing and difficulty breathing. Cigarette smoking is the most common cause.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Cough: can be dry, with phlegm, can occur due to smoking, during exercise, or severe </li>
                                <li>Respiratory: difficulty breathing, expiratory wheezing, fast breathing, frequent respiratory infections, rapid breathing, shortness of breath, shortness of breath at night, shortness of breath on exercise, shortness of breath on lying down, or wheezing </li>
                                <li>Whole body: fatigue or inability to exercise </li>
                                <li>Weight: underweight or weight loss </li>
                                <li>Also common: phlegm, anxiety, bulging chest, chest tightness, depression, fast heart rate, flare, high carbon dioxide levels in blood, limping, muscle weakness, phlegm with pus, or sleeping difficulty</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>To be diagnosed with chronic bronchitis, the cough and excessive mucus production must have occurred for 3 months or more in at least 2 consecutive years and not be due to any other disease or condition. Tests to diagnose chronic bronchitis include: Pulmonary function tests. Arterial blood gas.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Pneumonia. Around one person in 20 with bronchitis may develop a secondary infection in the lungs leading to pneumonia.</li>
                                                <li>Patients with heart or lung diseases co-existing with bronchitis are also at risk of pneumonia. </li>
                                                <li>This includes heart failure patients, asthma patients, COPD patients.</li>
                                            </ul><br><br>
        <b>Prevention</b><br>Avoid cigarette smoke. Cigarette smoke increases your risk of chronic bronchitis. Get vaccinated. You may also want to consider vaccination that protects against some types of pneumonia.<br><br>
        """)]

    elif ICD == 'd_288':
        # 63
        return [
            "https://aos.iacpublishinglabs.com/question/aq/1400px-788px/happens-many-red-blood-cells_891c9a08c6bfe4aa.jpg?domain=cx.aos.ask.com",
            Markup("""
        <b>Description</b> <br>Neutropenia or neutropenia, is an abnormally low concentration of neutrophils in the blood. Neutrophils make up the majority of circulating white blood cells.It can be caused by diseases that damage the bone marrow, infections or certain medication. <br><br>
        <b>Symptoms</b> <br>There can be no symptoms other than an increased vulnerability to infection.<br><br>
        <b>Diagnosis</b> <br>Neutropenia is diagnosed by a blood cell count performed on a sample of blood removed from a vein. To determine the specific cause of neutropenia in a given situation, other tests may be required. Sometimes a bone marrow biopsy may be required to diagnose the specific cause of neutropenia.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Death</li>
                                                <li>Intensive care unit admission</li>
                                                <li>Confusion</li>
                                                <li>Cardiac complications</li>
                                                <li>Respiratory failure</li>
                                                <li>Renal failure</li>
                                                <li>Hypotension,</li>
                                            </ul><br><br>
        <b>Prevention</b><br>This practice is supported by international guidelines, all of which recommend that primary prophylaxis with granulocyte colony-stimulating factors should be used with chemotherapy where the risk of febrile neutropenia is 20% or greater.<br><br>
        """)]

    elif ICD == 'd_600':
        # 64
        return [
            "https://www.cookmedical.com/urology/wp-content/uploads/sites/14/2015/06/URO-D19526_2015-06-15_124326.jpg",
            Markup("""
        <b>Description</b> <br>Age-associated prostate gland enlargement that can cause urination difficulty. With this condition, the urinary stream may be weak or stop and start. In some cases, it can lead to infection, bladder stones and reduced kidney function.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Pain circumstances: can occur during urination </li>
                                <li>Urinary: bladder spasm, dribbling after urination, excessive urination at night, frequent urge to urinate, frequent urination, sense of incomplete bladder emptying, urge to urinate and leaking, urinary retention, urinary tract infection, or weak urinary stream </li>
                                <li>Also common: incontinence, recurrent infection, or sexual dysfunction</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>A urinalysis and urine culture check for a urinary tract infection that might be the cause of the symptoms. A prostate-specific antigen (PSA) test helps check forprostate cancer, which can cause the same symptoms as BPH.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Complete blockage of the urethra (acute urinary retention, or AUR). This results in a complete inability to urinate. It can cause kidney damage, which may be reversed if the problem is diagnosed and treated before the damage becomes too severe. It may also result in waste products building up in the blood. A tube called a catheter may be needed to drain urine from the bladder.</li>
                                                <li>Long-term, partial blockage of urine flow from the bladder (chronic urinary retention, or CUR). This causes urine to remain in the bladder after urination (post-void residual urine). In rare cases, this may lead to kidney damage, which may be reversed if the problem is diagnosed and treated before the damage becomes too severe. It may also result in waste products building up in the blood.</li>
                                                <li>A urinary tract infection (UTI). But repeated urinary tract infections can also be caused by long-term inflammation or infection in the prostate (chronic prostatitis)</li>
                                            </ul><br><br>
        <b>Prevention</b><br>Benign Prostatic Hyperplasia (BPH) The urination problems caused by benignprostatic hyperplasia (BPH) cannot be prevented. Some people believe that regular ejaculations will help prevent prostate enlargement. But there is no scientific proof that ejaculation helps.<br><br>
        """)]

    elif ICD == 'd_577':
        # 65
        return [
            "http://www.good-legal-advice.com/wp-content/uploads/2012/08/victoza-injuries-cancer-of-the-pancreas-infographic-victoza-lawyer.jpg",
            Markup("""
        <b>Description</b> <br>An inflammation of the organ lying behind the lower part of the stomach (pancreas). Pancreatitis may start suddenly and last for days or it can occur over many years. It has many causes, including gallstones and chronic, heavy alcohol use.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Symptoms include upper abdominal pain, nausea and vomiting. </li>
                                <li>Treatment usually requires hospitalisation. Once they stabilise the patient, doctors treat the underlying cause.</li>
                            </ul>
        <b>Diagnosis</b> <br>Diagnosis of Acute Pancreatitis. Acute pancreatitis is confirmed by medical history, physical examination, and typically a blood test (amylase or lipase) for digestive enzymes of the pancreas. Blood amylase or lipase levels are typically elevated 3 times the normal level during acute pancreatitis.<br><br>
        <b>Potential Complications</b> <br><ul>
                                            <li>Pseudocyst. Acute pancreatitis can cause fluid and debris to collect in cystlike pockets in your pancreas. </li>
                                            <li>Infection. </li>
                                            <li>Kidney failure. </li>
                                            <li>Breathing problems. </li>
                                            <li>Diabetes. </li>
                                            <li>Malnutrition. </li>
                                            <li>Pancreatic cancer.</li>
                                        </ul>
        <b>Prevention</b><br>Eat a low-fat diet. Gallstones, a leading cause ofacute pancreatitis, can develop when too much cholesterol accumulates in your bile, the substance made by your liver to help digest fats. To reduce your risk for gallstones, eat a low-fat diet that includes whole grains and a variety of fresh fruits and vegetables.<br><br>
        """)]

    elif ICD == 'd_362':
        # 66
        return ["https://c1.staticflickr.com/1/757/20739543108_08694a8447_b.jpg",
                Markup("""
        <b>Description</b> <br>Diabetic retinopathy affects blood vessels in the light-sensitive tissue called the retina that lines the back of the eye. It is the most common cause of vision loss among people with diabetes and the leading cause of vision impairment and blindness among working-age adults. Diabetic macular edema (DME).<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Spots or dark strings floating in your vision (floaters) </li>
                                <li>Blurred vision. </li>
                                <li>Fluctuating vision. </li>
                                <li>Impaired color vision. </li>
                                <li>Dark or empty areas in your vision. </li>
                                <li>Vision loss.</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>Diabetic retinopathy can be detected during a dilated eye exam by an ophthalmologist or optometrist. An exam by your primary doctor, during which your eyes are not dilated, is not an adequate substitute for a full exam done by an ophthalmologist. Eye exams for people with diabetes can include: Visual acuity testing.<br><br>
        <b>Potential Complications</b> <br><ul>
                                            <li>Retinal detachment.</li>
                                            <li> The abnormal blood vessels associated with diabetic retinopathy stimulate the growth of scar tissue, which can pull the retina away from the back of the eye.</li>
                                            <li> This may cause spots floating in your vision, flashes of light or severe vision loss.</li>
                                        </ul><br><br>
        <b>Prevention</b><br>However, regular eye exams, good control of your blood sugar and blood pressure, and early intervention for vision problems can help prevent severe vision loss. If you have diabetes, reduce your risk of getting diabetic retinopathy by doing the following: Manage your diabetes.<br><br>
        """)]

    elif ICD == 'd_519':
        # 67
        return ["https://www.unitedwaynems.org/wp-content/uploads/2014/03/Flag_of_the_Red_Cross.png",
                Markup("""
        <b>Description</b> <br>Tracheostomy is a surgical procedure to create an opening in the neck for direct access to the trachea (the breathing tube).Tracheostomy is performed because of airway obstruction, problems with secretions, and inefficient oxygen delivery. Tracheostomy can havecomplications.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Bleeding. </li>
                                <li>Air trapped around the lungs (pneumothorax) </li>
                                <li>Air trapped in the deeper layers of the chest(pneumomediastinum) </li>
                                <li>Air trapped underneath the skin around the tracheostomy (subcutaneous emphysema)</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>It includes nursing diagnosis for: Risk for ineffective airway clearance, risk for infection, and impaired verbal communication. As a nurse you may encounter a patient who has a tracheostomy. In the medical setting you may hear it called a “trach”<br><br>
        <b>Potential Complications</b> <br><ul>
                                            <li>Blocked tube.</li>
                                            <li>Bleeding from the airway/tracheostomy tube.</li>
                                            <li>Pneumothorax.</li>
                                            <li>Subcutaneous and/or mediastinal emphysema.</li>
                                            <li>Respiratory and/or cardiovascular collapse.</li>
                                            <li>Dislodged tube.</li>
                                            <li>Granulation tissue.</li>
                                            <li>Tracheo-oesophageal fistula.</li>
                                        </ul><br><br>
        """)]

    elif ICD == 'd_562':
        # 68
        return ["https://www.unitedwaynems.org/wp-content/uploads/2014/03/Flag_of_the_Red_Cross.png",
                Markup("""
        <b>Description</b> <br>Diverticula can form while straining during a bowel movement, such as with constipation. They are most common in the lower portion of the large intestine(called the sigmoid colon). Diverticulosis is very common and occurs in 10% of people over age 40 and in 50% of people over age 60.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Pain areas: in the abdomen or side part of the body </li>
                                <li>Gastrointestinal: bloating, blood in stool, change in bowel habits, constipation, diarrhoea, indigestion, nausea, vomiting, or flatulence </li>
                                <li>Whole body: chills, fever, or loss of appetite </li>
                                <li>Abdominal: cramping or tenderness</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>Plain abdominal X-ray may show signs of a thickened wall, ileus, constipation, smallbowel obstruction or free air in the case of perforation. Plain X-rays are insufficient to diagnose diverticular disease.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>The resulting complication is known as a non-inflammatory stricture and may require surgery to correct.</li>
                                                <li> If the bowels become completely blocked by scarring, this is known as intestinal blockage or bowel obstruction.</li>
                                                <li> These complicationsare life-threatening, as they can lead to a ruptured intestine and peritonitis</li>
                                            </ul><br><br>
        <b>Prevention</b><br>Diverticulitis occurs when the bulging sacs that appear in the lining of your large intestine, or colon, get acutely infected or inflamed. The most common and severe symptom is sudden pain on the lower left side of the abdomen. Drinking plenty of water and eating fiber-rich foods can help you avoid diverticulitis.<br><br>
        """)]

    elif ICD == 'd_294':
        # 69
        return ["http://cdn2.factorialist.com/wp-content/uploads/2015/05/factorialist_snapchat_infographic_thumb.jpg",
                Markup("""
        <b>Description</b> <br>The amnestic disorders are a group of disorders that involve loss of memories previously established, loss of the ability to create new memories, or loss of the ability to learn new information.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Difficulty recalling remote events or information. </li>
                                <li>Difficulty learning and then recalling new information. </li>
                                <li>The patient in some cases is fully aware of the memory impairment, and frustrated by it</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>To diagnose amnesia, a doctor will do a comprehensive evaluation to rule out other possible causes of memory loss, such as Alzheimer's disease, other forms of dementia, depression or brain tumor.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Amnesia varies in severity and scope, but even mild amnesia takes a toll on daily activities and quality of life.</li>
                                                <li> The syndrome can cause problems at work, at school and in social settings.</li>
                                                <li>It may not be possible to recover lost memories. Some people with severe memory problems need to live in a supervised situation or extended-care facility.</li>
                                            </ul><br><br>
        <b>Prevention</b><br><ul>
                                <li>Avoid excessive alcohol use.</li>
                                <li>Wear a helmet when bicycling and a seat belt when driving.</li>
                                <li>Treat any infection quickly so that it doesn't have a chance to spread to the brain.</li>
                                <li>Seek immediate medical treatment if you have any symptoms that suggest a stroke or brain aneurysm, such as a severe headache or one-sided numbness or paralysis.</li>
                            </ul><br><br>
        """)]

    elif ICD == 'd_275':
        # 70
        return ["https://www.unitedwaynems.org/wp-content/uploads/2014/03/Flag_of_the_Red_Cross.png",
                Markup("""
        <b>Description</b> <br>Iron metabolism disorder. Genes involved in iron metabolism disorders include HFE and TFR2. Hepcidin is the master regulator of iron metabolism and, therefore, most genetic forms of iron overload can be thought of as relative hepcidin deficiency in one way or another.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Iron Overload. </li>
                                <li>Hemochromatosis. Classic Hemochromatosis. Juvenile Hemochromatosis. Neonatal Hemochromatosis. African Hemochromatosis. Other. </li>
                                <li>Iron Deficiency Anemia. </li>
                                <li>Anemia of Chronic Disease. </li>
                                <li>Iron Overload with Anemia. </li>
                                <li>Iron-Out-of-Balance. </li>
                                <li>Rare. </li>
                                <li>Too Much or Too Little Iron.</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>Various Blood Tests.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>iver, with cirrhosis.</li>
                                                <li>Heart, with cardiomyopathy.</li>
                                                <li>Pancreas, with diabetes mellitus.</li>
                                                <li>Skin, with pigmentation.</li>
                                                <li>Joints, with polyarthropathy.</li>
                                                <li>Gonads, with hypogonadotrophic hypogonadism</li>
                                            </ul><br><br>
        <b>Prevention</b><br>To prevent iron deficiency anemia in infants, feed your baby breast milk or iron-fortified formula for the first year.  After age 6 months, start feeding your baby iron-fortified cereals or pureed meats at least twice a day to boost ironintake.<br><br>
        """)]

    elif ICD == 'd_E8798':
        # 71
        return ["https://www.unitedwaynems.org/wp-content/uploads/2014/03/Flag_of_the_Red_Cross.png",
                Markup("""
        <b>Description</b> <br>Abn react-procedure NEC (Other specified procedures as the cause of abnormal reaction of patient, or of later complication, without mention of misadventure at time of procedure).<br><br>
        """)]

    elif ICD == 'd_263':
        # 72
        return ["https://www.unitedwaynems.org/wp-content/uploads/2014/03/Flag_of_the_Red_Cross.png",
                Markup("""
        <b>Description</b> <br>Protein-energy undernutrition (PEU), previously called protein-energy malnutrition, is an energy deficit due to chronic deficiency of all macronutrients. It commonly includes deficiencies of many micronutrients. PEU can be sudden and total (starvation) or gradual.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Poor weight gain. </li>
                                <li>Slowing of linear growth. </li>
                                <li>Behavioral changes - Irritability, apathy, decreased social responsiveness, anxiety, and attention deficits</li>
                            </ul><br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>A sluggish metabolism.</li>
                                                <li>Trouble losing weight.</li>
                                                <li>Trouble building muscle mass.</li>
                                                <li>Low energy levels and fatigue.</li>
                                                <li>Poor concentration and trouble learning.</li>
                                                <li>Moodiness and mood swings.</li>
                                                <li>Muscle, bone and joint pain.</li>
                                                <li>Blood sugar changes that can lead to diabetes.</li>
                                            </ul><br><br>
        <b>Prevention</b><br>In patients who are asymptomatic carriers of protein S deficiency, the goal of therapy is prevention of the first thrombosis. In such patients, avoid drugs that predispose to thrombosis, including oral contraceptives. In these patients, if surgery or orthopedic injury occurs, prophylaxis with heparin is mandatory.<br><br>
        """)]

    elif ICD == 'd_441':
        # 73
        return ["https://www.unitedwaynems.org/wp-content/uploads/2014/03/Flag_of_the_Red_Cross.png",
                Markup("""
        <b>Description</b> <br>An aneurysm that bleeds into the brain can lead to stroke or death. Aortic dissection occurs when the layers of the wall of the aorta separate or are torn, allowing blood to flow between those layers and causing them to separate further.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Anxiety and a feeling of doom. </li>
                                <li>Fainting or dizziness. </li>
                                <li>Heavy sweating (clammy skin) </li>
                                <li>Nausea and vomiting. </li>
                                </ul>
        <b>Diagnosis</b> <br><ul>
                                <li>Your doctor will examine you and use a stethoscope to listen for abnormal noises coming from your aorta. When your blood pressure is taken, the reading may be different in one arm than in the other.</li>
                                <li>You may also need to have imaging scans done. These can include:</li>
                                <li>a chest X-ray</li>
                                <li>a CT scan</li>
                                </ul>
        <b>Potential Complications</b> <br><ul>
                                                <li>high blood pressure</li>
                                                <li>smoking</li>
                                                <li>atherosclerosis, or hardening of your arteries</li>
                                                <li>conditions such as Marfan syndrome, in which your body’s tissues are weaker than normal</li>
                                                <li>surgical procedures done near the heart</li>
                                                </ul>
        <b>Prevention</b><br><ul>
                            <li>Type B dissection can often be treated with medication. Type A dissection typically requires surgery.</li>
                            <li>Medications</li>
                            <li>You’ll receive drugs to relieve your pain. Morphine is often used for pain. You’ll also get medication to lower your blood pressure. Beta-blockers are usually used for this.</li>
                            </ul>
        """)]

    elif ICD == 'd_569':
        # 74
        return ["https://www.unitedwaynems.org/wp-content/uploads/2014/03/Flag_of_the_Red_Cross.png",
                Markup("""
        <b>Description</b> <br>An intestinal polyp is any mass of tissue that arises from the bowel wall and protrudes into the lumen. Most are asymptomatic except for minor bleeding, which is usually occult. The main concern is malignant transformation; most colon cancers arise in a previously benign adenomatous polyp.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>blood in the stool or rectal bleeding. </li>
                                <li>pain, diarrhea, or constipation that lasts longer than one week. </li>
                                <li>nausea or vomiting if you have a large polyp.</li>
                            </ul>
        <b>Diagnosis</b> <br>In most cases, polyps do not cause symptoms and are usually found on routine colon cancer screening exams. However, if you do experience symptoms, they may include: blood in the stool or rectalbleeding. pain, diarrhea, or constipation that lasts longer than one week.<br><br>
        <b>Potential Complications</b> <br>Some colon polyps may become cancerous. The earlier polyps are removed, the less likely it is that they will become malignant.<br><br>
        <b>Prevention</b><br><ul>
                                <li>Eat fruits, vegetables and whole grains.</li>
                                <li>Reduce your fat intake.</li>
                                <li>Limit alcohol consumption.</li>
                                <li>Don't use tobacco.</li>
                                <li>Stay physically active and maintain a healthy body weight.</li>
                                <li>Talk to your doctor about calcium.</li>
                            </ul>
        """)]

    elif ICD == 'd_564':
        # 75
        return ["https://www.unitedwaynems.org/wp-content/uploads/2014/03/Flag_of_the_Red_Cross.png",
                Markup("""
        <b>Description</b> <br>A condition in which there is difficulty in emptying the bowels, usually associated with hardened faeces. <br><br>
        <b>Symptoms</b> <br><ul>
                            <li>Passing fewer than three stools a week </li>
                            <li>Having lumpy or hard stools </li>
                            <li>Straining to have bowel movements </li>
                            <li>Feeling as though there's a blockage in your rectum that prevents bowel movements </li>
                           </ul>
        <b>Diagnosis</b> <br>In addition to a general physical exam and a digital rectal exam, doctors use the following tests and procedures to diagnose chronic constipation and try to find the cause: Blood tests. Your doctor will look for a systemic condition such as low thyroid (hypothyroidism)<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Swollen veins in your anus (hemorrhoids)</li>
                                                <li>Torn skin in your anus (anal fissure</li>
                                                <li>Stool that can't be expelled (fecal impaction)</li>
                                                <li>Intestine that protrudes from the anus (rectal prolapse)</li>
                                            </ul>
        <b>Prevention</b><br><ul>
                                <li>Add veggies. You don't have to count grams of fiber to get the amount you need. </li>
                                <li>Go for grains. </li>
                                <li>Bulk up on beans. </li>
                                <li>Add fiber gradually. </li>
                                <li>Consider a fiber supplement. </li>
                                <li>Stay hydrated.</li>
                            </ul>
        """)]

    elif ICD == 'd_293':
        # 76
        return [
            "https://static1.squarespace.com/static/546e1217e4b093626abfbae7/t/580743aa5016e12d2c5b4441/1476871098342/Recognising+Delirium+in+the+ED+%5Binfographic%5D.png",
            Markup("""
        <b>Description</b> <br>An acutely disturbed state of mind characterized by restlessness, illusions, and incoherence, occurring in intoxication, fever, and other disorders. <br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Emotional or personality changes, with frequent changes in moods, including anger,agitation, anxiety, apathy, depression, fear, euphoria, irritability, suspicion.Incontinence. </li>
                                <li>Hallucinations (visual, but not auditory) </li>
                                <li>Signs of medical illness (such as fever, chills, pain, etc) or drug side effects</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>The diagnosis of delirium is clinical. No laboratory test can diagnose delirium. The Diagnostic and Statistical Manual of Mental Disorders, Fifth Edition (DSM-5)diagnostic criteria for delirium is as follows : Disturbance in attention (ie, reduced ability to direct, focus, sustain, and shift attention) and awareness.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Medically ill individuals, especially the elderly, may have serious complicationsassociated with delirium, including malnutrition</li>
                                                <li>Fluid and electrolyte abnormalities</li>
                                                <li>Pneumonia and decubitus ulcers.</li>
                                            </ul><br><br>
        <b>Prevention</b><br>Evidence indicates that these strategies — promoting good sleep habits, helping the person remain calm and well-oriented, and helping prevent medical problems or other complications — can help prevent or reduce the severity ofdelirium.<br><br>
        """)]

    elif ICD == 'd_338':
        # 77
        return ["https://www.unitedwaynems.org/wp-content/uploads/2014/03/Flag_of_the_Red_Cross.png",
                Markup("""
        <b>Description</b> <br>Central pain syndrome is a neurological condition caused by damage to or dysfunction of the central nervous system (CNS), which includes the brain, brainstem, and spinal cord. This syndrome can be caused by stroke, multiple sclerosis, tumors, epilepsy, brain or spinal cord trauma, or Parkinson's disease.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>brain hemorrhage. </li>
                                <li>a stroke. </li>
                                <li>multiple sclerosis. </li>
                                <li>brain tumors. </li>
                                <li>an aneurysm. </li>
                                <li>a spinal cord injury. </li>
                                <li>a traumatic brain injury. </li>
                                <li>epilepsy.</li>
                            </ul><br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>stress</li>
                                                <li>anxiety</li>
                                                <li>depression</li>
                                                <li>fatigue</li>
                                                <li>sleep disturbances</li>
                                                <li>relationship problems</li>
                                                <li>anger</li>
                                                <li>a decrease in quality of life</li>
                                                <li>isolation</li>
                                                <li>suicidal thoughts</li>
                                            </ul><br><br>
        """)]

    elif ICD == 'd_456':
        # 78
        return ["https://www.unitedwaynems.org/wp-content/uploads/2014/03/Flag_of_the_Red_Cross.png",
                Markup("""
        <b>Description</b> <br>This increased pressure in the portal vein causes the development of large, swollen veins (varices) within the esophagus and stomach. The varices are fragile and can rupture easily, resulting in a large amount of blood loss. The most common cause of portal hypertension is cirrhosis of the liver<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Gastrointestinal: bloating, blood in stool, dark stool from digested blood, fluid in the abdomen, or vomiting blood </li>
                                <li>Also common: portal hypertension, bleeding, enlarged veins around belly button, flapping hand tremor, gastric varices, or web of swollen blood vessels in the skin</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br>Scarring (cirrhosis) of the liver is the most common cause of esophageal varices. ... This extra blood flow causes the veins in the esophagus to balloon outward. If these veins break open, they can bleed severely. Any type of chronic liver disease can cause esophageal varices.<br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Vomiting and seeing significant amounts of blood in your vomit.</li>
                                                <li>Black, tarry or bloody stools.</li>
                                                <li>Lightheadedness.</li>
                                                <li>Loss of consciousness (in severe case)</li>
                                            </ul><br><br>
        <b>Prevention</b><br>Eat a healthy diet that largely consists of lean protein, whole grains, fruits, and vegetables. Stop drinking alcohol.<br><br>
        """)]

    elif ICD == 'd_715':
        # 79
        return ["http://thumbnails-visually.netdna-ssl.com/living-with-osteoarthritis_53d6dee8df8e1_w1500.jpg",
                Markup("""
        <b>Description</b> <br>Osteoarthrosis is a non-inflammatory joint disease characterized by degeneration of the articular cartilage, hypertrophy of bone at the margins, and changes in the synovial membrane. It is also known as Degenerative Arthritis, Hypertrophic Arthritis and Degenerative Joint Disease.<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Pain areas: in the joints, knee, hip, lower back, or neck </li>
                                <li>Pain types: can be severe in the joints </li>
                                <li>Joints: crackles, stiffness, swelling, swelling, or tenderness </li>
                                <li>Also common: creaky joints, joint deformity, limping, muscle weakness, or tenderness</li>
                            </ul><br><br>
        <b>Diagnosis</b> <br><ul>
                                    <li>Pain areas: in the joints, knee, hip, lower back, or neck </li>
                                    <li>Pain types: can be severe in the joints </li>
                                    <li>Joints: crackles, stiffness, swelling, swelling, or tenderness </li>
                                    <li>Also common: creaky joints, joint deformity, limping, muscle weakness, or tenderness</li>
                                </ul><br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>Rapid, complete breakdown of cartilage resulting in loose tissue material in the joint (chondrolysis).</li>
                                                <li>Bone death (osteonecrosis).</li>
                                                <li>Stress fractures (hairline crack in the bone that develops gradually in response to repeated injury or stress).</li>
                                                <li>Bleeding inside the joint.</li>
                                            </ul><br><br>
        <b>Prevention</b><br>Maintain a healthy weight. Excess weight is one of the biggest risk factors of OA, as it puts extra stress on your joints, which can speed up the deterioration of joint cartilage. Overweight and obese individuals are at high risk of developing OA. Losing weight can help reduce pain and improve symptoms.<br><br>
        """)]

    elif ICD == 'd_996':
        # 80
        return ["http://www.oklahomaheart.com/sites/default/files/Cardiac%20Monitoring%20Infographic_FINAL.jpg",
                Markup("""
        <b>Description</b> <br>Breakdown (mechanical), Displacement, Leakage, Obstruction, mechanical, Perforation, Protrusion<br><br>
        <b>Symptoms</b> <br><ul>
                                <li>Symptoms can vary considerably and also vary in severity. </li>
                                <li>Symptoms include pulsation</li>
                                <li> fullness in the neck,</li>
                                <li> dizziness,</li>
                                <li> palpitations,</li>
                                <li> fatigue, </li>
                                <li>light-headedness and </li>
                                <li>syncope.</li>
                            </ul><br><br>
        <b>Potential Complications</b> <br><ul>
                                                <li>signs of heart failure may occur. </li>
                                                <li>Signs include hypotension, tachycardia, tachypnoea, raised JVP and cannon waves. </li>
                                                <li>There may be variations in pulses and fluctuating blood pressure - a drop of 20 mm Hg or more during ventricular pacing compared with that during atrial or AV synchronous pacing is suggestive.</li>
                                            </ul><br><br>
        """)]

    else:
        return ["https://www.unitedwaynems.org/wp-content/uploads/2014/03/Flag_of_the_Red_Cross.png",
                Markup(""" Information Currently Unavailable!""")]


if __name__ == '__main__':
    app.run(debug=True)
