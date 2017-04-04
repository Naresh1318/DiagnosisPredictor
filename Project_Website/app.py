from flask import Flask, render_template, request
from flask_mail import Mail, Message
# import dense_fully_connected_tfidf
import test
import os
import sys

app = Flask(__name__)

# Parameters for Flask-mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'compvisionnn@gmail.com'
app.config['MAIL_PASSWORD'] = 'chzmbufftnjnyfah'
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

        length = test.delay(input_seq)
        # length = dense_fully_connected_tfidf.predict(input_seq)
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
            sendMail(emailID, name, input_seq, length)

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
        print(request.form)
        input_seq = request.form['input_seq']
        actual_diagnosis_temp = request.form['diagnosis_feedback']

        actual_diagnosis = ""
        # Add the d_ for diagnosis
        for i in actual_diagnosis_temp.split(','):
            actual_diagnosis += 'd_' + i + " "

        feedback_text = input_seq + '|' + actual_diagnosis
        feedback_text = feedback_text.strip()

        # Save the previous sequences in the file
        with open('data/feedback.txt', 'r') as fb:
            temp_text = fb.readlines()
        print(temp_text)
        with open('data/feedback.txt', 'w') as fb:
            temp_text.append(feedback_text)
            print(temp_text)
            for i in temp_text:
                fb.write(i + '\n')

        return render_template('thankyou.html')


def sendMail(emailID, name, input_seq, seq):
    msg = Message('Your Predicted Diagnosis', sender='compvisionnn@gmail.com', recipients=[emailID])
    html_message = '\
    <!DOCTYPE>\
    <html>\
    <meta name="viewport" content="width=device-width, initial-scale=1">\
    <link rel="stylesheet" href="http://www.w3schools.com/lib/w3.css" />\
    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Lato">\
    <head>\
    <title> Diagnosis </title>\
    </head>\
    <body>\
    <!-- Header -->\
    <section class="w3-container w3-center" style="max-with:600px">\
      <h2 class="w3-wide">Here is your diagnosis {0}</h2>\
      <p class="w3-opacity"><i>hope it is not too bad</i></p>\
    </section>\
    \
    <div class="w3-container" style="margin-left: auto; margin-right: auto;" class="small_div">\
    <div class="w3-panel w3-card-8 w3-center">\
    \
    <p><span class="w3-wide"> Previous History </span>\
        <br>\
        \
        {1}<br>\
        \
        </p>\
        \
        </div>\
        </div>\
    \
    <!-- Header of diagnosis -->\
    <div class="w3-container w3-center small_div_header" style="margin-left: auto; margin-right: auto;">\
        <div class="w3-panel w3-card-8 w3-dark-grey">\
        \
            <p><span class="w3-wide">Predicted Diagnosis</span> </p>\
        </div>\
\
    </div>\
    <div class="w3-container">\
    <p> {2} </p>\
    </div>\
    </html>\
    '.format(name, input_seq, seq)

    msg.html = html_message
    # with open('templates/email.html', 'r') as e:
    #    msg.html = e.read()
    mail.send(msg)


def get_disease_info(ICD):
    return {'description': 'An intestinal obstruction is a potentially serious condition in which the intestines are '
                           'blocked. The blockage may be either partial or complete, occurring at one or more '
                           'locations. \n Both the small intestine and large intestine, called the colon, '
                           'can be affected. When a blockage occurs, food and drink cannot pass through the body. '
                           'Obstructions are serious and need to be treated immediately. They may even require '
                           'surgery.',
            'symptoms': ['severe abdominal pain', 'cramps that come in waves', 'bloating', 'nausea and vomiting',
                         'diarrhea', 'constipation, or inability to have a bowel movement', 'inability to pass gas',
                         'distention or swelling of the abdomen', 'loud noises from the abdomen', 'foul breath'],
            'diagnosis': 'To diagnose a bowel obstruction, your doctor will need to feel and listen to your abdomen '
                         'and feel inside your rectum. A blockage in the intestine is confirmed by X-rays of your '
                         'abdomen, which show gas and liquid bowel contents above the area of the blockage, '
                         'but no gas below the blockage. Blood tests must be done to check for dehydration or loss of '
                         'electrolytes (such as sodium and potassium) if your symptoms have included vomiting. If '
                         'your doctor suspects you have a large-bowel obstruction, he or she may use a colonoscope, '
                         'a tube that is inserted through the rectum to view the lower intestine. If the obstruction '
                         'is caused by a volvulus, the passing of this instrument into the bowel not only confirms '
                         'the diagnosis, but also untwists the intestine and relieves the obstruction. It may not be '
                         'possible to know the cause of a bowel obstruction unless surgery is done. Surgery permits a '
                         'doctor to look at your intestine and at scar tissue if you have adhesions.',
            'potential_complications': ['Infection', 'Tissue Death', 'Intestinal Perforation', 'Sepsis',
                                        'Multisystem organ failure', 'Death'],
            'prevention': "Eat a balanced diet low in fat with plenty of vegetables and fruits, don't smoke, "
                          "and see your doctor for colorectal cancer screening once a year after age 50."}


if __name__ == '__main__':
    app.run(debug=True)
