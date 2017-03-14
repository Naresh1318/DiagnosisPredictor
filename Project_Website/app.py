from flask import Flask, render_template, request
import dense_fully_connected_tfidf
import os
import sys

lib_path = os.path.abspath(os.path.join('../', 'lib'))
sys.path.append(lib_path)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index_form.html')


@app.route('/results', methods=['POST'])
def hello():
    if request.method == 'POST':
        input_seq = request.form['text']
        length = dense_fully_connected_tfidf.predict(input_seq)
        seq = []
        for each_seq in length:
            word = ''
            seq_list = []
            each_seq = each_seq.split()
            seq_list.append(each_seq[2])
            seq_list.append(each_seq[5])
            for w in each_seq[8:]:
                word += w + ' '
            seq_list.append(word)
            seq.append(seq_list)

        name = request.form['name']
        return render_template('results_page.html', input_seq=input_seq, length=seq, name=name)

# TODO: debug=False when the app is deployed
if __name__ == '__main__':
    app.run(debug=True)
