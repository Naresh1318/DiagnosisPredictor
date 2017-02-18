import psycopg2
import math
import json
import datetime
import os
from random import randrange, shuffle
from collections import defaultdict
from os import path

uniq_p_feat = ["gender", "age", "white", "asian", "hispanic", "black", "multi", "portuguese",
               "american", "mideast", "hawaiian", "other"]

seq_path = '../../Data/patient_sequences/'
balanced_seq_path = '../../Data/mimic_balanced/'
num_pred_diag = 80
balance_percent = 4 / 100


def calculate_window(events, days):
    event_count = defaultdict(lambda: 0)
    events_per_day = defaultdict(lambda: 0)
    first_day = days[0]

    for i, e in enumerate(events):
        event_count[e] += 1
        days[i] = (days[i] - first_day).days
        events_per_day[days[i]] += 1

    pre = [0] * len(events)
    suf = [0] * len(events)

    events_in_day = 0
    previous_day = None
    for i, e in enumerate(events):
        if previous_day != days[i]:
            previous_day = days[i]
            events_in_day = 0

        # Original equation for limit is (7 * event_count[e]) + 15
        limit = 365
        pre[i] = events_in_day
        for d in range(max(days[i] - limit, 0), days[i]):
            pre[i] += events_per_day[d]

        suf[i] = events_per_day[days[i]] - events_in_day - 1
        for d in range(days[i] + 1, days[i] + limit):
            suf[i] += events_per_day[d]

        events_in_day += 1

    pre_str = " ".join(map(str, pre))
    suf_str = " ".join(map(str, suf))

    return (pre_str, suf_str)


def set_p_features(hadm_id):
    cur.execute("""SELECT dob, admittime, gender, ethnicity
                FROM admissions JOIN patients
                ON admissions.subject_id = patients.subject_id
                WHERE hadm_id = %(hadm_id)s """ % {'hadm_id': str(hadm_id)})
    subject_info = cur.fetchall()
    feats = {}
    for k in uniq_p_feat:
        feats[k] = 0

    feats["gender"] = int(subject_info[0][2] == "M")
    num_years = (subject_info[0][1] - subject_info[0][0]).days / 365.25
    feats["age"] = num_years

    r = subject_info[0][3]
    if "WHITE" in r:
        feats["white"] = 1
    elif "ASIAN" in r:
        feats["asian"] = 1
    elif "HISPANIC" in r:
        feats["hispanic"] = 1
    elif "BLACK" in r:
        feats["black"] = 1
    elif "MULTI" in r:
        feats["multi"] = 1
    elif "PORTUGUESE" in r:
        feats["portuguese"] = 1
    elif "AMERICAN INDIAN" in r:
        feats["american"] = 1
    elif "MIDDLE EASTERN" in r:
        feats["mideast"] = 1
    elif "HAWAIIAN" in r or "CARIBBEAN" in r:
        feats["hawaiian"] = 1
    else:
        feats["other"] = 1
    return feats


print("Start")

# Try connecting to the database
try:
        conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost' password='password'")
except:
        print("I am unable to connect to the database")

cur = conn.cursor()
cur.execute("""SET search_path TO mimiciii""")
cur.execute("""SELECT subject_id, charttime, event_type, event, icd9_3, hadm_id
            FROM allevents ORDER BY subject_id, charttime,
            CASE event_type
            WHEN 'condition' THEN 1
            WHEN 'symptom' THEN 2
            WHEN 'labevent' THEN 3
            WHEN 'diagnosis' THEN 4
            WHEN 'prescription' THEN 5 END,
            event""")
rows = cur.fetchall()
print("Query executed")

prev_time = None
prev_subject = None
prev_hadm_id = None
diags = set()
total_diags = set()
event_seq = []
temp_event_seq = []
all_seq = []
all_days = []
unique_events = set()
diag_count = defaultdict(lambda: 0)

for row in rows:
    if row[2] == "diagnosis":
        # Add the d_ for diagnosis icd9_3 code
        event = row[2][:1] + "_" + row[4]
        if not row[2].startswith("E"):  # ?????
            diag_count[event] += 1
    else:
        # Add l_ or p_ for lab events or prescriptions resp
        event = row[2][:1] + "_" + row[3]

    if row[0] is None or row[1] is None or row[5] is None:
        continue

    elif prev_time is None or prev_subject is None:
        pass

    elif (row[0] != prev_subject) or (row[1] > prev_time + datetime.timedelta(365)):
        if len(diags) > 0 and len(event_seq) > 4:
            p_features = set_p_features(prev_hadm_id)
            pre, suf = calculate_window(event_seq+temp_event_seq, all_days)
            all_seq.append([p_features, event_seq, temp_event_seq, diags, pre, suf])
        diags = set()
        event_seq = []
        temp_event_seq = []
        all_days = []

    elif prev_hadm_id != row[5]:
        event_seq += temp_event_seq
        temp_event_seq = []
        diags = set()

    temp_event_seq.append(event)
    unique_events.add(event)
    all_days.append(row[1])

    prev_time = row[1]
    prev_subject = row[0]
    prev_hadm_id = row[5]

    if row[2] == "diagnosis":
        diags.add(event)
        total_diags.add(event)

# Write down the vocabulary used and diagnoses that we want to predict
# Get the top 80 most common diagnosis
predicted_diags = [y[0] for y in
                   sorted(diag_count.items(), key=lambda x: x[1], reverse=True)[:num_pred_diag]]

uniq = open(seq_path + 'vocab', 'w')
uniq.write(' '.join(unique_events) + '\n')
uniq.write(' '.join(predicted_diags))
uniq.close()

uniq = open(balanced_seq_path + 'vocab', 'w')
uniq.write(' '.join(unique_events) + '\n')
uniq.write(' '.join(predicted_diags))
uniq.close()

print("Number of total sequences {}".format(len(all_seq)))
print("Data structures created. Now writing files:")
train = {}
test = {}
valid = {}
trainv = {}
trainv_pre = {}
trainv_suf = {}
test_pre = {}
test_suf = {}

# To include all diagnoses change it to total_diags
for i in range(10):
    train[str(i)] = open(seq_path+'_train_'+str(i), 'w')
    trainv[str(i)] = open(seq_path+'_trainv_'+str(i), 'w')
    test[str(i)] = open(seq_path+'_test_'+str(i), 'w')
    valid[str(i)] = open(seq_path+'_valid_'+str(i), 'w')

segment = 0
shuffle(all_seq)
total = len(all_seq)
print(total)


valid_count = 0
for seq_index, seq in enumerate(all_seq):
    if math.floor(seq_index * 10 / total) > segment:
        print("New Segment "+str(segment))
        valid_count = 0
        segment += 1

    [patient, events, final_events, diagnoses, pre, suf] = seq
    serial = (",").join(diagnoses)
    serial += "|" + json.dumps(patient)
    serial += "|" + " ".join(events)
    serial += "|" + " ".join(final_events)
    serial += "|" + pre
    serial += "|" + suf

    test[str(segment)].write(serial+'\n')
    for f in range(10):
        if f != segment:
            trainv[str(f)].write(serial+'\n')
            if valid_count < math.floor(total / 10):
                valid[str(f)].write(serial+'\n')
                valid_count += 1
            else:
                train[str(f)].write(serial+'\n')

for i in range(10):
    train[str(i)].close()
    trainv[str(i)].close()
    test[str(i)].close()
    valid[str(i)].close()

print("Raw sequences generated")
''''# Generate balanced datasets
files = [f for f in os.listdir(seq_path)
         if path.isfile(path.join(seq_path, f)) and f.startswith("_")]

for f in files:
    print("Balancing file " + f)
    total = 0
    diag_lines = defaultdict(lambda: [])
    diag_counts = defaultdict(lambda: 0)
    final_lines = []

    with open(path.join(seq_path, f)) as old:
        for line in old.readlines():
            final_lines.append(line)
            total += 1
            diags = line.split("|")[0].split(",")
            for d in diags:
                if d in predicted_diags:
                    diag_lines[d].append(line)
                    diag_counts[d] += 1

    while min(diag_counts.values()) < int(total * balance_percent):
        minimum = int(total * balance_percent)
        d = min(diag_counts, key=diag_counts.get)
        for _ in range(minimum + 5 - diag_counts[d]):
            line = diag_lines[d][randrange(len(diag_lines[d]))]
            total += 1
            diags = line.split("|")[0].split(",")
            final_lines.append(line)
            for di in diags:
                if di in predicted_diags:
                    diag_counts[di] += 1

    shuffle(final_lines)
    with open(path.join(balanced_seq_path, f), 'w') as new:
        for line in final_lines:
            new.write(line)

print("Raw balanced sequences generated")'''

''''# Split seq, pre and suf files for balanced and normal ones
for directory in [seq_path, balanced_seq_path]:
    for f in files:
        with open(path.join(directory, f)) as combined:
            # Remove the first underscore from the name of the file
            p = path.join(directory, f[1:])
            with open(p, 'w') as seq, open(p+"_pre", 'w') as pre, open(p+"_suf", 'w') as suf:
                for line in combined.readlines():
                    data = line.split("|")
                    pre.write(data[4]+'\n')
                    suf.write(data[5])
                    seq.write("|".join(data[:4])+'\n')

        # Remove the original file
        os.remove(path.join(directory, f))

print("Final seq, pre, suf files created")'''
