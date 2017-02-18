"""Script to add values to the icd9_3 column in allevents table"""

import psycopg2
import re

# Try connecting to the database, replace dbname, user, host and password with appropriate values
try:
    conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost' password='password'")
except:
    print("I am unable to connect to the database")

# Get the cursor
cur = conn.cursor()

# Set the search path to mimiciii, this is used to avoid typing mimiciii.allevents every time
cur.execute("""set search_path to mimiciii""")
# Select all diagnosis event_type
cur.execute("""SELECT id, event from allevents where event_type='diagnosis'""")
# Get a list of tuples which contains id and event
rows = cur.fetchall()

# Set the icd9_3 column to the first 3 icd9 number when it doesn't start with a character
for row in rows:
    if re.match("^\d{3}", row[1]):
        updateRecordStatus = "update allevents set icd9_3='" + row[1][:3]
    else:
        updateRecordStatus = "update allevents set icd9_3='" + row[1]
    updateRecordStatus += "' where id=" + str(row[0]) + ";"
    cur.execute(updateRecordStatus)
    conn.commit()
