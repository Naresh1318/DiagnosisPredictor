SET search_path TO mimiciii;

/* Create allevents table */ 
DROP TABLE if EXISTS allevents;
DROP SEQUENCE if EXISTS allevents_ids;
CREATE TABLE allevents (hadm_id integer, subject_id integer, charttime timestamp, event_type varchar, event varchar);

/* Adding most common abnormal 58 Labevents */
INSERT INTO allevents
SELECT hadm_id, subject_id, (array_agg(charttime ORDER BY charttime ASC))[1] AS charttime, 'labevent' AS event_type, itemid AS event FROM labevents
WHERE flag='abnormal' AND itemid IN
(
  SELECT itemid FROM
  (
    SELECT hadm_id, itemid FROM labevents WHERE flag='abnormal' GROUP BY hadm_id, itemid
  ) AS uniqlab
  GROUP BY itemid HAVING count(*) > 50
)
AND subject_id IN (SELECT subject_id FROM admissions GROUP BY subject_id HAVING count(*) > 1)
GROUP BY itemid, hadm_id, subject_id
;

/* Adding most common 66 Prescriptions */
INSERT INTO allevents
SELECT hadm_id, subject_id, starttime AS charttime, 'prescription' AS event_type, formulary_drug_cd AS event FROM prescriptions
WHERE formulary_drug_cd IN
(
  SELECT formulary_drug_cd FROM
  (
    SELECT hadm_id, formulary_drug_cd FROM prescriptions GROUP BY hadm_id, formulary_drug_cd
  ) AS uniqlab
  GROUP BY formulary_drug_cd HAVING count(*) > 50
)
AND subject_id IN (SELECT subject_id FROM admissions GROUP BY subject_id HAVING count(*) > 1)
;

/* Adding most common 58 Diagnoses */
INSERT INTO allevents
SELECT admissions.hadm_id, admissions.subject_id, dischtime AS charttime, 'diagnosis' AS event_type, icd9_code AS event FROM diagnoses_icd
LEFT JOIN admissions ON admissions.hadm_id = diagnoses_icd.hadm_id
WHERE icd9_code IN
(
  SELECT icd9_code FROM
  (
      SELECT hadm_id, icd9_code FROM diagnoses_icd GROUP BY hadm_id, icd9_code
  ) AS uniqlab
  GROUP BY icd9_code HAVING count(*) > 5
)
AND diagnoses_icd.subject_id IN
(SELECT subject_id FROM admissions GROUP BY subject_id HAVING count(*) > 1)
;

/* Delete unimportant diagnoses */
DELETE FROM allevents where event = 'NULL';
DELETE FROM allevents where event = '';
-- The event_type is a condition when icd9 starts with V
UPDATE allevents SET event_type='condition' WHERE event_type='diagnosis' AND event ~ '^V.*';
-- The event_type is a symptom when icd9 starts with 7 and contains either 8 or 9
UPDATE allevents SET event_type='symptom' WHERE event_type='diagnosis' AND event ~ '^7[89]\d.*';

/* Create unique ID*/ 
CREATE SEQUENCE allevents_ids;
ALTER TABLE allevents ADD id INT UNIQUE;
UPDATE allevents SET id = NEXTVAL('allevents_ids');

/* Adding icd9 levels */
ALTER TABLE allevents ADD COLUMN icd9_1 varchar(10);
ALTER TABLE allevents ADD COLUMN icd9_2 varchar(10);
ALTER TABLE allevents ADD COLUMN icd9_3 varchar(10);
