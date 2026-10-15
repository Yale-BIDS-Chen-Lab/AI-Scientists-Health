#!/usr/bin/env python3
"""Build the agent workspace (set-a + set-b, neutral ids) and the withheld set-c
directory from the PhysioNet/CinC Challenge 2012 files.

Agent-visible output  : <workspace>/data/{patients,measurements,outcomes}.csv + docs/data_dictionary.md
Agent-invisible output: <stage>/truth/setc/{patients,measurements}.csv, <stage>/truth/setc_outcomes.csv,
                        <stage>/truth/id_map.csv, <stage>/truth/dev_outcomes_full.csv

Run: python3 prepare_workspace.py --source <dir with PhysioNet files> --workspace <path> --stage <path>

<source> must contain set-a.tar.gz, set-b.tar.gz, set-c.tar.gz and Outcomes-a.txt,
Outcomes-b.txt, Outcomes-c.txt as downloaded from PhysioNet.
"""
import argparse, csv, io, os, random, tarfile
from pathlib import Path

STATIC = {"RecordID", "Age", "Gender", "Height", "ICUType", "Weight"}
ICU = {1: "coronary_care", 2: "cardiac_surgery_recovery", 3: "medical", 4: "surgical"}


def read_set(tar_path):
    """Return {record_id: (static dict, [(minutes, var, value), ...])}."""
    out = {}
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            if not m.name.endswith(".txt"):
                continue
            text = tf.extractfile(m).read().decode()
            rows = list(csv.reader(io.StringIO(text)))[1:]
            static, series = {}, []
            rid = None
            for t, p, v in rows:
                hh, mm = t.split(":")
                minutes = int(hh) * 60 + int(mm)
                if p == "RecordID":
                    rid = int(v)
                    continue
                if p in STATIC and minutes == 0 and p not in static:
                    static[p] = float(v)
                    if p != "Weight":
                        continue
                series.append((minutes, p, float(v)))
            out[rid] = (static, series)
    return out


def source_file(src, name):
    """PhysioNet file name, or the same name with a "physionet2012_" prefix."""
    for cand in (src / name, src / f"physionet2012_{name}"):
        if cand.exists():
            return cand
    raise SystemExit(f"missing {name} in {src}")


def read_outcomes(path):
    with open(path) as f:
        r = csv.DictReader(f)
        return {int(x["RecordID"]): int(x["In-hospital_death"]) for x in r}


def write_split(records, outcomes, id_map, data_dir, with_outcomes=True):
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "patients.csv", "w", newline="") as fp, \
         open(data_dir / "measurements.csv", "w", newline="") as fm:
        wp = csv.writer(fp); wm = csv.writer(fm)
        wp.writerow(["patient_id", "age_years", "sex", "height_cm", "weight_kg", "icu_type"])
        wm.writerow(["patient_id", "minutes_since_admission", "variable", "value"])
        for rid in sorted(records, key=lambda r: id_map[r]):
            static, series = records[rid]
            pid = id_map[rid]
            sex = {0.0: "female", 1.0: "male"}.get(static.get("Gender", -1.0), "")
            def clean(x):
                return "" if x is None or x < 0 else x
            wp.writerow([pid, int(static.get("Age", -1)), sex, clean(static.get("Height")),
                         clean(static.get("Weight")), ICU.get(int(static.get("ICUType", 0)), "")])
            for minutes, var, val in series:
                wm.writerow([pid, minutes, var, val])
    if with_outcomes:
        with open(data_dir / "outcomes.csv", "w", newline="") as fo:
            wo = csv.writer(fo)
            wo.writerow(["patient_id", "in_hospital_death"])
            for rid in sorted(records, key=lambda r: id_map[r]):
                wo.writerow([id_map[rid], outcomes[rid]])


DICTIONARY = """# Data dictionary

Adult ICU admissions from a single US academic hospital. Each admission contributes
its first 48 hours in the ICU. Admissions shorter than 48 hours, and patients under
16, are not included. Only one admission per patient. Identifiers are study codes.

## `patients.csv` (one row per admission)

| column | description |
|---|---|
| patient_id | study code |
| age_years | age at admission |
| sex | female / male; blank if not recorded |
| height_cm | blank if not recorded |
| weight_kg | admission weight; blank if not recorded |
| icu_type | coronary_care / cardiac_surgery_recovery / medical / surgical |

## `measurements.csv` (long format, one row per observation)

| column | description |
|---|---|
| patient_id | study code |
| minutes_since_admission | 0 to 2880 |
| variable | name from the list below |
| value | numeric |

Variables are recorded when they were measured; there is no fixed sampling grid.
Absence of a row means the variable was not measured at that time.

| variable | description | unit |
|---|---|---|
| Albumin | serum albumin | g/dL |
| ALP | alkaline phosphatase | IU/L |
| ALT | alanine transaminase | IU/L |
| AST | aspartate transaminase | IU/L |
| Bilirubin | total bilirubin | mg/dL |
| BUN | blood urea nitrogen | mg/dL |
| Cholesterol | total cholesterol | mg/dL |
| Creatinine | serum creatinine | mg/dL |
| DiasABP | invasive diastolic arterial pressure | mmHg |
| FiO2 | fraction of inspired oxygen | 0 to 1 |
| GCS | Glasgow Coma Score | 3 to 15 |
| Glucose | serum glucose | mg/dL |
| HCO3 | serum bicarbonate | mmol/L |
| HCT | hematocrit | % |
| HR | heart rate | bpm |
| K | serum potassium | mEq/L |
| Lactate | lactate | mmol/L |
| Mg | serum magnesium | mmol/L |
| MAP | invasive mean arterial pressure | mmHg |
| MechVent | mechanical ventilation in use (1 = yes) | |
| Na | serum sodium | mEq/L |
| NIDiasABP | non-invasive diastolic arterial pressure | mmHg |
| NIMAP | non-invasive mean arterial pressure | mmHg |
| NISysABP | non-invasive systolic arterial pressure | mmHg |
| PaCO2 | partial pressure of arterial CO2 | mmHg |
| PaO2 | partial pressure of arterial O2 | mmHg |
| pH | arterial pH | |
| Platelets | platelet count | cells/nL |
| RespRate | respiration rate | bpm |
| SaO2 | O2 saturation in hemoglobin | % |
| SysABP | invasive systolic arterial pressure | mmHg |
| Temp | temperature | degrees C |
| TroponinI | troponin I | ug/L |
| TroponinT | troponin T | ug/L |
| Urine | urine output | mL |
| WBC | white blood cell count | cells/nL |
| Weight | weight | kg |

## `outcomes.csv`

| column | description |
|---|---|
| patient_id | study code |
| in_hospital_death | 1 if the patient died before hospital discharge, else 0 |
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--seed", type=int, default=2012)
    a = ap.parse_args()
    src, ws, st = Path(a.source), Path(a.workspace), Path(a.stage)

    sets = {k: read_set(source_file(src, f"set-{k}.tar.gz")) for k in "abc"}
    outs = {k: read_outcomes(source_file(src, f"Outcomes-{k}.txt")) for k in "abc"}
    for k in "abc":
        assert set(sets[k]) == set(outs[k]), k
        print(f"set-{k}: {len(sets[k])} admissions, deaths {sum(outs[k].values())}")

    # neutral ids: one random permutation across all 12,000 so set membership is not visible
    all_ids = sorted(r for k in "abc" for r in sets[k])
    rng = random.Random(a.seed)
    perm = list(range(1, len(all_ids) + 1)); rng.shuffle(perm)
    id_map = {rid: f"P{perm[i]:05d}" for i, rid in enumerate(all_ids)}

    dev = {**sets["a"], **sets["b"]}
    dev_out = {**outs["a"], **outs["b"]}
    write_split(dev, dev_out, id_map, ws / "data", with_outcomes=True)
    (ws / "docs").mkdir(parents=True, exist_ok=True)
    (ws / "docs" / "data_dictionary.md").write_text(DICTIONARY)

    truth = st / "truth"
    write_split(sets["c"], outs["c"], id_map, truth / "setc", with_outcomes=False)
    with open(truth / "setc_outcomes.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["patient_id", "in_hospital_death", "record_id"])
        for rid in sorted(sets["c"], key=lambda r: id_map[r]):
            w.writerow([id_map[rid], outs["c"][rid], rid])
    with open(truth / "dev_outcomes_full.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["patient_id", "in_hospital_death", "record_id", "source_set"])
        for k in "ab":
            for rid in sorted(sets[k], key=lambda r: id_map[r]):
                w.writerow([id_map[rid], outs[k][rid], rid, k])
    with open(truth / "id_map.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["record_id", "patient_id", "source_set"])
        for k in "abc":
            for rid in sorted(sets[k]):
                w.writerow([rid, id_map[rid], k])
    print("workspace:", ws, "| truth:", truth)


if __name__ == "__main__":
    main()
