"""
generate_data.py  —  Run this first to create the dataset.
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 600
os.makedirs("data", exist_ok=True)

study_hours    = np.round(np.random.uniform(1, 12, N), 1)
attendance     = np.round(np.random.uniform(40, 100, N), 1)
previous_score = np.round(np.random.uniform(25, 95, N), 1)
sleep_hours    = np.round(np.random.uniform(4, 10, N), 1)
internet       = np.random.choice([0, 1], N, p=[0.25, 0.75])
par_edu_map    = {0:"None",1:"Primary",2:"Secondary",3:"Bachelor",4:"Master"}
par_edu_num    = np.random.choice([0,1,2,3,4], N, p=[0.05,0.12,0.33,0.32,0.18])
extracurricular= np.random.choice([0, 1], N, p=[0.38, 0.62])

# Score formula tuned so ~30% fail (realistic classroom setting)
scores = (
      study_hours    * 3.8
    + attendance     * 0.20
    + previous_score * 0.28
    + sleep_hours    * 1.0
    + internet       * 3.0
    + par_edu_num    * 1.5
    + extracurricular* 1.2
    + np.random.normal(0, 5, N)
)
# Normalise to 0-100 range
scores = np.clip(scores, scores.min(), scores.max())
scores = (scores - scores.min()) / (scores.max() - scores.min()) * 85 + 15
scores = np.round(scores, 1)

pass_fail = ["Pass" if s >= 50 else "Fail" for s in scores]

df = pd.DataFrame({
    "student_id":         [f"STU{str(i).zfill(4)}" for i in range(1, N+1)],
    "study_hours":        study_hours,
    "attendance_percent": attendance,
    "previous_score":     previous_score,
    "sleep_hours":        sleep_hours,
    "internet_access":    ["Yes" if x else "No" for x in internet],
    "parental_education": [par_edu_map[i] for i in par_edu_num],
    "extracurricular":    ["Yes" if x else "No" for x in extracurricular],
    "final_score":        scores,
    "pass_fail":          pass_fail,
})

# ~5% missing values in 3 columns
for col in ["study_hours", "sleep_hours", "attendance_percent"]:
    idx = np.random.choice(df.index, size=int(N*0.05), replace=False)
    df.loc[idx, col] = np.nan

df.to_csv("data/student_data.csv", index=False)
print(f"✅  dataset saved  ({N} rows)")
print(f"    Pass: {(df.pass_fail=='Pass').sum()}  |  Fail: {(df.pass_fail=='Fail').sum()}")
print(f"    Score range: {df.final_score.min()} – {df.final_score.max()}")
