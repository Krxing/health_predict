from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
submission_dir = project_dir / "data" / "submissions"

baseline = pd.read_csv(submission_dir / "submission_baseline.csv")
v1 = pd.read_csv(submission_dir / "submission_v1.csv")

print("baseline 预测分布：")
print(baseline["health_condition"].value_counts(normalize=True))

print("v1 预测分布：")
print(v1["health_condition"].value_counts(normalize=True))