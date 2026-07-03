# Kaggle Student Health Risk Baseline

This project is a practice solution for the Kaggle Playground Series S6E7 competition:

https://www.kaggle.com/competitions/playground-series-s6e7

The goal is to predict a student's `health_condition` from lifestyle and health-related features such as sleep duration, heart rate, BMI, calorie expenditure, step count, diet type, stress level, sleep quality, physical activity level, smoking/alcohol habits, and gender.

## Task Type

This is a multiclass classification task.

Target column:

```text
health_condition
```

Classes:

```text
at-risk
fit
unhealthy
```

The target distribution is imbalanced. In the training data, `at-risk` is the majority class, while `fit` and `unhealthy` are minority classes. Therefore, this project reports both normal accuracy and balanced accuracy.

## Project Structure

```text
kaggle-student-health-risk/
├── data/
│   ├── raw/              # Kaggle raw data, not uploaded to GitHub
│   └── submissions/      # Generated submission files and experiment log
├── src/
│   ├── baseline.py       # Baseline training and submission script
│   └── demo.py           # Small script for comparing submission files
├── README.md
└── .gitignore
```

## Data

Download the data from Kaggle:

https://www.kaggle.com/competitions/playground-series-s6e7/data

After downloading and extracting the zip file, place the files here:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

The raw Kaggle data files are not included in this repository. They should be downloaded from Kaggle directly.

## Baseline Method

The current baseline uses:

```text
Pandas for data processing
RandomForestClassifier for modeling
train_test_split with stratify for validation
accuracy_score and balanced_accuracy_score for evaluation
classification_report for per-class performance
```

Preprocessing steps:

1. Read `train.csv`, `test.csv`, and `sample_submission.csv`
2. Separate the target column `health_condition`
3. Drop the `id` column before training
4. Identify numerical and categorical columns
5. Fill numerical missing values with the training median
6. Fill categorical missing values with the training mode
7. Apply one-hot encoding with `pd.get_dummies`
8. Align train and test feature columns
9. Split train/validation data with stratified sampling
10. Train a Random Forest model
11. Generate a Kaggle submission file
12. Save an experiment log

Current model:

```python
RandomForestClassifier(
    n_estimators=50,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
```

## Current Result

The baseline submission score on Kaggle is:

```text
0.89510
```

Local validation result from the current baseline script:

```text
accuracy: 0.895057
balanced accuracy: 0.903854
```

## How to Run

Install dependencies:

```bash
pip install pandas scikit-learn
```

Run the baseline script:

```bash
python src/baseline.py
```

The script will generate:

```text
data/submissions/rf_balanced_depth12.csv
data/submissions/experiment_log.csv
```

Upload the generated CSV file to Kaggle for scoring.

## Experiment Tracking

Each run writes an experiment record to:

```text
data/submissions/experiment_log.csv
```

The log records:

```text
experiment name
accuracy
balanced accuracy
model name
main model parameters
submission file name
```

When changing model parameters, update this line in `src/baseline.py`:

```python
experiment_name = "rf_balanced_depth12"
```

For example:

```python
experiment_name = "rf_no_weight_depth12"
```

This prevents different submissions from being mixed up.

## Notes

This is a beginner-friendly baseline project. The first goal is not to reach the best leaderboard score, but to build a complete machine learning workflow:

```text
load data -> preprocess data -> train model -> validate model -> generate submission -> record experiment
```

Future improvements may include:

```text
StratifiedKFold cross-validation
LightGBM / XGBoost / CatBoost models
target encoding
feature engineering
model ensembling
```
