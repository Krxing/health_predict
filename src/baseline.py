
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report


# 1. 路径设置
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data" / "raw"
submission_dir = project_dir / "data" / "submissions"
submission_dir.mkdir(parents=True, exist_ok=True)

experiment_name = "rf_balanced_depth12"



# 2. 读取数据
train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")
sample_submission = pd.read_csv(data_dir / "sample_submission.csv")


# 3. 设置目标列
target = "health_condition"

X = train.drop(columns=[target])
y = train[target]

test_id = test["id"]

'''下面的代码用于观察数据分布'''
# target = "health_condition"
#
# print("目标列分布：")
# print(train[target].value_counts())
# print(train[target].value_counts(normalize=True))
#
# num_cols = train.select_dtypes(include=["int64", "float64"]).columns.tolist()
# cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
#
# num_cols.remove("id")
# cat_cols.remove(target)
#
# print("数值列：", num_cols)
# print("类别列：", cat_cols)
#
# print("缺失值比例：")
# print(train.isnull().mean().sort_values(ascending=False))
#
# print("数值列统计：")
# print(train[num_cols].describe())
#
# for col in cat_cols:
#     print(col)
#     print(train[col].value_counts(dropna=False))
#     print("-" * 30)


# 4. 删除 id，id 只是编号，不参与训练
X = X.drop(columns=["id"])
test = test.drop(columns=["id"])


# 5. 区分数值列和类别列
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("数值列：", num_cols)
print("类别列：", cat_cols)


# 6. 缺失值处理
# 数值列：用训练集的中位数填充
for col in num_cols:
    median_value = X[col].median()
    X[col] = X[col].fillna(median_value)
    test[col] = test[col].fillna(median_value)

# 类别列：用训练集的众数填充
for col in cat_cols:
    mode_value = X[col].mode()[0]
    X[col] = X[col].fillna(mode_value)
    test[col] = test[col].fillna(mode_value)


# 7. 类别特征 one-hot 编码
X = pd.get_dummies(X)
test = pd.get_dummies(test)

# 保证 train 和 test 编码后的列完全一致
X, test = X.align(test, join="left", axis=1, fill_value=0)


# 8. 划分训练集和验证集
# stratify=y 是因为你的类别不平衡，保证划分后类别比例差不多
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 9. 训练基础模型
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# 10. 验证集评估
val_pred = model.predict(X_val)

acc = accuracy_score(y_val, val_pred)
balanced_acc = balanced_accuracy_score(y_val, val_pred)

print("验证集 accuracy：", acc)
print("验证集 balanced accuracy：", balanced_acc)
print(classification_report(y_val, val_pred))

print("验证集真实分布：")
print(y_val.value_counts(normalize=True))

print("验证集预测分布：")
print(pd.Series(val_pred).value_counts(normalize=True))


# 11. 预测测试集
test_pred = model.predict(test)


# 12. 生成 Kaggle 提交文件
submission = sample_submission.copy()
submission["health_condition"] = test_pred

print("测试集预测分布：")
print(pd.Series(test_pred).value_counts(normalize=True))

output_path = submission_dir / f"{experiment_name}.csv"
submission.to_csv(output_path, index=False)

log_path = submission_dir / "experiment_log.csv"
log_row = pd.DataFrame([{
    "experiment": experiment_name,
    "accuracy": acc,
    "balanced_accuracy": balanced_acc,
    "model": "RandomForest",
    "n_estimators": 50,
    "max_depth": 12,
    "class_weight": "balanced",
    "submission_file": output_path.name
}])

if log_path.exists():
    old_log = pd.read_csv(log_path)
    log = pd.concat([old_log, log_row], ignore_index=True)
else:
    log = log_row

log.to_csv(log_path, index=False)

print("实验记录已保存：", log_path)
print("提交文件已生成：", output_path)

