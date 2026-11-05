"""Reproduce model training on BITS Virtual Lab. Run: python train_models.py"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

BASE = Path(__file__).resolve().parent
source = pd.read_csv(BASE / "breast_cancer_dataset.csv")
X, y = source.drop(columns="diagnosis"), source["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
(BASE / "model").mkdir(exist_ok=True)
X_test.assign(diagnosis=y_test).to_csv(BASE / "test_data.csv", index=False)
models = {
    "Logistic Regression": ("logistic_regression.joblib", Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(max_iter=2000, random_state=42))])),
    "Decision Tree": ("decision_tree.joblib", Pipeline([("classifier", DecisionTreeClassifier(max_depth=5, random_state=42))])),
    "KNN": ("knn.joblib", Pipeline([("scale", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))])),
    "Naive Bayes": ("naive_bayes.joblib", Pipeline([("classifier", GaussianNB())])),
    "SVM": ("svm.joblib", Pipeline([("scale", StandardScaler()), ("classifier", SVC(probability=True, random_state=42))])),
}
results = []
for name, (filename, model) in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, BASE / "model" / filename)
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, prediction),
        "AUC": roc_auc_score(y_test, probability),
        "Precision": precision_score(y_test, prediction),
        "Recall": recall_score(y_test, prediction),
        "F1": f1_score(y_test, prediction),
        "MCC": matthews_corrcoef(y_test, prediction),
    })
comparison = pd.DataFrame(results)
comparison.to_csv(BASE / "model_comparison.csv", index=False)
print(comparison.to_string(index=False))
