import joblib
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)

st.set_page_config(page_title="Wisconsin Tumor Classification", page_icon="🔬", layout="wide")
st.title("🔬 Wisconsin Tumor Classification")
st.caption("BITS WILP | ML Assignment 2 | Educational demonstration only — not a medical diagnostic tool")
st.write("Upload the held-out test CSV containing all 30 numerical features and the `diagnosis` column (0 = benign, 1 = malignant).")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}
ROOT = Path(__file__).resolve().parent
@st.cache_resource
def load_model(filename):
    return joblib.load(ROOT / "model" / filename)

@st.cache_data
def expected_columns():
    return pd.read_csv(ROOT / "test_data.csv", nrows=0).columns.tolist()

def score_model(model, X, y):
    prediction = model.predict(X)
    class_position = list(model.classes_).index(1)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, class_position]
    else:
        scores = model.decision_function(X)
        if class_position == 0:
            scores = -scores
    return {
        "Accuracy": accuracy_score(y, prediction),
        "AUC": roc_auc_score(y, scores) if y.nunique() == 2 else float("nan"),
        "Precision": precision_score(y, prediction, zero_division=0),
        "Recall": recall_score(y, prediction, zero_division=0),
        "F1": f1_score(y, prediction, zero_division=0),
        "MCC": matthews_corrcoef(y, prediction),
    }, prediction

file = st.file_uploader("Upload test CSV", type=["csv"])
selected = st.selectbox("Choose classifier", list(MODEL_FILES))
compare = st.checkbox("Show comparison of all five models", value=True)

if file is None:
    st.info("Upload the included test_data.csv to evaluate the saved models. No retraining occurs.")
    st.stop()

try:
    data = pd.read_csv(file)
    columns = expected_columns()
    features = [c for c in columns if c != "diagnosis"]
    missing = [c for c in columns if c not in data.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()
    if data.empty:
        st.error("The CSV contains no records.")
        st.stop()
    X = data.loc[:, features].apply(pd.to_numeric, errors="raise")
    if X.isna().any().any():
        st.error("Input features contain missing values.")
        st.stop()
    y = pd.to_numeric(data["diagnosis"], errors="raise")
    if y.isna().any() or not y.isin([0, 1]).all():
        st.error("Diagnosis must contain only 0 (benign) and 1 (malignant).")
        st.stop()
    y = y.astype(int)
    st.success(f"Validated {len(data)} rows and {len(features)} input features.")
    model = load_model(MODEL_FILES[selected])
    metrics, predictions = score_model(model, X, y)
    st.subheader(f"Results: {selected}")
    cols = st.columns(3)
    for i, (metric, value) in enumerate(metrics.items()):
        cols[i % 3].metric(metric, "N/A" if pd.isna(value) else f"{value:.4f}")
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, predictions, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot(ax=ax, colorbar=False)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Classification report")
        report = classification_report(y, predictions, labels=[0, 1], target_names=["Benign", "Malignant"], zero_division=0)
        st.code(report, language="text")
    if compare:
        st.subheader("All-model comparison on uploaded data")
        rows = []
        for name, filename in MODEL_FILES.items():
            result, _ = score_model(load_model(filename), X, y)
            rows.append({"Model": name, **result})
        st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.4f}"), use_container_width=True)
except (ValueError, TypeError) as exc:
    st.error(f"Invalid CSV content: {exc}")
except Exception as exc:
    st.error(f"Could not evaluate the uploaded file: {exc}")
