# ML-Assignment2-2026ak05212
Repo for BITS ML Assignment2
# Breast Cancer Wisconsin (Diagnostic)

> **Student action required:** Execute this project in BITS Virtual Lab, verify the outputs, add your own analysis, and capture an authentic lab screenshot. AI-generated scaffolding is for learning and adaptation, not direct copy-paste submission.

## A. Problem statement

Predict whether a cell-nucleus sample is benign (0) or malignant (1) from numerical features. Educational use only; not suitable for clinical decisions.

## B. Dataset description

- Source: UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic): https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- 569 observations, 30 numeric features, binary diagnosis target.
- Benign: 357; malignant: 212. No missing values in the distributed dataset.
- Original scikit-learn labels (0 malignant, 1 benign) are **reversed** here to (0 benign, 1 malignant).
- Reproducible stratified 80/20 split (`random_state=42`), yielding 455 training and 114 test rows.
- StandardScaler is inside training pipelines for Logistic Regression, KNN and SVM to avoid test-set leakage.

## C. GitHub repository link

**Replace this text with your actual GitHub repository URL after uploading.**

## D. Models used and evaluation

All five models use the same held-out test set. AUC is calculated using predicted positive-class probabilities; other metrics use predicted labels. Malignant (1) is the positive class.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| SVM | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Model observations — interpret and rewrite after running on BITS Virtual Lab

- **Logistic Regression:** A linear baseline; standardization helps optimization. Compare its malignant recall and MCC with nonlinear methods.
- **Decision Tree:** Interpretable split rules but can overfit; maximum depth is restricted to five.
- **KNN:** Distance-based and sensitive to feature scale; uses standardized inputs and k=5.
- **Naive Bayes:** Fast probabilistic baseline with strong conditional-independence assumptions.
- **SVM:** Nonlinear RBF decision boundary with standardized features; probability estimates are enabled.

**Preliminary winner (highest held-out MCC): SVM**, MCC=0.9442. Review false negatives and malignant recall before selecting your final winner. These are results generated in a separate environment and must be verified in BITS Virtual Lab.

## Running locally or in BITS Virtual Lab

1. Install dependencies: `python -m pip install -r requirements.txt` (or `%pip install -r requirements.txt` in Jupyter).
2. Open `ML_Assignment_2.ipynb` and execute all cells, or run `python train_models.py`.
3. Confirm that five `.joblib` files and `test_data.csv` are present.
4. Launch Streamlit: `streamlit run app.py`.
5. Upload `test_data.csv`, select each model and check the metrics and confusion matrix. The app **never retrains** using uploaded data.

## Deployment

Push the repository to GitHub. In https://streamlit.io/cloud create a new app pointing at the repository's `main` branch and `app.py`. Confirm that the deployment works with `test_data.csv`.

**Live Streamlit URL:** Replace with your deployed URL.

## Final PDF submission order

1. GitHub repository URL (clickable).
2. Deployed Streamlit app URL (clickable).
3. Screenshot showing **your actual execution on BITS Virtual Lab**.
4. Entire updated README content, including comparison and personal observations.

## Notes

- Never upload confidential data. This repository contains public educational data.
- Saved `joblib` models should only be loaded from trusted sources.
- Package versions should be consistent when training and deploying saved models.
- Check results after training in the required lab environment; model output can differ by package version.
