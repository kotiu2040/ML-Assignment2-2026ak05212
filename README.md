# Machine Learning Assignment 2
## Breast Cancer Wisconsin (Diagnostic) — Classification Model Comparison

**Course:** Machine Learning  
**Programme:** M.Tech in Artificial Intelligence and Machine Learning  
**Institution:** BITS Pilani — Work Integrated Learning Programmes

## A. Problem Statement

The objective of this assignment is to build and compare five machine learning classification models using the Breast Cancer Wisconsin (Diagnostic) dataset.

The dataset contains numerical measurements of cell nuclei obtained from breast mass samples. Based on these measurements, the models predict whether a sample is benign or malignant.

The assignment involves data preprocessing, model training, performance evaluation, and deployment of an interactive web application using Streamlit.

The main goal is to compare different classification algorithms on the same dataset and identify which model performs best based on the evaluation metrics.

## B. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** UCI Machine Learning Repository  
**Dataset link:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The dataset contains 569 observations and 30 numerical input features. These features describe characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

### Dataset Summary

| Description | Value |
|---|---|
| Total records | 569 |
| Input features | 30 |
| Classification type | Binary |
| Benign samples | 357 |
| Malignant samples | 212 |
| Missing values | 0 |
| Training records | 455 |
| Testing records | 114 |

The target variable is encoded as follows:

- **0:** Benign
- **1:** Malignant

The original target encoding from Scikit-learn was reversed to make malignant the positive class during evaluation.

### Data Preprocessing

I performed the following preprocessing steps:

1. Loaded the dataset and examined its structure.
2. Checked the data for missing values and duplicate records.
3. Separated the input features and target variable.
4. Split the dataset into 80% training data and 20% testing data using stratified sampling.
5. Applied StandardScaler to Logistic Regression, KNN, and SVM through Scikit-learn pipelines.

A fixed random state of 42 was used to make the train-test split reproducible.

The preprocessing was included in the model pipelines so that the scaling parameters were learned only from the training data. This prevents data leakage from the test dataset.

## C. GitHub Repository Link

**GitHub:** [Insert your actual GitHub repository URL]

The repository contains the Python source code, dataset, test CSV file, saved classification models, requirements file, and Streamlit application.

### Project Files

| File | Description |
|---|---|
| app.py | Streamlit application for model evaluation |
| train_models.py | Python script for training and evaluating the classifiers |
| requirements.txt | Python dependencies required to run the project |
| breast_cancer_dataset.csv | Complete dataset |
| test_data.csv | Held-out test dataset containing 114 records |
| model_comparison.csv | Evaluation results for all five models |
| model/ | Saved trained models in Joblib format |
| README.md | Project documentation |

## D. Models Used and Evaluation

Five classification algorithms were implemented using the same dataset and train-test split.

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models were evaluated using six metrics: Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

### Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| SVM | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

These results were obtained using the held-out test dataset containing 114 records.

### Model Performance Observations

**1. Logistic Regression**

Logistic Regression achieved an accuracy of 96.49% and an F1 score of 0.9512. Its AUC score of 0.9960 was the highest among the five models.

The model performed well despite using a relatively simple linear classification approach. Its malignant recall of 92.86% was also among the highest in the comparison.

**2. Decision Tree Classifier**

The Decision Tree achieved an accuracy of 92.11%, which was the lowest among the five models.

Its recall was 83.33%, indicating that it missed more malignant samples than Logistic Regression and SVM. Although decision trees are easy to interpret, this model did not perform as well as the other classifiers on the selected test dataset.

**3. K-Nearest Neighbors (KNN)**

KNN achieved an accuracy of 95.61% and an F1 score of 0.9383.

The model produced good results after feature standardization. However, its performance was slightly lower than Logistic Regression and SVM, particularly in terms of malignant recall and MCC.

**4. Gaussian Naive Bayes**

Gaussian Naive Bayes achieved an accuracy of 93.86% and a precision of 1.0000.

This means that all samples predicted as malignant were actually malignant in the test dataset. However, its recall was only 83.33%, showing that some malignant samples were incorrectly classified as benign.

The model performed reasonably well despite its simplifying assumption that features are conditionally independent.

**5. Support Vector Machine (SVM)**

SVM achieved the highest accuracy of 97.37%, the highest F1 score of 0.9630, and the highest MCC score of 0.9442.

It also achieved a precision of 1.0000 and a malignant recall of 92.86%. This indicates that the model performed well in identifying malignant samples while avoiding false positive predictions in this test set.

The RBF kernel allowed the classifier to model nonlinear relationships between the input features.

### Overall Best Performing Model

**Support Vector Machine (SVM)** was the best-performing model on this dataset based on the overall evaluation results.

SVM achieved:

- Accuracy: 97.37%
- AUC: 0.9947
- Precision: 100%
- Recall: 92.86%
- F1 Score: 0.9630
- MCC Score: 0.9442

Although Logistic Regression achieved a slightly higher AUC score, SVM performed better in accuracy, F1 score, and MCC.

Based on these results, I selected SVM as the overall winner for this experiment.

One important observation is that both SVM and Logistic Regression achieved the same malignant recall. Therefore, SVM's higher overall score does not mean it detected more malignant cases in this test set.

## E. Streamlit Application

An interactive web application was developed using Streamlit to evaluate the trained classification models.

**Live Application:** https://ml-assignment2-2026ak05212-s2fafxmlzybdzyyq2wfve5.streamlit.app/

The application provides the following features:

1. Upload a CSV file containing the test dataset.
2. Select one of the five trained classification models.
3. Display Accuracy, AUC, Precision, Recall, F1, and MCC.
4. Display a confusion matrix and classification report.
5. Compare the performance of all five models.

The application loads the previously trained models from the saved Joblib files.

No model retraining is performed when the user uploads the test dataset. The uploaded CSV is used only for prediction and evaluation.

## F. Execution and Deployment

The project was executed using Python and Scikit-learn.

The training script was used to prepare the classification models and save them for subsequent evaluation.

The Streamlit application was also executed inside BITS Virtual Lab. The application successfully loaded the test CSV containing 114 records and 30 input features and displayed the classification results.

The completed application was deployed on Streamlit Community Cloud using the GitHub repository.

### Steps to Run the Project

**1. Install the dependencies**

```bash
python -m pip install -r requirements.txt
```

**2. Train and save the models**

```bash
python train_models.py
```

**3. Start the Streamlit application**

```bash
python -m streamlit run app.py
```

**4. Evaluate the models**

Upload `test_data.csv`, select a classifier, and review the evaluation metrics and confusion matrix.

## G. Conclusion

In this assignment, I implemented and compared five classification algorithms using the Breast Cancer Wisconsin dataset.

All five models produced reasonably good results, although there were noticeable differences in their classification performance.

Logistic Regression and SVM performed better than the other models on most evaluation metrics. SVM achieved the highest accuracy, F1 score, and MCC, making it the best-performing model for this experiment.

The results also showed why it is important to consider multiple evaluation metrics rather than relying only on accuracy. In particular, recall is important for understanding how many malignant samples were correctly identified.

The Streamlit application provides a simple way to upload test data, select different models, and compare their results without retraining.

Overall, this assignment helped me understand the complete machine learning workflow, from data preprocessing and model evaluation to deployment of a working classification application.

---

**Note:** This project is intended for educational purposes only. The classification results should not be used for medical diagnosis.