# 🏦 Bank Marketing Subscription Prediction

## 📌 Executive Summary

This project aims to predict whether a customer will subscribe to a term deposit using the Bank Marketing dataset.  
Six machine learning models were implemented and compared using multiple evaluation metrics suitable for imbalanced classification.  
Among all models, **XGBoost achieved the best overall performance**, demonstrating superior predictive capability and robustness.  
A Streamlit web application was developed to allow interactive model evaluation and prediction.

---

## 📌 Problem Statement

The objective of this project is to predict whether a client will subscribe to a term deposit based on historical bank marketing campaign data.  

This is formulated as a **binary classification problem**, where the target variable indicates whether a customer subscribed (`yes`) or not (`no`).

---

## 📊 Dataset Description

- **Dataset**: Bank Marketing Dataset  
- **File Used**: `bank-additional-full.csv`  
- **Total Instances**: 41,188  
- **Input Features**: 20  
- **Target Variable**: `y` (yes/no)  
- **Problem Type**: Binary Classification  

The dataset includes demographic, economic, and campaign-related features describing customer interactions during marketing campaigns.

---

## ⚙️ Data Preprocessing

The following preprocessing steps were applied:

- Target encoding (`yes → 1`, `no → 0`)
- One-hot encoding for categorical features
- Standard scaling for numerical features
- Stratified train-test split to maintain class distribution
- Pipeline-based preprocessing for reproducibility

---

## 🤖 Models Implemented

The following six machine learning models were trained and evaluated:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes  
5. Random Forest  
6. XGBoost  

---

## 📈 Evaluation Metrics

To assess model performance, the following metrics were used:

- Accuracy  
- AUC-ROC  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

Since the dataset is imbalanced, special emphasis was placed on **F1 Score** and **MCC** for reliable model comparison.

---

## 📊 Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| ------- | ---------- | ------ | ----------- | -------- | ---------- | ------ |
| Logistic Regression | 0.916 | 0.942 | 0.709 | 0.434 | 0.539 | 0.514 |
| Decision Tree | 0.895 | 0.741 | 0.531 | 0.543 | 0.537 | 0.478 |
| kNN | 0.903 | 0.877 | 0.598 | 0.436 | 0.505 | 0.460 |
| Naive Bayes | 0.844 | 0.849 | 0.384 | 0.636 | 0.479 | 0.411 |
| Random Forest | 0.916 | 0.947 | 0.674 | 0.490 | 0.568 | 0.531 |
| **XGBoost** | **0.919** | **0.950** | 0.663 | **0.578** | **0.617** | **0.574** |

---

## 🏆 Best Performing Model

**XGBoost** achieved the best overall performance with:

- Highest Accuracy (0.919)  
- Highest AUC (0.950)  
- Highest F1 Score (0.617)  
- Highest MCC (0.574)  

This highlights the effectiveness of gradient boosting techniques on structured tabular datasets, particularly for imbalanced classification problems.

---

## 📌 Model Observations

| Model | Key Observations |
| ------ | ------------------ |
| Logistic Regression | High AUC (0.942) indicating strong class separability. High precision but low recall, meaning many positive cases are missed. Performs well as a linear baseline model. |
| Decision Tree | Lower AUC (0.741) compared to ensemble models. Balanced precision and recall but more prone to overfitting. Offers interpretability advantages. |
| kNN | Moderate AUC (0.877). Sensitive to feature scaling and data distribution. Lower recall reduces effectiveness in detecting minority class. |
| Naive Bayes | Highest recall among simpler models but very low precision, leading to many false positives. Independence assumption limits performance. |
| Random Forest | Strong AUC (0.947) and improved MCC over single tree. Demonstrates stable ensemble performance and reduced overfitting. |
| **XGBoost** | Best overall performance with highest AUC (0.950), F1 Score (0.617), and MCC (0.574). Effectively balances precision and recall on imbalanced data. |

---

## 🚀 Streamlit Application

A Streamlit web application was developed to:

- Upload a test dataset (CSV)
- Select a trained model
- Generate predictions
- Display evaluation metrics
- Show confusion matrix

---

## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Matplotlib  
- Seaborn  

---
