# Bank Marketing Subscription Prediction

📌 Problem Statement

The objective of this project is to predict whether a client will subscribe to a term deposit based on the Bank Marketing dataset using multiple machine learning classification models.

📊 Dataset Description

Source: UCI Bank Marketing Dataset

File used: bank-additional-full.csv

Instances: 41,188

Features: 20 input features + 1 target variable

Target Variable: y (yes/no – subscription status)

⚙️ Preprocessing Steps

One-hot encoding for categorical variables

Standard scaling for numerical features

Stratified train-test split

Pipeline-based preprocessing for reproducibility

🤖 Models Implemented

Logistic Regression

Decision Tree

K-Nearest Neighbors

Naive Bayes

Random Forest

XGBoost

📈 Evaluation Metrics

The following metrics were used:

Accuracy

AUC-ROC

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

🛠 Tech Stack

Python

Scikit-learn

XGBoost

Pandas

NumPy

Streamlit
