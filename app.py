import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

st.title("Bank Marketing Subscription Prediction")

st.write("Upload a CSV file to evaluate the selected model.")

# Model selection
model_option = st.selectbox(
    "Select Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

# Load model
def load_model(model_name):
    file_name = model_name.replace(" ", "_").lower() + ".pkl"
    with open(file_name, "rb") as file:
        model = pickle.load(file)
    return model

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    if "y" not in data.columns:
        st.error("Target column 'y' not found in dataset.")
    else:
        X = data.drop("y", axis=1)
        y = data["y"]

        model = load_model(model_option)

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        st.subheader("Evaluation Metrics")

        st.write("Accuracy:", accuracy_score(y, predictions))
        st.write("Precision:", precision_score(y, predictions))
        st.write("Recall:", recall_score(y, predictions))
        st.write("F1 Score:", f1_score(y, predictions))
        st.write("AUC:", roc_auc_score(y, probabilities))
        st.write("MCC:", matthews_corrcoef(y, predictions))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, predictions)
        st.write(cm)
