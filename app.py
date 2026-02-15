import streamlit as st
import pandas as pd
import pickle
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)


st.set_page_config(
    page_title="Bank Marketing Subscription Prediction",
    layout="wide"
)

st.title("Bank Marketing Subscription Prediction")
st.write("Upload a CSV file containing test data to evaluate a selected model, or use the below sample dataset for quick testing.")

file_path = "data/test_data.csv"

if os.path.exists(file_path):
    with open(file_path, "rb") as file:
        st.download_button(
            label="Download test_set.csv",
            data=file,
            file_name="test_set.csv",
            mime="text/csv",
            
        )
else:
    st.error("Test dataset not found in repository.")

model_options = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "kNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

selected_model_name = st.selectbox("Select Model", list(model_options.keys()))

@st.cache_resource
def load_model(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file, sep=";")

        st.subheader("Uploaded Data Preview")
        st.dataframe(data.head())

        if "y" not in data.columns:
            st.error("Target column 'y' not found in uploaded dataset.")
        else:
            # Encoding target same way as training
            data["y"] = data["y"].map({"yes": 1, "no": 0})

            X = data.drop("y", axis=1)
            y = data["y"]

            # Load selected model
            model_path = model_options[selected_model_name]
            model = load_model(model_path)

            # Predictions
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

            # Metrics

            accuracy = accuracy_score(y, predictions)
            precision = precision_score(y, predictions)
            recall = recall_score(y, predictions)
            f1 = f1_score(y, predictions)
            auc = roc_auc_score(y, probabilities)
            mcc = matthews_corrcoef(y, predictions)

            st.subheader("📊 Evaluation Metrics")

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", round(accuracy, 4))
            col2.metric("Precision", round(float(precision), 4))
            col3.metric("Recall", round(float(recall), 4))

            col4, col5, col6 = st.columns(3)
            col4.metric("F1 Score", round(float(f1), 4))
            col5.metric("AUC", round(float(auc), 4))
            col6.metric("MCC", round(float(mcc), 4))

            # ------------------------------
            # Confusion Matrix
            # ------------------------------

            st.subheader("🔍 Confusion Matrix")
            cm = confusion_matrix(y, predictions)

            cm_df = pd.DataFrame(
                cm,
                index=["Actual No", "Actual Yes"],
                columns=["Predicted No", "Predicted Yes"]
            )

            st.dataframe(cm_df)

            # ------------------------------
            # Prediction Output
            # ------------------------------

            st.subheader("📌 Sample Predictions")
            output_df = X.copy()
            output_df["Actual"] = y
            output_df["Predicted"] = predictions

            st.dataframe(output_df.head())

    except Exception as e:
        st.error(f"Error processing file: {e}")
