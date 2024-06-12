import streamlit as st
import pandas as pd
from model import load_data, preprocess_data, train_decision_tree, train_random_forest, train_xgboost, save_model, load_model
from sklearn.metrics import accuracy_score, classification_report
import os

st.title('Fraud Detection Model Training and Evaluation')

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with open(os.path.join("temp", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getvalue())

    data = load_data(os.path.join("temp", uploaded_file.name))
    st.write("Data Preview:")
    st.dataframe(data.head())

    x_train, x_test, y_train, y_test = preprocess_data(data)

    if st.button('Train Decision Tree'):
        model = train_decision_tree(x_train, y_train)
        save_model(model, 'decision_tree_model.pkl')
        st.write("Decision Tree model trained and saved!")

    if st.button('Train Random Forest'):
        model = train_random_forest(x_train, y_train)
        save_model(model, 'random_forest_model.pkl')
        st.write("Random Forest model trained and saved!")

    if st.button('Train XGBoost'):
        model = train_xgboost(x_train, y_train)
        save_model(model, 'xgboost_model.pkl')
        st.write("XGBoost model trained and saved!")

    if st.button('Evaluate Models'):
        models = {
            'Decision Tree': 'decision_tree_model.pkl',
            'Random Forest': 'random_forest_model.pkl',
            'XGBoost': 'xgboost_model.pkl'
        }

        for model_name, model_path in models.items():
            model = load_model(model_path)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            st.write(f"### {model_name} Model")
            st.write(f"**Accuracy:** {accuracy}")
            st.text(f"**Classification Report:**\n{report}")

# Optional: Add more UI components as needed
