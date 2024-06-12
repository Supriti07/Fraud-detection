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
    temp_filepath = os.path.join("temp", uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())

    if os.path.exists(temp_filepath):
        st.write("File saved to temporary location:", temp_filepath)
    else:
        st.error("Error: Failed to save file to temporary location")
        st.stop()  # Stop the script execution if the file is not saved successfully

    # Load data
    try:
        data = load_data(temp_filepath)
        st.write("Data Preview:")
        st.dataframe(data.head())
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()  # Stop the script execution if there's an error loading the data

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
