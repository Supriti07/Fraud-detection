import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle

filepath = "C:\\Users\\KIIT\\Desktop\\PROJECTS\\FRAUD DETECTION\\modified_Fraud.csv"
def load_data(filepath):
    df = pd.read_csv(filepath)
    df.replace({'type':{'CASH_OUT':1, 'PAYMENT':2, 'CASH_IN':3, 'TRANSFER':4, 'DEBIT':5}}, inplace=True)
    df.columns = df.columns.str.replace('\u00A0', '')
    columns_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
    df = df.drop(columns_to_drop, axis=1)
    
    # Handle missing values in the target variable
    print("Before handling missing values:")
    print("Shape of y:", df['isFraud'].shape)
    print("Data type of y:", df['isFraud'].dtype)
    print("Unique values of y:", df['isFraud'].unique())
    
    df['isFraud'].fillna(df['isFraud'].mean(), inplace=True)
    
    # Convert target variable to integer type
    df['isFraud'] = df['isFraud'].astype(int)
    
    print("\nAfter handling missing values:")
    print("Shape of y:", df['isFraud'].shape)
    print("Data type of y:", df['isFraud'].dtype)
    print("Unique values of y:", df['isFraud'].unique())
    
    return df

def preprocess_data(df):
    x = df.drop(['isFraud'], axis=1)
    y = df['isFraud']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=69)
    return x_train, x_test, y_train, y_test

def train_decision_tree(x_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    return model

def train_random_forest(x_train, y_train):
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    return rfc

def train_xgboost(x_train, y_train):
    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.fit(x_train, y_train)
    return xgb_classifier

def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

# Load data
data = load_data(filepath)

# Preprocess data
x_train, x_test, y_train, y_test = preprocess_data(data)

# Train decision tree model
model = train_decision_tree(x_train, y_train)

# Save model
save_model(model, 'decision_tree_model.pkl')
print("Decision Tree model trained and saved!")
