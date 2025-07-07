import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Initialization of variables
data = None
x_train = None
x_test = None
y_train = None
y_test = None
y_pred = None

# Function to load dataset
def load_dataset(uploaded_file):
    global data
    data = pd.read_csv(uploaded_file)
    return data

# Function to display dataset info
def info():
    st.subheader("Dataset Information")
    st.write("**Data Preview (First 5 Rows):**")
    st.dataframe(data.head())
    st.write("**Data Info:**")
    buffer = pd.DataFrame(data.dtypes, columns=['Data Type'])
    st.write(buffer)
    st.write("**Missing Values:**")
    st.write(data.isnull().sum())
    st.write("**Statistical Summary:**")
    st.write(data.describe())

# Function for SimpleImputer
def simple_imputer(method, col_entry):
    global data
    try:
        cols = [int(c) for c in col_entry.split()]
        imputer = SimpleImputer(strategy=method)
        for col in cols:
            data.iloc[:, col] = imputer.fit_transform(data.iloc[:, col].values.reshape(-1, 1))
        st.success(f"Imputation with {method} completed!")
        return data
    except Exception as e:
        st.error(f"Error in imputation: {str(e)}")
        return data

# Function for OneHotEncoder
def one_hot_encoder():
    global data
    try:
        cols = data.columns[data.dtypes == object].tolist()
        if cols:
            encoder = OneHotEncoder(sparse_output=False)
            encoded_data = encoder.fit_transform(data[cols])
            encoded_cols = encoder.get_feature_names_out(cols)
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols)
            data = pd.concat([encoded_df, data.drop(columns=cols)], axis=1)
            st.success("One-Hot Encoding completed!")
        else:
            st.warning("No categorical columns found for encoding.")
        return data
    except Exception as e:
        st.error(f"Error in One-Hot Encoding: {str(e)}")
        return data

# Function for LabelEncoder
def label_encoder(col_entry):
    global data
    try:
        le = LabelEncoder()
        data[col_entry] = le.fit_transform(data[col_entry])
        st.success(f"Label Encoding on column '{col_entry}' completed!")
        return data
    except Exception as e:
        st.error(f"Error in Label Encoding: {str(e)}")
        return data

# Function for MinMaxScaler
def minmax_scaler():
    global data
    try:
        scaler = MinMaxScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        st.success("MinMax Scaling completed!")
        return data
    except Exception as e:
        st.error(f"Error in MinMax Scaling: {str(e)}")
        return data

# Function for StandardScaler
def standard_scaler():
    global data
    try:
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        st = StandardScaler()
        x = pd.DataFrame(st.fit_transform(x), columns=x.columns)
        data = pd.concat([x, y.reset_index(drop=True)], axis=1)
        st.success("Standard Scaling completed!")
        return data
    except Exception as e:
        st.error(f"Error in Standard Scaling: {str(e)}")
        return data

# Function to split data
def split(test_size):
    global x_train, x_test, y_train, y_test
    try:
        x = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(test_size), random_state=1)
        st.success("Data split completed!")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        st.error(f"Error in splitting data: {str(e)}")
        return None, None, None, None

# Function for Random Forest
def random_forest(test_size):
    global y_pred
    try:
        x_train, x_test, y_train, y_test = split(test_size)
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        st.success("Random Forest training completed!")
    except Exception as e:
        st.error(f"Error in Random Forest: {str(e)}")

# Function for Decision Tree
def decision_tree(test_size):
    global y_pred
    try:
        x_train, x_test, y_train, y_test = split(test_size)
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        st.success("Decision Tree training completed!")
    except Exception as e:
        st.error(f"Error in Decision Tree: {str(e)}")

# Function for SVM
def svc(test_size):
    global y_pred
    try:
        x_train, x_test, y_test = split(test_size)
        model = SVC()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        st.success("SVM training completed!")
    except Exception as e:
        st.error(f"Error in SVM: {str(e)}")

# Function for Accuracy
def accuracy():
    global y_pred, y_test
    try:
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {acc:.4f}")
    except Exception as e:
        st.error(f"Error in calculating accuracy: {str(e)}")

# Function for Precision
def precision():
    global y_pred, y_test
    try:
        pre = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        st.write(f"**Precision:** {pre:.4f}")
    except Exception as e:
        st.error(f"Error in calculating precision: {str(e)}")

# Function for F1 Score
def f1_measure():
    global y_pred, y_test
    try:
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        st.write(f"**F1 Score:** {f1:.4f}")
    except Exception as e:
        st.error(f"Error in calculating F1 score: {str(e)}")

# Function for Recall
def recall():
    global y_pred, y_test
    try:
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        st.write(f"**Recall:** {rec:.4f}")
    except Exception as e:
        st.error(f"Error in calculating recall: {str(e)}")

# Function for Confusion Matrix
def confusion_matrix_plot():
    global y_pred, y_test
    try:
        cm = confusion_matrix(y_test, y_pred)
        labels = np.unique(y_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16}, 
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error in plotting confusion matrix: {str(e)}")

# Streamlit App Layout
st.title("Machine Learning Project")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = load_dataset(uploaded_file)
    st.success("Dataset loaded successfully!")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Dataset Info", "Preprocessing", "Algorithms", "Evaluation Metrics"])

# Dataset Info Section
if section == "Dataset Info":
    if data is not None:
        if st.button("Show Dataset Info"):
            info()
    else:
        st.warning("Please upload a dataset first.")

# Preprocessing Section
elif section == "Preprocessing":
    if data is not None:
        st.subheader("Preprocessing Options")
        
        # Simple Imputer
        st.write("**Simple Imputer**")
        method = st.selectbox("Select Imputation Method", ["mean", "median", "most_frequent"])
        col_entry = st.text_input("Enter column numbers (space-separated, e.g., '0 1 2')")
        if st.button("Apply Imputer"):
            data = simple_imputer(method, col_entry)
            st.write("Updated Data Preview:", data.head())

        # Label Encoder
        st.write("**Label Encoder**")
        label_col = st.text_input("Enter column name for Label Encoding")
        if st.button("Apply Label Encoder"):
            data = label_encoder(label_col)
            st.write("Updated Data Preview:", data.head())

        # One-Hot Encoder
        if st.button("Apply One-Hot Encoder"):
            data = one_hot_encoder()
            st.write("Updated Data Preview:", data.head())

        # MinMax Scaler
        if st.button("Apply MinMax Scaler"):
            data = minmax_scaler()
            st.write("Updated Data Preview:", data.head())

        # Standard Scaler
        if st.button("Apply Standard Scaler"):
            data = standard_scaler()
            st.write("Updated Data Preview:", data.head())
    else:
        st.warning("Please upload a dataset first.")

# Algorithms Section
elif section == "Algorithms":
    if data is not None:
        st.subheader("Model Training")
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        
        if st.button("Train Random Forest"):
            random_forest(test_size)
        
        if st.button("Train Decision Tree"):
            decision_tree(test_size)
        
        if st.button("Train SVM"):
            svc(test_size)
    else:
        st.warning("Please upload a dataset first.")

# Evaluation Metrics Section
elif section == "Evaluation Metrics":
    if y_pred is not None and y_test is not None:
        st.subheader("Evaluation Metrics")
        if st.button("Calculate Accuracy"):
            accuracy()
        
        if st.button("Calculate Precision"):
            precision()
        
        if st.button("Calculate F1 Score"):
            f1_measure()
        
        if st.button("Calculate Recall"):
            recall()
        
        if st.button("Show Confusion Matrix"):
            confusion_matrix_plot()
    else:
        st.warning("Please train a model first.")