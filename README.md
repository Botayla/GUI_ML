Machine Learning Project

Overview

This project is a graphical user interface (GUI) application built using Python's tkinter library, designed to facilitate machine learning workflows. It provides a user-friendly interface to load datasets, preprocess data, apply machine learning algorithms, and evaluate model performance.

Features

Data Loading: Load datasets from CSV files using a specified file path.

Data Exploration: Display basic information, statistical summaries, and check for missing values.

Preprocessing:

Handle missing values using SimpleImputer (mean, median, or most frequent strategies).

Encode categorical data using LabelEncoder or OneHotEncoder.

Scale numerical data using MinMaxScaler or StandardScaler.

Machine Learning Algorithms:

Random Forest Classifier

Decision Tree Classifier

Support Vector Machine (SVM)

Evaluation Metrics:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix visualization

Requirements

Python 3.x

Required libraries:

tkinter

pandas

numpy

scikit-learn

imblearn (for SMOTE)

matplotlib

seaborn

Install the dependencies using:

pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

Usage

Run the Application:

Execute the project_ML.ipynb notebook or convert it to a .py file and run it.

Load Data:

Enter the path to your CSV file in the "load data" input field and click "load data".

Explore Data:

Click "some info" to view the dataset's head, info, missing values, and descriptive statistics.

Preprocessing:

Click "preprocessing" to open a window with options for imputation, encoding, and scaling.

Select the method and provide column numbers (if required) then click the corresponding button.

Apply Algorithms:

Click "Algorithm" to open a window, enter the test size (e.g., 0.2), and choose an algorithm (Random Forest, Decision Tree, or SVM).

Evaluate Models:

Click "Evaluation matrix" to open a window and select a metric (Accuracy, Precision, Recall, F1 Score, or Confusion Matrix) to assess the model's performance.

Project Structure

The main script is contained within a single Jupyter Notebook (project_ML.ipynb).

Functions are organized to handle data loading, preprocessing, model training, and evaluation, with a GUI to interact with these functions.

Notes

Ensure the dataset has a valid format with a target column as the last column for proper splitting and scaling.

The application uses global variables to manage data and predictions across functions.

Missing value imputation requires column indices as input (space-separated).

Contributing

Feel free to fork this repository, make improvements, and submit pull requests. Suggestions for enhancing the GUI or adding new algorithms are welcome!

License

This project is open-source. Feel free to use and modify it as needed.
