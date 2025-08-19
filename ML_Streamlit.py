import streamlit as st
import io
# from ydata_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler,RobustScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor
from sklearn.svm import SVC , SVR 
from sklearn.cluster import KMeans,DBSCAN , AgglomerativeClustering
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix ,r2_score,mean_squared_error,mean_absolute_error
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
import base64

st.set_page_config(
    page_title="Machine Learning 🤖",
    page_icon="🤖",
    layout="wide"
)

def set_bg(path):
    with open(path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/png;base64,{encoded});
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("download.jfif")

# ================== Helper functions ==================
def drop_columns(cols):
    data = st.session_state.data.copy()
    data = data.drop(columns=cols)
    st.session_state.data = data

def auto_imputer(strategy_num="mean"):
    data = st.session_state.data.copy()
    # Numeric columns
    num_cols = data.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy=strategy_num)
        data[num_cols] = imputer_num.fit_transform(data[num_cols])

    # Categorical columns
    cat_cols = data.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        data[cat_cols] = imputer_cat.fit_transform(data[cat_cols])

    st.session_state.data = data
def handle_outliers(method = "IQR" ):
    data = st.session_state.data.copy()
    cols = data.select_dtypes(include =['int','float'] ).columns
    for col in cols:
        if method == "IQR" :
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            data[col] = np.where(data[col]<lower , lower ,
                                 np.where(data[col]>upper , upper,data[col]))
            
        elif method == "Z_score":
            mean= data[col].mean()
            std = data[col].std()
            z_scores = (data[col] - mean) / std
            data = data[(z_scores > -3) & (z_scores < 3 ) ]

        elif method == "log"  :
            data[col] = np.log1p(data[col])

    st.session_state.data = data

def one_hot_encoder(cols):
    data = st.session_state.data.copy()
    # cat_cols = data.select_dtypes(include=["object"]).columns
    encoder = OneHotEncoder(sparse_output=False)
    encoded = pd.DataFrame(encoder.fit_transform(data[cols]),
                           columns=encoder.get_feature_names_out(cols), index=data.index)
    # encoded.columns = encoder.get_feature_names_out(cols)
    data = pd.concat([data.drop(columns=cols), encoded], axis=1)
    st.session_state.data = data

def label_encoder(cols):
    data = st.session_state.data.copy()
    le = LabelEncoder()
    for col in cols:
        data[col] = le.fit_transform(data[col])
    st.session_state.data = data
def ordinal_encoder(cols):
    data = st.session_state.data.copy()
    encoder = OrdinalEncoder()
    data[cols] = encoder.fit_transform(data[cols])
    st.session_state.data = data

def minmax_scaler(target_colum=None):
    data = st.session_state.data.copy()
    if target_colum is not None:
        x = data.drop(target_colum, axis=1)
        y = data[target_colum]
        scaler = MinMaxScaler()
        x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
        data = pd.concat([x, y], axis=1)
    else:
        scaler = MinMaxScaler()
        x = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        data = x
    st.session_state.data = data

def standard_scaler(target_colum=None):
    data = st.session_state.data.copy()
    if target_colum is not None:
        x = data.drop(target_colum, axis=1)
        y = data[target_colum]
        scaler = StandardScaler()
        x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
        data = pd.concat([x, y], axis=1)
    else:
        scaler = StandardScaler()
        x = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        data = x
    st.session_state.data = data


def robust_scaler(target_colum=None):
    data = st.session_state.data.copy()
    if target_colum is not None:
        x = data.drop(target_colum, axis=1)
        y = data[target_colum]
        scaler = RobustScaler()
        x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
        data = pd.concat([x, y], axis=1)
    else:
        scaler = RobustScaler()
        x = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        data = x
    st.session_state.data = data

def split_and_train_cls(test_size, model_name,target_colum):
    data = st.session_state.data.copy()
    X = data.drop(target_colum , axis = 1)
    y = data[target_colum]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=42)

    if model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Logistic Regression" :
        model = LogisticRegression()  
    else:
        model = SVC()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Save results
    st.session_state.results = {
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "model": model
    }
def split_and_train_reg(test_size, model_name,target_colum):
    data = st.session_state.data.copy()
    X = data.drop(target_colum , axis = 1)
    y = data[target_colum]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=42)

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree Regressor":
        model = DecisionTreeRegressor()
    elif model_name == "Random Forest Regressor":
        model = RandomForestRegressor()    
    else:
        model = SVR()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Save results
    st.session_state.results = {
        "y_test": y_test,
        "y_pred": y_pred,
        "R2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "model": model
    }
def split_and_train_cluster(test_size, model_name,n_clusters=None,eps=0.5,min_samples=5):
    data = st.session_state.data.copy()
    X = data.select_dtypes(include=[np.number])

    if model_name == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif model_name == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif model_name == "Aglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters) 


    labels = model.fit_predict(X)
    if len(set(labels)) > 1 :
        sil_score = silhouette_score(X,labels)

    else :
        sil_score = "Not applicable (only 1 cluster or noise detected)"
    # Save results
    st.session_state.results = {
        "labels": labels,
        "model": model,
        "silhouette": sil_score
    }

def download_data():
    csv = st.session_state.data.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Processed Data", data=csv, file_name="processed_data.csv")

# ================== Streamlit UI ==================
st.title("🤖 Machine Learning Project")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
if uploaded_file:
    if "data" not in st.session_state:
        st.session_state.data = pd.read_csv(uploaded_file)

    # ================== Tabs ========================
    tab1, tab2, tab3 ,tab4 = st.tabs(["📊 Overview",  "📈 Visualization", "⚙️ Preprocessing","🧠 Modeling"])        
     # ============ TAB 1: Overview ============
    with tab1:
        st.subheader("Data Preview")
        st.write(st.session_state.data.head())

        st.subheader("Dataset Info")
        buffer = io.StringIO()
        st.session_state.data.info(buf=buffer)
        st.text(buffer.getvalue())

        st.subheader("Statistical Summary for numerical columns")
        st.write(st.session_state.data.describe())

        cat_cols = st.session_state.data.select_dtypes(include="object").columns

        if len(cat_cols) > 0:
            st.subheader("Categorical Summary")
            st.write(st.session_state.data.describe(include="O"))
        else:
            st.info("🚫 No categorical columns available (maybe already encoded).")

        # # ydata profiling
        # if st.button("Generate Data Profile Report"):
        #     profile = ProfileReport(st.session_state.data, explorative=True)
        #     st_profile_report(profile)
    # ============ TAB 3: Visualization ============
    with tab2:
        st.subheader("Data Visualization")
        cols = st.multiselect("Select columns to visualize", st.session_state.data.columns)
        if st.button("Show Histogram") and cols:
            for col in cols:
                fig, ax = plt.subplots()
                sns.histplot(st.session_state.data[col], kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)

        if st.button("Show Boxplot") and cols:
            for col in cols:
                fig, ax = plt.subplots()
                sns.boxplot(x=st.session_state.data[col], ax=ax)
                ax.set_title(f"Boxplot of {col}")
                st.pyplot(fig)

        if st.button("Show Correlation Heatmap"):
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(st.session_state.data[cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        if st.button("Show Countplot for Categorical") and cols:
            for col in cols:
                if st.session_state.data[col].dtype == "object":
                    fig, ax = plt.subplots()
                    sns.countplot(x=st.session_state.data[col], ax=ax)
                    ax.set_title(f"Countplot of {col}")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
        # ============ TAB 3: Preprocessing ============
    with tab3:
        st.subheader("Preprocessing Steps")
            # drop columns
        drop_cols = st.multiselect("Select columns to drop", st.session_state.data.columns)
        if st.button("Drop Selected Columns"):
            drop_columns(drop_cols)
            st.success(f"Columns {drop_cols} dropped successfully!")
             
            # imputation
        method = st.selectbox("Choose Imputation Strategy for Numeric Columns", ["mean", "median"])
        if st.button("Apply Imputer"):
            auto_imputer(method)
            st.success(f"Imputation applied! ({method} for numeric, most_frequent for categorical)")

            # handle outliers
        method = st.selectbox("Choose Outliers handeling technique" , ["IQR","Z_score","log"])    
        if st.button("Apply outliers technique"):
            handle_outliers(method)
            st.success(f"{method} Applied on numerical columns")   
            # encoding
        encode_cols = st.multiselect("Select Columns to Encode",st.session_state.data.columns)
        if st.button("Apply One-Hot Encoding") and encode_cols:
            one_hot_encoder(encode_cols)
            st.success(f"One-Hot Encoding applied on {encode_cols}!")

        if st.button("Apply Label Encoding") and encode_cols:
            label_encoder(encode_cols)
            st.success(f"Label Encoding applied on {encode_cols}!")

        if st.button("Apply Ordinal Encoding") and encode_cols:
            ordinal_encoder(encode_cols)
            st.success(f"Ordinal Encoding applied on {encode_cols}")

            # Scalling
        target_col = st.selectbox("select target column to exclude from scalling", 
                                  options=[None] + list(st.session_state.data.columns))    
        if st.button("Apply MinMaxScaler"):
            minmax_scaler(target_col)
            st.success("MinMax Scaling applied!")

        if st.button("Apply StandardScaler"):
            standard_scaler(target_col)
            st.success("Standard Scaling applied!")
        
        if st.button("Apply RobustScaler"):
            robust_scaler(target_col)
            st.success("Robust Scaler applied") 

        st.write("🔍 Current Data Shape:", st.session_state.data.shape)

        if st.button("Download Dataset After Preprocessing"):
             download_data()

    # ============ TAB 4: Modeling ============
    with tab4:
        st.subheader("Choose Task Type")
        task = st.radio("Select Task Type", ["Classification", "Regression","Clustering"])

        if task in ["Classification" ,"Regression"]:
            target_col = st.selectbox("Select Target Column", st.session_state.data.columns)

        if task == "Classification":
                    model_choice = st.selectbox("Choose Model", ["Random Forest", "Decision Tree", "SVM","Logistic Regression"])
        elif task == "Regression":
            model_choice = st.selectbox("Choose Model", ["Linear Regression", "Random Forest Regressor","SVM","Decision Tree Regressor"])
        else : # for clustering
            model_choice = st.selectbox("Choose Clustering Model", ["KMeans" ,"DBSCAN" , "Aglomerative"])
            if model_choice =="KMeans" or model_choice =="Aglomerative":
                n_clusters = st.number_input("Select number of clusters (K)",min_value=1,max_value= 10,
                                            value=3,step=1)
            elif model_choice == "DBSCAN" :
                eps = st.number_input("Select value of eps",min_value=0.1,max_value= 10.0,value=0.5,step=0.1)
                min_samples = st.number_input("Select number of minpts" , min_value =1 , max_value =20,value=5, 
            step=1)

            

        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

        if st.button("Train & Evaluate"):
            if task == "Classification":
                split_and_train_cls(test_size, model_choice,target_col)
                st.success(f"{model_choice} model trained successfully!")
            elif task == "Regression":
                split_and_train_reg(test_size, model_choice,target_col)
                st.success(f"{model_choice} model trained successfully!")  

            else : # for clustering
                if model_choice == "KMeans":
                    split_and_train_cluster(test_size,model_choice,n_clusters=n_clusters) 
                elif model_choice == "DBSCAN":
                    split_and_train_cluster(test_size,model_choice,eps=eps, min_samples=min_samples)
                else : # for Aglomerative
                    split_and_train_cluster(test_size,model_choice,n_clusters=n_clusters)          
                st.success(f"{model_choice} model trained successfully!") 

        # Evaluation
        if "results" in st.session_state:
            st.subheader("📈 Evaluation Metrics")
            res = st.session_state.results
            if task == "Classification":
                st.write("✅ Accuracy:", res["accuracy"])
                st.write("✅ Precision:", res["precision"])
                st.write("✅ Recall:", res["recall"])
                st.write("✅ F1 Score:", res["f1"])

                # Confusion Matrix
                st.subheader("🔲 Confusion Matrix")
                cm = confusion_matrix(res["y_test"], res["y_pred"])
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                st.pyplot(fig)

            elif task == "Regression":
                st.write("📉 MSE:", res["mse"])
                st.write("📉 MAE:", res["mae"])
                st.write("📈 R² Score:", res["R2"])

            else:  # Clustering results
                st.subheader("📊 Clustering Results")
                # st.write("Cluster Labels:", res["labels"])
                st.write("📌 Silhouette Score:", res["silhouette"])
                data = st.session_state.data.copy()
                data["Cluster"] = res["labels"]
        
