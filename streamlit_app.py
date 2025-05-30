import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import SVC
import plotly.express as px

# --- Load Data ---
DATA_URL = "https://raw.githubusercontent.com/adinplb/LR-SVM-NN_Breast-Tumor_Prediction/refs/heads/main/dataset/Breast%20Cancer%20Wisconsin.csv"
FEATURE_NAMES = ["id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
                 "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
                 "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se",
                 "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
                 "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
                 "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                 "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, names=FEATURE_NAMES, header=0)
    df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')  # Handle missing 'Unnamed: 32'
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df


df = load_data()


# --- Preprocessing Functions ---
def preprocess_data(df):
    df = df.copy()  # Operate on a copy to avoid modifying the original DataFrame

    # Handle potential outliers (using a simplified method for demonstration)
    for col in df.select_dtypes(include=np.number).columns:
        if col != 'diagnosis':  # Don't remove outliers from target variable
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]

    # Feature Correlation and PCA
    correlated_features = ['radius_mean', 'perimeter_mean', 'area_mean', 'radius_worst',
                           'perimeter_worst', 'area_worst']
    if all(col in df.columns for col in correlated_features):
        pca = PCA(n_components=1, random_state=123)
        df['dimension'] = pca.fit_transform(df[correlated_features]).flatten()
        df = df.drop(columns=correlated_features)

    return df


def split_and_scale(df):
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    if 'diagnosis' in numerical_features:
        numerical_features.remove('diagnosis')  # Ensure 'diagnosis' is not scaled

    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled


# --- Model Training and Evaluation Functions ---
def train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test, use_smote=False):
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)
    model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced') # Add class_weight
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, confusion_matrix(y_test, y_pred), classification_report(y_test, y_pred, output_dict=True)


def train_and_evaluate_nn(X_train, X_test, y_train, y_test, use_smote=False):
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)  # Reduced verbosity
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int)
    return model, confusion_matrix(y_test, y_pred), classification_report(y_test, y_pred, output_dict=True)


def train_and_evaluate_svm(X_train, X_test, y_train, y_test, use_smote=False):
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)
    model = SVC(kernel='rbf', random_state=42, class_weight='balanced') # Add class_weight
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, confusion_matrix(y_test, y_pred), classification_report(y_test, y_pred, output_dict=True)


# --- Visualization Functions ---
def plot_confusion_matrix(cm, class_names, title):
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="True", color="Count"),
                    x=class_names,
                    y=class_names,
                    color_continuous_scale=px.colors.sequential.Blues,
                    title=title)
    st.plotly_chart(fig)


def plot_classification_report(report, title):
    report_df = pd.DataFrame(report).transpose()
    fig = px.bar(report_df,
                 barmode='group',
                 title=title,
                 labels={'value': 'Score', 'index': 'Metric', 'variable': 'Class'},
                 color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(fig)


def plot_feature_distribution(df, feature, title):
    fig = px.histogram(df, x=feature, color='diagnosis', marginal='box',
                      title=title, color_discrete_sequence=['green', 'red'])
    st.plotly_chart(fig)


def plot_correlation_matrix(df, title):
    corr_matrix = df.corr()
    fig = px.imshow(corr_matrix,
                    color_continuous_scale=px.colors.diverging.RdBu,
                    title=title)
    st.plotly_chart(fig)


# --- Streamlit App ---
def main():
    st.title("Breast Tumor Prediction and Analysis")
    st.markdown("A web application for breast tumor analysis and prediction using machine learning.")

    # --- Data Loading and Preprocessing ---
    with st.expander("Data Overview"):
        st.header("Data Overview")
        st.dataframe(df.head())
        st.write(df.describe())

    preprocessed_df = preprocess_data(df)

    with st.expander("Data Preprocessing"):
        st.header("Data Preprocessing")
        st.subheader("Outlier Handling and Feature Engineering")
        st.write("Outliers were removed using the IQR method. Highly correlated features were reduced to a single dimension using PCA.")
        st.dataframe(preprocessed_df.head())

    X_train, X_test, y_train, y_test = split_and_scale(preprocessed_df)

    # --- Exploratory Data Analysis ---
    with st.expander("Exploratory Data Analysis"):
        st.header("Exploratory Data Analysis")
        st.subheader("Feature Distributions")
        for col in preprocessed_df.columns:
            if col not in ['diagnosis']:
                plot_feature_distribution(preprocessed_df, col, f"Distribution of {col} by Diagnosis")

        st.subheader("Correlation Matrix")
        plot_correlation_matrix(preprocessed_df.drop(columns=['diagnosis'], errors='ignore'), "Correlation Matrix")

    # --- Model Training and Evaluation ---
    st.header("Model Training and Evaluation")
    model_choice = st.selectbox("Select Model", ["Logistic Regression", "Neural Network", "SVM"])
    use_smote = st.checkbox("Apply SMOTE for class balancing", value=True)

    if model_choice == "Logistic Regression":
        st.subheader("Logistic Regression Model")
        model, cm, report = train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test, use_smote)
        plot_confusion_matrix(cm, ['Benign', 'Malignant'], "Logistic Regression Confusion Matrix")
        plot_classification_report(report, "Logistic Regression Classification Report")

    elif model_choice == "Neural Network":
        st.subheader("Neural Network Model")
        model, cm, report = train_and_evaluate_nn(X_train, X_test, y_train, y_test, use_smote)
        plot_confusion_matrix(cm, ['Benign', 'Malignant'], "Neural Network Confusion Matrix")
        plot_classification_report(report, "Neural Network Classification Report")

    elif model_choice == "SVM":
        st.subheader("SVM Model")
        model, cm, report = train_and_evaluate_svm(X_train, X_test, y_train, y_test, use_smote)
        plot_confusion_matrix(cm, ['Benign', 'Malignant'], "SVM Confusion Matrix")
        plot_classification_report(report, "SVM Classification Report")


if __name__ == "__main__":
    main()
