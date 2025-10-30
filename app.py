# ============================================================
# 🧠 Advanced Kraljic Matrix Procurement Classification App
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Kraljic Matrix Classifier", page_icon="🧠", layout="wide")
st.title("🧠 Advanced Kraljic Matrix Procurement Classification System")

train_file = st.sidebar.file_uploader("Upload Training Data (CSV)", type=["csv"])
if train_file:
    df = pd.read_csv(train_file)
    st.success("✅ Training dataset uploaded successfully!")
    st.dataframe(df.head())
    if 'Kraljic_Category' not in df.columns:
        st.error("❌ Column 'Kraljic_Category' not found.")
        st.stop()
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'Kraljic_Category' in string_cols:
        string_cols.remove('Kraljic_Category')
    df = df.drop(columns=string_cols)
    le_target = LabelEncoder()
    df['Kraljic_Category'] = le_target.fit_transform(df['Kraljic_Category'])
    X = df.drop('Kraljic_Category', axis=1)
    y = df['Kraljic_Category']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='rbf', probability=True),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)
    results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
    st.subheader("🏆 Model Accuracy Comparison")
    st.bar_chart(results_df.set_index('Model'))
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]
    st.success(f"Best Model: {best_model_name} ({results[best_model_name]:.2f})")
    test_file = st.file_uploader("Upload Test CSV", type=["csv"], key="test")
    if test_file:
        test_df = pd.read_csv(test_file)
        test_df = test_df.drop(columns=string_cols, errors='ignore')
        test_scaled = scaler.transform(test_df)
        predictions = best_model.predict(test_scaled)
        test_df['Predicted_Category'] = le_target.inverse_transform(predictions)
        st.dataframe(test_df[['Predicted_Category']])
        csv = test_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", data=csv, file_name="Kraljic_Predictions.csv", mime="text/csv")
else:
    st.info("👈 Upload your training CSV to start.")
