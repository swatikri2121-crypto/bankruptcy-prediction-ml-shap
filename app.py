import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


models = {
    "Logistic Regression": joblib.load("log_reg.pkl"),
    "Random Forest": joblib.load("rf.pkl"),
    "XGBoost": joblib.load("xgb.pkl"),
    "CatBoost": joblib.load("cat.pkl"),
    "Neural Network": joblib.load("nn.pkl")
}

scaler = joblib.load("scaler.pkl")

-
st.set_page_config(page_title="Bankruptcy Risk Dashboard", layout="wide")
st.title("🏦 Corporate Bankruptcy & Financial Distress Prediction")

st.sidebar.header("Settings")
input_method = st.sidebar.radio("Choose Input Method:", ["Manual Entry", "Upload CSV"])
selected_model = st.sidebar.selectbox("Choose Model:", list(models.keys()))


if input_method == "Manual Entry":
    st.subheader("Enter Company Financial Ratios")

    
    feature_names = [
        "ROA(C) before interest and depreciation before interest",
        "ROA(A) before interest and % after tax",
        "ROA(B) before interest and depreciation after tax",
        "Operating Gross Margin",
        "Realized Sales Gross Margin",
        "Operating Profit Rate",
        "Pre-tax net Interest Rate"
    ]

    manual_data = {}
    for feat in feature_names:
        manual_data[feat] = st.number_input(feat, value=0.0)

    input_df = pd.DataFrame([manual_data])

   
    X_scaled = scaler.transform(input_df)
    model = models[selected_model]
    y_pred = model.predict_proba(X_scaled)[:, 1][0]

    
    st.subheader("Prediction Result")
    st.write(f"**Bankruptcy Probability:** {y_pred:.2f}")

    if y_pred < 0.33:
        st.success("Risk Category: LOW")
    elif y_pred < 0.66:
        st.warning("Risk Category: MEDIUM")
    else:
        st.error("Risk Category: HIGH")

   
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)
    st.subheader("Feature Importance (SHAP)")
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())


elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV with financial ratios", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

       
        if "Bankrupt?" in input_df.columns:
            input_df = input_df.drop(columns=["Bankrupt?"])

       
        X_scaled = scaler.transform(input_df)
        model = models[selected_model]
        y_pred = model.predict_proba(X_scaled)[:, 1]

        
        result_df = pd.DataFrame({
            "Bankruptcy Probability": y_pred,
            "Risk Category": pd.cut(y_pred, bins=[-1,0.33,0.66,1],
                                    labels=["Low","Medium","High"])
        })
        st.subheader("Batch Predictions")
        st.write(result_df)

       
        st.subheader("Risk Distribution")
        fig, ax = plt.subplots()
        result_df["Risk Category"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

        
        explainer = shap.Explainer(model, X_scaled)
        shap_values = explainer(X_scaled[:1])
        st.subheader("Feature Importance (SHAP) - Example Company")
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(plt.gcf())
