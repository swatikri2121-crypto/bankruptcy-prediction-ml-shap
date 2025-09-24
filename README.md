Project Overview :

Corporate bankruptcy and financial distress prediction is crucial for investors, auditors, and financial institutions.
This project uses financial ratios as predictors to classify companies as Healthy or Bankrupt using advanced machine learning models.
What makes this project unique is the integration of Explainable AI (SHAP) and a Streamlit Dashboard for easy visualization and decision-making.


Features: 

Predicts bankruptcy risk using multiple ML models:
Logistic Regression
Random Forest
XGBoost
CatBoost
Neural Network (MLP)
Handles manual entry or CSV batch upload

Outputs:

Bankruptcy Probability
Risk Category (Low / Medium / High)
SHAP Explainability Graphs (feature importance)
Streamlit Dashboard for interactive use
Risk Distribution Graphs for batch analysis


Dataset :

Source:https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction
Input: Financial ratios (liquidity, leverage, profitability, cash flow metrics, etc.)
Target: Bankrupt? (0 = Healthy, 1 = Bankrupt)


Tech Stack :

Python 
Scikit-Learn
XGBoost, CatBoost
Streamlit (for Dashboard)
SHAP (Explainable AI)
Matplotlib, Pandas, NumPy

Project Structure

Bankruptcy-Prediction-Project
  
  app.py                
  train_models.py       
  data.csv              
  scaler.pkl           
  log_reg.pkl           
  rf.pkl                
  xgb.pkl               
  cat.pkl               
  README.md    

How to Run ? 

Make sure bankruptcy.ipynb, train_models.ipynb, app.py, and the dataset are in the same folder.
First, run bankruptcy.ipynb to preprocess the data.
Then, run train_models.ipynb to train the models.
Finally, launch the app using the terminal or command prompt: streamlit run app.py


Developed by [swati kumari] — passionate data vizualization and data science.
