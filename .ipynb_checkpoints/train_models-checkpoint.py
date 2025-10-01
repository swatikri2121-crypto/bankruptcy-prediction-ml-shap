import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("data.csv")

# Separate features and target
X = df.drop(columns=["Bankrupt?"])
y = df["Bankrupt?"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ----------------------------
# Scaling
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# ----------------------------
# Train Models
# ----------------------------

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
log_reg.fit(X_train_scaled, y_train)
joblib.dump(log_reg, "log_reg.pkl")

# Random Forest
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)  # RF doesn’t need scaling
joblib.dump(rf, "rf.pkl")

# XGBoost
scale_pos_weight = (y_train.value_counts()[0] / y_train.value_counts()[1])  # Handle imbalance
xgb = XGBClassifier(
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False  # safe for older versions
)
xgb.fit(X_train, y_train)
joblib.dump(xgb, "xgb.pkl")

# CatBoost (silent mode)
cat = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0, random_state=42)
cat.fit(X_train, y_train)
joblib.dump(cat, "cat.pkl")

# Neural Network (MLP)
nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
nn.fit(X_train_scaled, y_train)
joblib.dump(nn, "nn.pkl")

print("✅ All models and scaler saved successfully!")
