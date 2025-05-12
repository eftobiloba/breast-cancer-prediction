from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the breast cancer dataset
data = load_breast_cancer()
cancer_df = pd.DataFrame(data.data, columns=data.feature_names)
cancer_df['target'] = data.target

# Split data into features and label
features = cancer_df[data.feature_names].copy()
labels = cancer_df["target"].copy()

# Instantiate scaler and fit on features
scaler = StandardScaler()
scaler.fit(features)

# Transform features
X_scaled = scaler.transform(features.values)

# Split data into train and test
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, labels, train_size=0.7, random_state=0)

# Check the splits are correct
print(f"Train size: {round(len(X_train_scaled) / len(features) * 100)}% \n\
Test size: {round(len(X_test_scaled) / len(features) * 100)}%")

# Logistic Regression
logistic_regression = LogisticRegression(max_iter=10000)
logistic_regression.fit(X_train_scaled, y_train)
log_reg_preds = logistic_regression.predict(X_test_scaled)
print("Logistic Regression Results:")
print(classification_report(y_test, log_reg_preds))

# Support Vector Machine
svm = SVC()
svm.fit(X_train_scaled, y_train)
svm_preds = svm.predict(X_test_scaled)
print("SVM Results:")
print(classification_report(y_test, svm_preds))

# Decision Tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train_scaled, y_train)
tree_preds = tree.predict(X_test_scaled)
print("Decision Tree Results:")
print(classification_report(y_test, tree_preds))

# Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)

# Evaluation
print("Random Forest Results:")
print(classification_report(y_test, rf_preds))