from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = load_wine()
wine_df = pd.DataFrame(data.data, columns=data.feature_names)
wine_df['target'] = data.target

# Split data into features and label 
features = wine_df[data.feature_names].copy()
labels = wine_df["target"].copy()

# Instantiate scaler and fit on features
scaler = StandardScaler()
scaler.fit(features)

# Transform features
X_scaled = scaler.transform(features.values)

# Split data into train and test
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, labels, train_size=.7, random_state=0)

# Check the splits are correct
print(f"Train size: {round(len(X_train_scaled) / len(features) * 100)}% \n\
Test size: {round(len(X_test_scaled) / len(features) * 100)}%")

# Initializing the model 
logistic_regression = LogisticRegression()
# Training the models
logistic_regression.fit(X_train_scaled, y_train)
# Making predictions with the model
log_reg_preds = logistic_regression.predict(X_test_scaled)
# Evaluate the model
from sklearn.metrics import classification_report
print(classification_report(y_test, log_reg_preds))

# Initializing the model 
svm = SVC()
# Training the models
svm.fit(X_train_scaled, y_train)
# Making predictions with the model
svm_preds = svm.predict(X_test_scaled)
# Evaluate the model
from sklearn.metrics import classification_report
print(classification_report(y_test, svm_preds))

tree = DecisionTreeClassifier(random_state=42) #Why do use think we are using random state and why arent we using it for the others, Some Machine learning 
                                              #algorithms are prone to randomization and would not produce the same result if random state is not decleared
# Training the models
tree.fit(X_train_scaled, y_train)
# Making predictions with the model
tree_preds = tree.predict(X_test_scaled)
# Evaluate the model
from sklearn.metrics import classification_report
print(classification_report(y_test, tree_preds))