import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC

app = Flask(__name__)
os.makedirs("model", exist_ok=True)

# Load and train model (do this only once or use a saved model)
data = load_breast_cancer()
X, y = data.data, data.target

# Select top k features
k = 10  # You can change this
selector = SelectKBest(score_func=f_classif, k=k)
X_selected = selector.fit_transform(X, y)
selected_feature_names = data.feature_names[selector.get_support()]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
svc_model = SVC(kernel='linear', probability=True)
svc_model.fit(X_train, y_train)

# Save model, scaler, selector, and selected features
with open("model/svc_model.pkl", "wb") as f:
    pickle.dump(svc_model, f)
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("model/selector.pkl", "wb") as f:
    pickle.dump(selector, f)
with open("model/selected_features.pkl", "wb") as f:
    pickle.dump(selected_feature_names, f)

@app.route('/')
def home():
    selected_feature_names = pickle.load(open("model/selected_features.pkl", "rb"))
    return render_template('index.html', feature_names=selected_feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    selected_feature_names = pickle.load(open("model/selected_features.pkl", "rb"))
    input_features = [float(request.form.get(f)) for f in selected_feature_names]

    loaded_model = pickle.load(open("model/svc_model.pkl", "rb"))
    loaded_scaler = pickle.load(open("model/scaler.pkl", "rb"))

    scaled_input = loaded_scaler.transform([input_features])
    prediction = loaded_model.predict(scaled_input)[0]
    prob = loaded_model.predict_proba(scaled_input)[0][prediction]

    result_text = "Malignant (Cancerous)" if prediction == 0 else "Benign (Non-cancerous)"
    return render_template('result.html', result=result_text, probability=f"{prob:.2%}")

if __name__ == '__main__':
    app.run(debug=True)
