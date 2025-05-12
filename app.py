import pickle
import numpy as np
from flask import Flask, render_template, request
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

app = Flask(__name__)

# Load and train model (do this only once or use a saved model)
data = load_breast_cancer()
X, y = data.data, data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
svc_model = SVC(kernel='linear', probability=True)
svc_model.fit(X_train, y_train)

# Save model and scaler for later use
with open("svc_model.pkl", "wb") as f:
    pickle.dump(svc_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

@app.route('/')
def home():
    return render_template('index.html', feature_names=data.feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(request.form.get(f)) for f in data.feature_names]
    loaded_model = pickle.load(open("svc_model.pkl", "rb"))
    loaded_scaler = pickle.load(open("scaler.pkl", "rb"))
    scaled_input = loaded_scaler.transform([input_features])
    prediction = loaded_model.predict(scaled_input)[0]
    prob = loaded_model.predict_proba(scaled_input)[0][prediction]

    result_text = "Malignant (Cancerous)" if prediction == 0 else "Benign (Non-cancerous)"
    return render_template('result.html', result=result_text, probability=f"{prob:.2%}")

if __name__ == '__main__':
    app.run(debug=True)
