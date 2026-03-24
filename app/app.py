from flask import Flask, request, render_template
import joblib
import sys
import os

sys.path.append(os.path.abspath('../src'))

from preprocessing import clean_text

app = Flask(__name__)

model = joblib.load("../src/model.pkl")
vectorizer = joblib.load("../src/vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned]).toarray()
    
    prediction = model.predict(vector)[0]
    
    result = "Phishing" if prediction == 1 else "Legitimate"
    
    return render_template('index.html', prediction_text=f"Result: {result}")

if __name__ == "__main__":
    app.run(debug=True)
