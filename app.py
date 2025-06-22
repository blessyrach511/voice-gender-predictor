from flask import Flask, render_template, request
import librosa
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

def extract_features(filename):
    y, sr = librosa.load(filename, duration=2.5, offset=0.6)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    return mfccs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join('static', filename)
    file.save(filepath)

    features = extract_features(filepath).reshape(1, -1)
    prediction = model.predict(features)[0]

    return render_template('index.html', prediction=prediction, audio=filename)

if __name__ == '__main__':
    app.run(debug=True)
