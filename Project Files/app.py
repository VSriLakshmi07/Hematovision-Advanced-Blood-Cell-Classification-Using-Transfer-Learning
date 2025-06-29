import os
import numpy as np
import cv2
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import io
import base64

# Load the saved model
from tensorflow.keras.models import load_model
model = load_model('Blood Cell.h5')

# Define Flask app
app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64))
    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    class_names = ['Neutrophil', 'Monocyte', 'Eosinophil', 'Lymphocyte']
    predicted_class = class_names[class_index]

    _, buffer = cv2.imencode('.png', img)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    return render_template('result.html', prediction=predicted_class, image=encoded_img)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)