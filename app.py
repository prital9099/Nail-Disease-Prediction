from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load trained model
model = load_model("nail_disease_model.keras")

# Class names (IMPORTANT: Must match training folder names)
class_names = [
    'Darier_s disease',
    'Muehrck-e_s lines',
    'aloperia areata',
    'beau_s lines',
    'bluish nail',
    'clubbing',
    'eczema',
    'half and half nailes (Lindsay_s nails)',
    'koilonychia',
    'leukonychia',
    'onycholycis',
    'pale nail',
    'red lunula',
    'splinter hemmorrage',
    'terry_s nail',
    'white nail',
    'yellow nails'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html')

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    # Proper formatting
    formatted_disease = predicted_class.replace('_', ' ').title()

    prediction_sentence = f'The Person is diagnosed with <strong>{formatted_disease}</strong> Nail Disease'

    return render_template('index.html', prediction_text=prediction_sentence)

if __name__ == "__main__":
    app.run(debug=True, port=8080)

