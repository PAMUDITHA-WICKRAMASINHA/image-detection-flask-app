from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# Initialize Flask application
app = Flask(__name__)

# Load your custom-trained model
model = tf.keras.models.load_model('my_model.h5')  # Replace 'my_model.h5' with the path to your model file

# Define image dimensions
img_height, img_width = 220, 220

# Define a route for the API
@app.route('/predict', methods=['POST'])
def predict():
    classes = [ 'Brown_blotch','Croweb', 'Dry_Bubble','Green_Molds','Pseudomonas_Tolaasii','Soft_Rot','Healthy']

    if 'image' not in request.files:
        response = {'error': True, 'message': 'No file part', 'data': {'class': None, 'disease_name': None}}
        return jsonify(response)
    else:
        file = request.files['image']
        if file.filename == '':
            response = {'error': True, 'message': 'Empty file', 'data': {'class': None, 'disease_name': None}}
            return jsonify(response)
        img_bytes = file.read()
        img = image.load_img(io.BytesIO(img_bytes), target_size=(img_height, img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=-1)
        response = {'error': False, 'message': 'Detect sucess', 'data': {'class': int(predicted_class), 'disease_name': classes[int(predicted_class)]}}
        return jsonify(response)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
