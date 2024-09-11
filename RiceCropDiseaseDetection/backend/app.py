import os
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__,
            template_folder=r'..\\frontend',
            static_folder=r'..\\frontend')

# Define the upload folder path
UPLOAD_FOLDER = os.path.join(app.root_path, 'frontend', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model_path = 'model\model\\rice_disease_model.h5'
model = load_model(model_path)

# Disease classes and precautions
classes = ['Bacterial Blight Disease', 'Brown Spot Disease', 'False Smut Disease', 'Rice Blast Disease']
precautions = {
    'Bacterial Blight Disease': [
        'Remove infected plants and use resistant varieties.',
        'Ensure proper water management and avoid water logging.',
        'Apply bactericides like copper compounds if necessary.',
        'Maintain field hygiene and remove weed hosts.'
    ],
    'Brown Spot Disease': [
        'Use resistant varieties and apply appropriate fungicides.',
        'Ensure proper water management and avoid excessive nitrogen fertilization.',
        'Practice crop rotation and field sanitation.',
        'Apply balanced fertilizers and avoid dense planting.'
    ],
    'False Smut Disease': [
        'Use resistant varieties and apply appropriate fungicides at the flowering stage.',
        'Ensure proper water management.',
        'Avoid excessive nitrogen application.',
        'Clean and disinfect seeds before planting.'
    ],
    'Rice Blast Disease': [
        'Use resistant varieties and apply appropriate fungicides.',
        'Ensure proper water management and avoid excessive nitrogen fertilization.',
        'Implement crop rotation and remove crop residues.',
        'Apply silicon-based fertilizers to strengthen plant resistance.'
    ]
}

# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Ensure rescaling to [0, 1]
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file with its original filename
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Preprocess the uploaded image
    img_array = preprocess_image(file_path)

    # Debugging: Print the shape and content of the preprocessed image
    print(f"Preprocessed image shape: {img_array.shape}")
    print(f"Preprocessed image content (first 10 values): {img_array.flatten()[:10]}")

    # Predict disease
    predictions = model.predict(img_array)[0]
    print(f"Raw predictions: {predictions}")

    # Check for correct prediction ranges and normalization
    if not (0 <= predictions[0] <= 1):
        return jsonify({'error': 'Prediction values out of expected range.'}), 400

    predicted_class_index = np.argmax(predictions)
    predicted_class = classes[predicted_class_index]
    predicted_precaution = precautions[predicted_class]

    # Prepare result data
    result = {
        'prediction': predicted_class,
        'probabilities': predictions,
        'classes': classes,
        'image_path': f'/uploads/{filename}',
        'precaution': predicted_precaution
    }

    return render_template('result.html', result=result)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/precaution/<disease>')
def precaution(disease):
    precaution_list = precautions.get(disease, ['No precaution available for this disease.'])
    return render_template('precaution.html', disease=disease, precaution_list=precaution_list)

if __name__ == '__main__':
    app.run(debug=True)
