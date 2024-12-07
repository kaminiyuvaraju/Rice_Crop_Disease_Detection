import os
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, redirect, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import secrets
from datetime import timedelta

app = Flask(__name__, template_folder=r'..\\frontend', static_folder=r'..\\frontend')

# Generate a random secret key for session
app.secret_key = secrets.token_hex(16)  # Generates a random secure key

# Set session lifetime (optional)
app.permanent_session_lifetime = timedelta(days=7)

# Database connection function
def get_db_connection():
    conn = psycopg2.connect(
        dbname='ricecropdisease',
        user='postgres',
        password='yuva1432',
        host='localhost',
        port='5432'
    )
    return conn

# Define the upload folder path
UPLOAD_FOLDER = os.path.join(app.root_path, '../uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model_path = 'model\\rice_disease_model.h5'
model = load_model(model_path)

# Disease classes and precautions
classes = ['Bacterial Blight Disease', 'Brown Spot Disease', 'False Smut Disease', 'Rice Blast Disease','Sheath Blight Disease', 
    'Tungro Virus Disease']
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
    ],
    'Sheath Blight Disease': [
        'Use resistant varieties and avoid excessive nitrogen application.',
        'Apply appropriate fungicides like carbendazim or validamycin.',
        'Maintain field hygiene by removing infected plant debris.',
        'Ensure proper water management and avoid water logging.'
    ],
    'Tungro Virus Disease': [
        'Plant resistant varieties and ensure timely planting.',
        'Control leafhopper vectors using appropriate insecticides.',
        'Remove infected plants to reduce virus spread.',
        'Maintain field hygiene and manage weed hosts.'
    ]
}

# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Ensure rescaling to [0, 1]
    return img_array

# Default route to login
@app.route('/')
def index():
    if 'username' not in session:  # If not logged in, redirect to login page
        return redirect(url_for('login'))
    return render_template('index.html')  # Or any other page that you want to show after login

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        conn = get_db_connection()
        cursor = conn.cursor()

        # Query to check if the user exists
        cursor.execute('SELECT password FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if user and check_password_hash(user[0], password):  
            session['username'] = username  # Set session after successful login
            flash(f'Welcome, {username}!', 'success')
            return redirect(url_for('index'))  # Redirect to the index route after login
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')

# Route for signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return render_template('signup.html')

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (username, email, password) VALUES (%s, %s, %s)',
                       (username, email, hashed_password))
        conn.commit()
        cursor.close()
        conn.close()

        flash('Signup successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

# Route for predicting rice disease
# Route for predicting rice disease
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

    # Predict disease
    predictions = model.predict(img_array)[0]

    # Print the shape of predictions and values for debugging
    print(f"Predictions: {predictions}")  # Debugging: See the actual predictions
    print(f"Predictions Shape: {predictions.shape}")  # Check the shape of the prediction output

    # Slice the predictions to only use the first 4 values if the model outputs more than 4
    predictions = predictions[:4]

    # Check if predictions are within expected range
    if not (0 <= predictions[0] <= 1):
        return jsonify({'error': 'Prediction values out of expected range.'}), 400

    # Find the predicted class index, ensure it is within the correct range
    predicted_class_index = np.argmax(predictions)

    # Check if the predicted class index is within the bounds of the 'classes' list
    if predicted_class_index >= len(classes):
        return jsonify({'error': 'Prediction index out of bounds.'}), 400

    predicted_class = classes[predicted_class_index]
    predicted_precaution = precautions[predicted_class]

    # Save the prediction to the database
    conn = get_db_connection()
    cursor = conn.cursor()

    # Insert the prediction into the predictions table
    cursor.execute("""
        INSERT INTO predictions (image_path, predicted_disease)
        VALUES (%s, %s)
    """, (f'/uploads/{filename}', predicted_class))
    
    # Update the frequency of the predicted disease
    cursor.execute("""
        INSERT INTO disease_frequency (disease_name, count)
        VALUES (%s, 1)
        ON CONFLICT (disease_name)
        DO UPDATE SET count = disease_frequency.count + 1
    """, (predicted_class,))

    conn.commit()
    cursor.close()
    conn.close()

    # Prepare result data
    result = {
        'prediction': predicted_class,
        'probabilities': predictions.tolist(),  # Convert to list for rendering in templates
        'classes': classes,
        'image_path': f'/uploads/{filename}',
        'precaution': predicted_precaution
    }

    return render_template('result.html', result=result)

# Route for uploading image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route for disease precautions
@app.route('/precaution/<disease>')
def precaution(disease):
    precaution_list = precautions.get(disease, ['No precaution available for this disease.'])
    return render_template('precaution.html', disease=disease, precaution_list=precaution_list)
# Route for displaying disease frequency
@app.route('/frequency')
def frequency():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch the disease frequencies from the database
    cursor.execute('SELECT disease_name, count FROM disease_frequency ORDER BY count DESC')
    disease_data = cursor.fetchall()

    cursor.close()
    conn.close()

    # Prepare the data to send to the template
    frequency_data = [{'disease': row[0], 'count': row[1]} for row in disease_data]

    return render_template('frequency.html', frequency_data=frequency_data)

# Route for logout
@app.route('/logout')
def logout():   
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Route for disease frequency
@app.route('/disease-frequency')
def disease_frequency():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch the frequency of each disease
    cursor.execute('SELECT disease_name, frequency FROM disease_frequency ORDER BY frequency DESC')
    frequency_data = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('disease_frequency.html', frequency_data=frequency_data)

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)
