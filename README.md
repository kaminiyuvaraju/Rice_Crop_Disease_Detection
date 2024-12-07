# Rice Crop Disease Detection

This project aims to develop a machine learning model that can detect diseases in rice crops based on images of rice leaves. The backend of the application uses **Flask** for the server-side logic, while the machine learning model is built using **TensorFlow/Keras** for deep learning. The frontend is built using **HTML, CSS, and JavaScript**, allowing users to upload images for disease prediction.

## Table of Contents
- [Project Description](#project-description)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)
- [How to Use](#how-to-use)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Project Description

Rice crop diseases can significantly impact yield and quality. This project provides an automated solution for detecting and classifying common rice diseases such as:
- False Smut Disease
- Brown Spot Disease
- Blast Disease
- Bacterial Blight Disease

Users can upload images of rice leaves, and the system will predict the disease using a **ResNet-50** Convolutional Neural Network (CNN) model trained on a labeled dataset.

## Technologies Used

- **Backend**: Flask, TensorFlow/Keras
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: Convolutional Neural Networks (ResNet-50)
- **Dataset**: Custom dataset with labeled rice leaf images (stored in `data/train` directory)
- **Other Libraries**: NumPy, Pandas, OpenCV

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/kaminiyuvaraju/RiceCropDiseaseDetection.git
   cd RiceCropDiseaseDetection

### Create a Virtual Environment:

#### Windows:
python -m venv env
env\Scripts\activate
#### Ubuntu : 
python3 -m venv env
source env/bin/activate

## Install Dependencies:
pip install -r backend/requirements.txt

Run the Application:
In the backend directory, run:

## Run the Application:
python app.py
The app will be accessible at http://127.0.0.1:5000/.

## Project Structure
``` 
RiceCropDiseaseDetection/
├── backend/
│   ├── app.py              # Main Flask application
│   ├── model/
│   │   ├── model.py        # Pre-trained model
│   │   ├── predict.py      # Prediction script
│   │   ├── train.py        # Model training script
│   └── requirements.txt    # Python dependencies
├── data/
│   ├── train/              # Training dataset (labeled images of diseases)
├── frontend/
│   ├── index.html          # Frontend homepage (file upload interface)
│   ├── result.html         # Page to display disease prediction results
│   ├── style.css           # CSS styles
│   ├── script.js           # JavaScript for interactivity
└── README.md               # Project documentation



