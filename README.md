# Single Image De-Raining Using Machine Learning  

This project focuses on developing a machine learning model to remove rain streaks from images, enhancing clarity for applications like autonomous vehicles and surveillance. The model is built using TensorFlow and Keras, utilizing datasets like Rain100L for training.  

## 📂 Project Structure  

├── data_loader.py # Loads and preprocesses images from the dataset

├── derain_model.h5 # Pre-trained model for de-raining images

├── main.py # Main entry point for training and testing

├── model.py # Defines the CNN-based de-raining model

├── test.py # Tests the trained model on new images

├── test_model.py # Additional module for model evaluation

├── train.py # Trains the model using the dataset

perl
Copy
Edit

## 📌 Modules Overview  

### 1️⃣ Main Module (`main.py`)  
- Defines dataset paths and manages training and testing.  
- Calls `train_model()` for training and `test_model()` for evaluation.  

### 2️⃣ Data Loader (`data_loader.py`)  
- Loads images using OpenCV (`cv2.imread`) and resizes them to 256x256 pixels.  
- Normalizes pixel values to [0,1] for better model convergence.  

### 3️⃣ Model Architecture (`model.py`)  
- Implements a CNN-based model with convolutional layers and residual learning.  
- Uses Batch Normalization and ReLU activation for improved performance.  
- Compiled with Adam optimizer and MSE loss function.  

### 4️⃣ Training Module (`train.py`)  
- Loads training data and initializes the model from `model.py`.  
- Trains the model on the Rain100L dataset.  
- Saves the trained model as `derain_model.h5`.  

### 5️⃣ Testing Module (`test.py`)  
- Loads the trained model and a test image.  
- Predicts the derained image and displays results using `matplotlib`.  

## 🖼 Sample Results  
The trained model successfully removes rain streaks, improving image clarity.  

## 🚀 How to Run  

1. **Install dependencies:**  
   ```bash
   pip install tensorflow numpy opencv-python matplotlib
Train the model:
bash
Copy
Edit
python train.py
Test the model:
bash
Copy
Edit
python test.py


## 📜 License

This project is open-source and free to use for research and development.


This README is structured for GitHub, with clear module breakdowns, installation instructions, and execution 
