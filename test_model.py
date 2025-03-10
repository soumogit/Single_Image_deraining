import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def test_model(input_image_path):
    # Load the trained model
    model = load_model("derain_model.h5")
    
    # Load and preprocess the input image
    img = cv2.imread(input_image_path)
    img = cv2.resize(img, (256, 256))  # Resize to model's input size
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict using the model
    output_image = model.predict(img)
    output_image = np.squeeze(output_image, axis=0)  # Remove batch dimension

    # Display the original and derained images
    plt.subplot(1, 2, 1)
    plt.title("Input Rainy Image")
    plt.imshow(cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title("Derained Image")
    plt.imshow(output_image)
    plt.show()

# Test the model with a specific input image
test_model("C:/Users/gsoum/Downloads/image1.jpg ")

