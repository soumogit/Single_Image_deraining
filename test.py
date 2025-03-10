import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_loader import load_images

def test_model(rainy_images_path):
    model = load_model("derain_model.h5")
    rainy_images = load_images(rainy_images_path)
    
    sample_image = np.expand_dims(rainy_images[0], axis=0)
    output_image = model.predict(sample_image)

    plt.subplot(1, 2, 1)
    plt.title("Rainy Image")
    plt.imshow(rainy_images[0])

    plt.subplot(1, 2, 2)
    plt.title("Derained Image")
    plt.imshow(output_image[0])
    plt.show()