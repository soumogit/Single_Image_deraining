import os
import cv2
import numpy as np
import matplotlib.image as img
def load_images(folder, img_size=(256, 256)):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, img_size)
            img = img / 255.0
            images.append(img)
    return np.array(images)
