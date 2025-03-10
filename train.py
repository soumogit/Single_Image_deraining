from model import build_derain_model
from data_loader import load_images

def train_model(rainy_images_path, clean_images_path, epochs=10, batch_size=8):
    rainy_images = load_images(rainy_images_path)
    clean_images = load_images(clean_images_path)
    
    model = build_derain_model()
    model.summary()
    model.fit(rainy_images, clean_images, epochs=epochs, batch_size=batch_size)
    model.save("derain_model.h5")
