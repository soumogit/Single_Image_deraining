from train import train_model
from test import test_model

# Update the paths to your dataset location
rainy_images_path = "C:/Users/gsoum/Downloads/rain_data_train_Light-2024/rain_data_train_Light/train"
clean_images_path = "C:/Users/gsoum/Downloads/rain_data_train_Light-2024/rain_data_train_Light/norain"


# Train the model
train_model(rainy_images_path, clean_images_path, epochs=10, batch_size=8)

# Test the model
test_model(rainy_images_path)
