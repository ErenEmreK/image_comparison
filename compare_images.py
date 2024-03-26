import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras import config
import numpy as np
from siamese_older import image_size, save_file

# Load the model
config.enable_unsafe_deserialization()

# Load the model
siamese_model = tf.keras.models.load_model(save_file, compile=False)

# Function to load and preprocess a pair of images
def load_and_preprocess_pair(image1_path, image2_path, target_size=image_size):
    #TODO up to date this regarding last preprocess function
    img1 = img_to_array(load_img(image1_path, target_size=target_size)) / 255.0
    img2 = img_to_array(load_img(image2_path, target_size=target_size)) / 255.0

    return [img1, img2]

# Example usage
image1_path = 'test/1/n02087394_261.jpg'  # Replace with the actual path to your image
image2_path = 'test/1/n02087394_261.jpg'  # Replace with the actual path to your image

image_pair = load_and_preprocess_pair(image1_path, image2_path)

# Reshape the image pair to match the model's input shape
image_pair = np.array([image_pair[0], image_pair[1]])

# Make a prediction using the loaded model
similarity_percentage = siamese_model.predict([image_pair[:, 0], image_pair[:, 1]])

print(f"Similarity Percentage: {similarity_percentage[0][0] * 100:.2f}%")
