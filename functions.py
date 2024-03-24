from tensorflow.keras import layers, models, Input
from keras.layers import Input, Dense, Dropout, Flatten, Lambda
from keras.models import Model
import os
import numpy as np
import random
from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import tensorflow as tf

def create_pairs(directory):
    #directory is the path of the dataset (main folder)
    #folders are classes in dataset
    pairs = []
    labels = []
    
    folders = os.listdir(directory)
    for folder_name in folders:
        folder_path = os.path.join(directory, folder_name)
        
        if os.path.isdir(folder_path):
            images = os.listdir(folder_path)
            
            folder_length = len(images)
            for i in range(folder_length):
                for j in range(folder_length):
                    if i != j:
                        image_path = os.path.join(folder_path, images[i])
                        pairs.append([image_path, os.path.join(folder_path, images[j])])
                        labels.append(1)#positive pairs
                        
                        dif_folder = random.choice([x for x in folders if x != folder_name])
                        dif_folder_path = os.path.join(directory, dif_folder)
                        dif_image_path = os.path.join(dif_folder_path, random.choice(os.listdir(dif_folder_path)))

                        pairs.append([image_path, dif_image_path])
                        labels.append(0)#negative pairs
                        
    return np.array(pairs), np.array(labels) 

def process_images(pairs, target_size):
    images = []
     
    for pair in pairs:
        img1 = load_img(pair[0], target_size=target_size, color_mode='grayscale')
        img2 = load_img(pair[1], target_size=target_size, color_mode='grayscale')

        images.append((img1, img2))
        
    return images

def prepare_data(directory, target_size):
    X, y = create_pairs(directory)
    X = process_images(X, target_size)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    X_train, X_test = X_train / 255.0, X_test / 255.0

    X_train = np.expand_dims(X_train, axis = -1)
    X_test = np.expand_dims(X_test, axis = -1)
    y_train = np.expand_dims(y_train, axis = -1)
    y_test = np.expand_dims(y_test, axis = -1)
    
    return X_train, X_test, y_train, y_test


def euclidean_distance(vecs):
    (imgA, imgB) = vecs
    ss = K.sum(K.square(imgA - imgB), axis = 1, keepdims=True)
    return K.sqrt(K.maximum(ss, K.epsilon()))

def contrastiveLoss(y, y_preds, margin=1):
    y = tf.cast(y, y_preds.dtype)
    y_preds_squared = K.square(y_preds)
    margin_squared = K.square(K.maximum(margin - y_preds, 0))
    loss = K.mean(y * y_preds_squared + (1 - y) * margin_squared)
    return loss