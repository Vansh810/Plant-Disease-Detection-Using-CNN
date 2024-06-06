import numpy as np
import cv2
import pickle
import os
from glob import glob
from sklearn.model_selection import train_test_split

data_path = 'PlantVillage'
disease_labels = ['Bacterial_Spot', 'Early_Blight', 'Healthy', 'Late_Blight', 'Leaf_Mold', 'Septoria_Leaf_Spot',
                  'Spider_Mites', 'Target_Spot', 'Tomato_Mosaic_Virus', 'Tomato_YellowLeaf_Curl_Virus']


def preprocess():
    try:
        images = np.load('images.npy')
        labels = np.load('labels.npy')
        print("Loaded Preprocessed Files")

    except FileNotFoundError:
        images = []
        labels = []
        for disease in disease_labels:
            img_files = glob(os.path.join(data_path, disease, '*.jpg'))
            for file in img_files:
                img = cv2.imread(file)
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(disease_labels.index(disease))

        images = np.array(images)
        labels = np.array(labels)

        images = images / 255.0
        images = np.expand_dims(images, axis=-1)

        np.save('images.npy', images)
        np.save('labels.npy', labels)

        print("Preprocessing Done")

    return train_test_split(images, labels, test_size=0.2)


X_train, X_test, y_train, y_test = preprocess()