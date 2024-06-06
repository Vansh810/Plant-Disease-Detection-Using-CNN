import numpy as np
import cv2
import pickle
import os
from glob import glob
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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

        np.save('images.npy', images)
        np.save('labels.npy', labels)

        print("Preprocessing Done")

    return train_test_split(images, labels, test_size=0.2)


def train_model(X_train, X_test, y_train, y_test):
    try:
        model = load_model('plant_disease_detection_model.h5')
        with open('history.pkl') as f:
            history = pickle.load(f)
        print("Loaded Saved Model")

    except (OSError, FileNotFoundError):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(len(disease_labels), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        datagen.fit(X_train)

        history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

        with open('training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)

        history = history.history
        model.save('plant_disease_detection_model.h5')

        print("Model Trained")

    return model, history


# Function to evaluate the model
def evaluate(X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')


# Function to plot training history
def plot_history(history):
    plt.plot(history['accuracy'], label='Accuracy (training)')
    plt.plot(history['val_accuracy'], label='Accuracy (validation)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_confusion_matrix(model, X_test, y_test, labels):
    predictions = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, predictions, labels=np.arange(len(labels)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    plt.show()


X_train, X_test, y_train, y_test = preprocess()
model, history = train_model(X_train, X_test, y_train, y_test)
evaluate(X_test, y_test)
plot_history(history)
plot_confusion_matrix(model, X_test, y_test, disease_labels)
