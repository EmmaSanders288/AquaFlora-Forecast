import os
import cv2
import numpy as np
import tf_keras.models
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.models import *
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from keras.models import load_model
from PIL import Image, ImageCms


class PlantClassifier:
    def __init__(self, categories, imagepath, img_width=150, img_height=150):
        self.categories = categories
        self.img_width = img_width
        self.img_height = img_height
        self.image_path = imagepath

    def convolution_images(self, folder_path, category):
        # Convolute images using Sobel kernel
        sobel_kernel = np.array([[-1, 0, 1], [-1, 0, 2], [-1, 0, 1]])
        for filename in os.listdir(folder_path):
            img_sobel = cv2.filter2D(cv2.imread(os.path.join(folder_path, filename)), -1, sobel_kernel)
            cv2.imwrite(f"data/{category}/convolution{i}.jpg", img_sobel)

    def correct_image_srgb(self, image_path):
        # Correct image color space to sRGB
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        srgb_profile = ImageCms.createProfile("sRGB")
        image.save(image_path, 'JPEG', icc_profile=ImageCms.ImageCmsProfile(srgb_profile).tobytes())

    def load_and_preprocess_images(self, data_path):
        processed_images = []
        target_array = []
        for index, category in enumerate(self.categories):
            folder_path = os.path.join(data_path, category)
            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)
                img = cv2.imread(image_path)
                if img is not None:
                    resized_img = cv2.resize(img, (self.img_width, self.img_height))
                    processed_images.append(resized_img)
                    target_array.append(index)

        X = np.array(processed_images) / 255.0
        y = np.eye(len(self.categories))[np.array(target_array)]
        return X, y

    def _build_model(self):
        # Define CNN model architecture
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", input_shape=(self.img_width, self.img_height, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.categories), activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def create_keras_model(self):
        return self._build_model()


    def grid_search(self, X_train, y_train, gridsearch):
        cnn_model = KerasClassifier(model=self.create_keras_model(), verbose=0)
        # Define the grid search parameters
        if gridsearch:
            param_grid = {
                'epochs': [20, 30, 40],
                'batch_size': [32, 64, 128],
                'validation_split': [0.1, 0.2, 0.3]
            }

            # Create Grid Search
            model = GridSearchCV(estimator=cnn_model, param_grid=param_grid, cv=3, verbose=10)
            grid_result = model.fit(X_train, y_train)
            history = grid_result.best_estimator_.model.history_
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = model.cv_results_['mean_test_score']
            stds = model.cv_results_['std_test_score']
            params = model.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))

        else:
            model = KerasClassifier(model=self.create_keras_model(), batch_size=32, epochs=30, validation_split=0.1)
            history = model.fit(X_train, y_train)
        return model, history

    def predict(self, model, X_test, y_test):
        # Make predictions and evaluate model performance
        y_pred = model.predict(X_test)
        y_test_labels = np.argmax(y_test, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        precision = precision_score(y_test_labels, y_pred_labels, average='weighted')
        f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
        ConfusionMatrixDisplay.from_predictions(y_test_labels, y_pred_labels, display_labels=self.categories)
        plt.show()
        return accuracy, precision, f1

    def test(self, model, test_img_path):
        # Test the model on a single image
        test_img = image.load_img(test_img_path, target_size=(self.img_width, self.img_height, 3))
        test_img_array = image.img_to_array(test_img)
        test_img_array /= 255.0
        test_img_array = np.expand_dims(test_img_array, axis=0)
        predictions = model.predict(test_img_array)
        top_classes = np.argsort(predictions[0])[::-1][:4]
        print(f"The predicted image is {self.categories[top_classes[0]]}")
        for i in range(len(self.categories)):
            print(f"{self.categories[i]}     {predictions[0][i] * 100:.2f}%")
        img = plt.imread(test_img_path)
        plt.imshow(img)
        plt.axis('off')
        category = self.categories[top_classes[0]]
        return category

    def load_model(self, filepath, new_input_shape):
        # Load a pre-trained model
        loaded_model = tf_keras.models.load_model(filepath)
        my_input_tensor = Input(shape=new_input_shape)
        loaded_model.layers.pop(0)
        new_outputs = loaded_model(my_input_tensor)
        new_model = tf_keras.Model(inputs=my_input_tensor, outputs=new_outputs)
        print('Loaded model successfully')
        return new_model

    def plot_history(self, history):
        acc = history.history_["accuracy"]
        loss = history.history_["loss"]
        val_loss = history.history_["val_loss"]
        val_accuracy = history.history_["val_accuracy"]

        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, "b", label="traning_acc")
        plt.plot(x, val_accuracy, "r", label="traning_acc")
        plt.title("Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(x, loss, "b", label="traning_acc")
        plt.plot(x, val_loss, "r", label="traning_acc")
        plt.title("Loss")
        plt.show()

    def main(self):
        # Train and evaluate the model
        # X, y = self.load_and_preprocess_images('C:/Users/EmmaS/Documents/M7-Python/Final Project/data')
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        # model, history = self.grid_search(X_train, y_train, False)
        # model.model.save("model.keras", model)
        loaded_model = load_model('model.keras', compile=True)
        # self.predict(loaded_model, X_test, y_test)
        # self.plot_history(history)
        self.test(loaded_model, self.image_path)

    def main_for_category_prediction(self, filepath):
        # Predict category of a single image
        loaded_model = load_model('model.keras', compile=True)
        category = self.test(loaded_model, filepath)
        return category


# Create an instance of PlantClassifier and execute the main function


    # plant = PlantClassifier(['Chinese_money_plant', 'Sansieveria', 'Cactus', 'Succulents'], 'data/for_recognition/pancake.jpeg')
    # plant.main()
