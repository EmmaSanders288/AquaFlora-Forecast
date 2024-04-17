import os
import cv2
import numpy as np
import tf_keras.models
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from keras.models import Model


class PlantClassifier:
    def __init__(self, categories , img_width=150, img_height=150):
        self.categories = categories
        self.img_width = img_width
        self.img_height = img_height

    def load_and_preprocess_images(self, data_path):
        processed_images = []
        target_array = []
        for index, category in enumerate(self.categories):
            folder_path = os.path.join(data_path, category)
            for filename in os.listdir(folder_path):
                img = cv2.imread(os.path.join(folder_path, filename))
                if img is not None:
                    resized_img = cv2.resize(img, (self.img_width, self.img_height))
                    processed_images.append(resized_img)
                    target_array.append(index)

        X = np.array(processed_images) / 255.0
        y = np.eye(len(self.categories))[np.array(target_array)]
        return X, y
    def _build_model(self):
        model = Sequential()
        model.add(
            Conv2D(filters=16, kernel_size=(3, 3), activation="relu", input_shape=(self.img_width, self.img_height, 3)))
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



    def grid_search(self, X_train, y_train):
        model = KerasClassifier(model=self.create_keras_model(), verbose=0)
        # Define the grid search parameters
        param_grid = {
            'epochs': [20, 30, 40],
            'batch_size': [32, 64, 128],
            'validation_split': [0.1, 0.2, 0.3]
        }


        # Create Grid Search
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=10)

        # Fit the model
        grid_result = grid.fit(X_train, y_train)

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        return grid
    def predict(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        # Calculate the accuracy and precision of the model based on the predicted and actual data
        y_test_labels = np.argmax(y_test, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)

        # Calculate the accuracy and precision of the model based on the predicted and actual data
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        precision = precision_score(y_test_labels, y_pred_labels, average='weighted')

        # Generate confusion matrix display
        ConfusionMatrixDisplay.from_predictions(y_test_labels, y_pred_labels, display_labels=self.categories)
        plt.show()

        # Return the accuracy and precision
        return accuracy, precision
    def test(self, model, test_img_path):
        test_img = image.load_img(test_img_path, target_size=(self.img_width, self.img_height, 3))
        test_img_array = image.img_to_array(test_img)
        test_img_array /= 255.0
        test_img_array = np.expand_dims(test_img_array, axis=0)

        predictions = model.predict(test_img_array)
        top_classes = np.argsort(predictions[0])[::-1][:4]
        print(f"The predicted image is {self.categories[top_classes[0]]}")
        for i in range(len(self.categories)):
            print(f"{self.categories[i]}     {predictions[0][i]* 100:.2f}%")
        img = plt.imread(test_img_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()


    def save_model(self, filepath, model):
        model.save(filepath)  # Corrected method name

    def load_model(self, filepath, new_input_shape):
        loaded_model = tf_keras.models.load_model(filepath)
        my_input_tensor = Input(shape=new_input_shape)
        loaded_model.layers.pop(0)
        new_outputs = loaded_model(my_input_tensor)
        new_model = tf_keras.Model(inputs=my_input_tensor, outputs=new_outputs)
        print('Loaded model successfully')
        return new_model
    def main(self):
        X, y = self.load_and_preprocess_images('C:/Users/EmmaS/Documents/M7-Python/Final Project/data')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        model = self.grid_search(X_train, y_train)  # Make sure train_model returns the model
        print("Model trained successfully:", model)
        # self.save_model('model.h5', model)
        # self.load_model('model.h5', (150, 150, 3))
        self.test(model, 'data/for_recognition/test.jpeg')
        self.predict(model, X_test, y_test,)


if __name__ == '__main__':
    plant = PlantClassifier(['Chinese_money_plant', 'Sansieveria', 'Cactus', 'Succulents'])
    plant.main()
