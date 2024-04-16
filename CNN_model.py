import cv2
import os
from matplotlib import pyplot as plt
import joblib
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from tensorflow.keras import layers, models

class PlantClassifier:
    def _init_(self, categories):
        self.categories = categories

    def convolution_images(self, folder_path, category):
        sobel_kernel = np.array([[-1, 0, 1],
                                 [-1, 0, 2],
                                 [-1, 0, 1]])
        processed_images = []
        for filename in os.listdir(folder_path):
            img_sobel = cv2.filter2D(cv2.imread(os.path.join(folder_path, filename)), -1, sobel_kernel)
            processed_images.append(img_sobel)
        return processed_images

    def load_images(self, folder_path):
        images = []
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                images.append(img)
        return images

    def resize_image(self, images):
        processed_images = []
        for img in images:
            resized_img = cv2.resize(img, (32, 32))
            processed_images.append(resized_img.flatten())
        return processed_images

    def normalize_images(self, folder_path, category):
        min_value = 0
        max_value = 1
        norm_type = cv2.NORM_MINMAX
        normalized_images = []
        for filename in os.listdir(folder_path):
            image_rgb = cv2.cvtColor(cv2.imread(os.path.join(folder_path, filename)), cv2.COLOR_BGR2RGB)
            b, g, r = cv2.split(image_rgb)
            b_normalized = cv2.normalize(b.astype('float'), None, min_value, max_value, norm_type)
            g_normalized = cv2.normalize(g.astype('float'), None, min_value, max_value, norm_type)
            r_normalized = cv2.normalize(r.astype('float'), None, min_value, max_value, norm_type)
            normalized_image = cv2.merge((b_normalized, g_normalized, r_normalized))
            normalized_images.append(normalized_image)
        return normalized_images

    def load_and_preprocess_data(self):
        processed_images = []
        target_array = []
        for category in self.categories:
            folder_path = f'data/{category}'
            images = self.load_images(folder_path)
            processed_images.extend(images)
            target_label = self.categories.index(category)
            target_array.extend([target_label] * len(images))

        resized_images = self.resize_image(processed_images)
        flat_data = np.array(resized_images)
        target = np.array(target_array)

        return flat_data, target

    def create_cnn_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.categories), activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        X, y = self.load_and_preprocess_data()
        model.fit(X, y, epochs=10, validation_split=0.2)
        return model

    def save_model(self, filename):
        if self.model:
            joblib.dump(self.model, filename)
            print(f"Model saved as {filename}")
        else:
            print("Model not trained yet. Train the model first.")
        return filename

    def load_model(self, filename):
        self.model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return self.model

    def prediction(self, best_model, X_test, y_test, labels):
        # Create a prediction based on test data
        y_pred = best_model.predict(X_test)
        # Calculate the accuracy and precision of the model based on the predicted and actual data
        accuracy = accuracy_score(y_pred, y_test)
        precision = precision_score(y_test, y_pred, average='weighted')
        y_pred = best_model.predict(X_test)
        valid_labels = [label for label in labels if label in np.unique(y_test)]
        print(valid_labels)
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=labels)
        plt.show()
        # Return the accuracy and precision
        return accuracy, precision

    def returnCategorty(self, model_file, Categories, filepath):
        model = self.load_model(model_file)
        # Load all the images into a list
        img = cv2.imread(filepath)
        # Take an image
        # Resize, flatten and reshape image for testing
        img_resize = cv2.resize(img, (32, 32))
        img_flatten = img_resize.flatten()
        img_reshaped = img_flatten.reshape(1, -1)
        # Use the trained model to predict
        probability = model.predict_proba(img_reshaped)
        predicted_class = model.predict([img_flatten])[0]  # Wrap img_flatten in a list
        return Categories[predicted_class]
        # Print the predicted image category and percentages

    def main(self):
        # Create the categories for images
        Categories = ['Chinese_money_plant', 'Sansieveria', 'Cactus', 'Succulents']
        # X, y = self.load_and_preprocess_data(Categories)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # model = self.create_model_pipeline(X_train, y_train, False)
        # model = load_model('model.joblib')
        self.save_model(self.model)
        # predictions = prediction(model, X_test, y_test, Categories)
        # print(predictions)

        # test_image('model.joblib',Categories, 0)
        # category = returnCategorty('model.joblib',Categories, 1)
        # print(category)

    def main_for_category_prediction(self, filepath):
        Categories = ['Chinese_money_plant', 'Sansieveria', 'Cactus', 'Succulents']
        category = self.returnCategorty('model.joblib', Categories, filepath)
        return category


'''
categories = ["rose", "tulip", "daisy", "sunflower"]
plant_classifier = PlantClassifier(categories)
X, y = plant_classifier.load_and_preprocess_data()
model = plant_classifier.create_cnn_model()
model.summary()
     

# Function that saves the best_model, so that it can be used for predictions again without fitting again
    # Uses the model and gives a name to be saved with
    def save_model(self, model: Pipeline, model_name: str = 'model') -> str:
        # Create a name for the file that the model will be saved in
        filename = f"{model_name}.joblib"
        # Save the model to the file
        dump(model, filename=filename)
        # Print to check saving went well
        print(f"ML Model {model_name} was saved as '{filename}'.")
        # Return the filename
        return filename

    def load_model(self, model_file: str = 'model.joblib') -> SVC:
        # Load the model
        # print('loading model')
        model = load(model_file)
        # Returns the loaded model
        return model

    def test_image(self, model_file, Categories, index):
        # Load the model
        model = self.load_model(model_file)
        print("model_loaded")
        # Load all the images into a list
        all_images = self.load_images(f'C:/Users/EmmaS/Documents/M7-Python/Final Project/data/test')
        # Take an image
        img = all_images[index]
        # Resize, flatten and reshape image for testing
        img_resize = cv2.resize(img, (32, 32))
        img_flatten = img_resize.flatten()
        img_reshaped = img_flatten.reshape(1, -1)
        # Use the trained model to predict
        probability = model.predict_proba(img_reshaped)
        predicted_class = model.predict([img_flatten])[0]  # Wrap img_flatten in a list
        # Print the predicted image category and percentages
        print("The predicted image is:", Categories[predicted_class])
        print(f"Probability for {Categories[0]}: {probability[0][0] * 100:.2f}%")
        print(f"Probability for {Categories[1]}: {probability[0][1] * 100:.2f}%")
        print(f"Probability for {Categories[2]}: {probability[0][2] * 100:.2f}%")
        print(f"Probability for {Categories[3]}: {probability[0][3] * 100:.2f}%")
        # Show the tested image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
 def create_model_pipeline(self, X_train, y_train, grid_search=True):
        # Define the steps for the pipeline


        # Fit/train the best model with training data
        print('Going to fit:', model)

        model.fit(X_train, y_train)

        # Return the best model (if grid_search=True) or the default model
        return model
'''
