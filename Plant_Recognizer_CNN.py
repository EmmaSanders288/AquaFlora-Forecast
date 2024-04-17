import cv2
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from joblib import dump, load
from sklearn.metrics import ConfusionMatrixDisplay
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageCms


class Plant_recgonizer:

    def convolution_images(self, folder_path, category):
        # Setting up kernel to be used for convolution of images
        sobel_kernel = np.array([[-1, 0, 1],
                                 [-1, 0, 2],
                                 [-1, 0, 1]])

        # Creating list for the processed images
        list = []
        for filename in os.listdir(folder_path):
            # Use sobel-kernel on the images
            img_sobel = cv2.filter2D(cv2.imread(os.path.join(folder_path, filename)), -1, sobel_kernel)
            # Add the convoluted images to the list
            list.append(img_sobel)

        # Add the images to their corresponding data files based on their type (dog or muffin)
        for i in range(len(list)):
            cv2.imwrite(f"data/{category}/convolution{i}.jpg", list[i])

        # Clear the list when done so that the process can start again with clean list
        list.clear()

    def correct_image_srgb(self, imagepath):
        # Load the image
        image_path= 'data/for_recognition/test.jpeg'
        image = Image.open(image_path)

        # Ensure the image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Create an sRGB profile
        srgb_profile = ImageCms.createProfile("sRGB")

        # Save the image with the sRGB profile, overwriting the original
        image.save(image_path, 'JPEG', icc_profile=ImageCms.ImageCmsProfile(srgb_profile).tobytes())
        print(srgb_profile)
    def load_images(self, folder_path):
        # Create a list for the images
        images = []
        for filename in os.listdir(folder_path):
            # 'take' image from data file
            image_path = os.path.join(folder_path, filename)
            # self.correct_image_srgb(image_path)
            img = cv2.imread(image_path)

            if img is not None:
                # Add image to the list
                images.append(img)
        # Return the list with images
        return images


    def load_and_preprocess_data(self, categories):
        processed_images = []
        target_array = []
        for index, category in enumerate(categories):
            folder_path = f'C:/Users/EmmaS/Documents/M7-Python/Final Project/data/{category}'
            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)
                img = cv2.imread(image_path)
                if img is not None:
                    # Resize the image to the expected input shape of the CNN
                    resized_img = cv2.resize(img, (64, 64))
                    processed_images.append(resized_img)
                    target_array.append(index)  # Assuming categories are ordered and index is the class label

        # Convert lists to numpy arrays
        X = np.array(processed_images)
        y = np.array(target_array)

        # Normalize the image data to [0, 1]
        X = X / 255.0

        # If you have more than two classes, convert y to one-hot encoding
        y = np.eye(len(categories))[y]

        return X, y

    def create_cnn(self, num_classes,optimizer='adam', activation='relu'):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation=activation, input_shape=(64, 64, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=(3, 3), activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation=activation))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def create_model(self, X_train, y_train, categories, grid_search=True):
        num_classes = len(categories)
        cnn = KerasClassifier(model=Plant_recgonizer.create_cnn, verbose=10, num_classes=num_classes)

        if grid_search:
            # Create hyperparameter grid for hyperparameter tuning
            param_grid = {
                'batch_size': [32],  # Best performing batch size
                'epochs': [50, 60, 70],  # Exploring around the best epoch
                'model__activation': ['relu'],  # Best performing activation function
                'model__optimizer': ['adam'],  # Best performing optimizer
                # Add other parameters you want to explore
            }

            # Create the model with GridSearchCV
            model = GridSearchCV(cnn, param_grid, scoring='accuracy', cv=3, verbose=10)
        else:
            # If not using grid search, just use the KerasClassifier as is
            model = cnn

        # Fit/train the model with training data
        model.fit(X_train, y_train, epochs=60)

        # If using grid search, print the best parameters found
        if grid_search:
            print('Best parameters found:\n', model.best_params_)

        # Return the best model (if grid_search=True) or the default model
        return model

    def prediction(self, best_model, X_test, y_test, categories):
        # Create a prediction based on test data
        y_pred = best_model.predict(X_test)
        # Calculate the accuracy and precision of the model based on the predicted and actual data
        y_test_labels = np.argmax(y_test, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)

        # Calculate the accuracy and precision of the model based on the predicted and actual data
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        precision = precision_score(y_test_labels, y_pred_labels, average='weighted')

        # Generate confusion matrix display
        ConfusionMatrixDisplay.from_predictions(y_test_labels, y_pred_labels, display_labels=categories)
        plt.show()

        # Return the accuracy and precision
        return accuracy, precision

    # Function that saves the best_model, so that it can be used for predictions again without fitting again
    # Uses the model and gives a name to be saved with
    def save_model(self, model: KerasClassifier, model_name: str = 'model') -> str:
        # Create a name for the file that the model will be saved in
        filename = f"{model_name}.joblib"
        # Save the model to the file
        dump(model, filename=filename)
        # Print to check saving went well
        print(f"ML Model {model_name} was saved as '{filename}'.")
        # Return the filename
        return filename

    def load_model(self, model_file: str = 'model.joblib') -> KerasClassifier:
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
        self.correct_image_srgb(f'C:/Users/EmmaS/Documents/M7-Python/Final Project/data/for_recognition/farahwsa.png')
        all_images = self.load_images(f'C:/Users/EmmaS/Documents/M7-Python/Final Project/data/for_recognition')
        # Take an image
        img = all_images[index]
        # Resize image for testing
        img_resize = cv2.resize(img, (64, 64))
        # Reshape image to add batch dimension and channel dimension if needed
        img_reshaped = img_resize.reshape(1, 64, 64, 3)  # Assuming the images are color images with 3 channels
        # Use the trained model to predict
        probability = model.predict_proba(img_reshaped)
        predicted_probabilities = model.predict(img_reshaped)
        predicted_class = np.argmax(predicted_probabilities)  # Get the index of the max probability
        print(probability.shape)

        # Print the predicted image category and percentages
        print("The predicted image is:", Categories[predicted_class])
        print(f"Probability for {Categories[0]}: {probability[0] * 100:.2f}%")
        print(f"Probability for {Categories[1]}: {probability[1] * 100:.2f}%")
        print(f"Probability for {Categories[2]}: {probability[2] * 100:.2f}%")
        print(f"Probability for {Categories[3]}: {probability[3] * 100:.2f}%")
        # Show the tested image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

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
        # model = self.create_model(X_train, y_train,Categories, False )
        # model = self.load_model('model.joblib')
        # self.save_model(model)
        # predictions = self.prediction(model, X_test, y_test, Categories)
        # print(predictions)

        self.test_image('model.joblib',Categories, 0)
        # category = returnCategorty('model.joblib',Categories, 1)
        # print(category)

    def main_for_category_prediction(self, filepath):
        Categories = ['Chinese_money_plant', 'Sansieveria', 'Cactus', 'Succulents']
        category = self.returnCategorty('model.joblib', Categories, filepath)
        return category

if __name__ == '__main__':
    plantje = Plant_recgonizer()
    plantje.main()