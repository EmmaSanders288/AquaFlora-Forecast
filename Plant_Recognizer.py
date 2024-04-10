import cv2
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
def convolution_images(folder_path, category):
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
def load_images(folder_path):
    # Create a list for the images
    images = []
    for filename in os.listdir(folder_path):
        # 'take' image from data file
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            # Add image to the list
            images.append(img)
    # Return the list with images
    return images

def resize_image(images):
    # Create a list for the processed (resized) images
    processed_images = []
    # All the images in the images list
    for img in images:
        # Resize the image and add to the processed_images list
        resized_img = cv2.resize(img, (800, 800))
        processed_images.append(resized_img.flatten())
    # Return the list with resized images
    return processed_images
def normalize_images(folder_path, category):
    min_value = 0
    max_value = 1
    norm_type = cv2.NORM_MINMAX
    list = []
    for filename in os.listdir(folder_path):
        image_rgb = cv2.cvtColor(cv2.imread(os.path.join(folder_path, filename)), cv2.COLOR_BGR2RGB)
        b, g, r = cv2.split(image_rgb)
        b_normalized = cv2.normalize(b.astype('float'), None, min_value, max_value, norm_type)
        g_normalized = cv2.normalize(g.astype('float'), None, min_value, max_value, norm_type)
        r_normalized = cv2.normalize(r.astype('float'), None, min_value, max_value, norm_type)
        normalized_image = cv2.merge((b_normalized, g_normalized, r_normalized))
        list.append(normalized_image)
    for i in range(len(list)):
        cv2.imwrite(f"data/{category}/normalized{i}.jpg", list[i])
def load_and_preprocess_data(Categories):
    # List for processed images
    processed_images = []
    # Array for test images
    target_array = []
    for category in Categories:
        folder_path = f'C:/Users/EmmaS/PycharmProjects/AquaFlora-Forecast/data_backuo/{category}'
        # Add convolution to images for extra data
        # convolution_images(folder_path, category)
        # normalize_images(folder_path, category)
        images = load_images(folder_path)
        processed_images.extend(images)  # Combine images from both categories

        # Assign target labels based on category
        target_label = Categories.index(category)
        target_array.extend([target_label] * len(images))

    # Resize and flatten images
    resized_images = resize_image(processed_images)
    flat_data = np.array(resized_images)
    target = np.array(target_array)
    # Create a DataFrame for further analysis
    df = pd.DataFrame(flat_data)
    df['Target'] = target

    # Split data for training and testing
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


def create_model_pipeline(X_train, y_train, grid_search=True):
    # Define the steps for the pipeline
    steps = [
        ('scaler', StandardScaler()),  # Standardize features
        ('classifier', OneVsRestClassifier(SVC(probability=True)))  # Support Vector Classifier
    ]

    if grid_search:
        # Create hyperparameter grid for hyperparameter tuning
        param_grid = {
            "classifier__estimator__C": [0.1, 1, 10, 100],
            'classifier__estimator__gamma': [0.0001, 0.001, 0.1, 1, 10],
            'classifier__estimator__kernel': [ 'kernel','linear', 'rbf']
        }

        # Create the pipeline with GridSearchCV
        model = GridSearchCV(Pipeline(steps), param_grid, scoring="accuracy", cv=3, verbose=14)
    else:
        # Create the pipeline without hyperparameter tuning
        model = Pipeline(steps)

    # Fit/train the best model with training data
    print('Going to fit:', model)
    model.fit(X_train, y_train)

    # Return the best model (if grid_search=True) or the default model
    return model
def prediction(best_model, X_test, y_test):
    # Create a prediction based on test data
    y_pred = best_model.predict(X_test)
    # Calculate the accuracy and precision of the model based on the predicted and actual data
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_test, y_pred, average='weighted')
    # Return the accuracy and precision
    return accuracy, precision

# Function that saves the best_model, so that it can be used for predictions again without fitting again
# Uses the model and gives a name to be saved with
def save_model(model: Pipeline, model_name: str = 'model') -> str:
    # Create a name for the file that the model will be saved in
    filename = f"{model_name}.joblib"
    # Save the model to the file
    dump(model, filename=filename)
    # Print to check saving went well
    print(f"ML Model {model_name} was saved as '{filename}'.")
    # Return the filename
    return filename

def load_model(model_file: str = 'model.joblib') -> SVC:
    # Load the model
    model = load(model_file)
    # Returns the loaded model
    return model

def main():
    # Create the categories for images
    Categories = ['Chinese_money_plant','Sansieveria', 'Cactus','Succulents']
    X, y = load_and_preprocess_data(Categories)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = create_model_pipeline(X_train,y_train,True)
    # model = load_model('model.joblib')
    predictions = prediction(model, X_test, y_test)
    print(predictions)
    save_model(model)
    # # test_image('model.joblib',Categories, 6)

if __name__ == '__main__':
    main()