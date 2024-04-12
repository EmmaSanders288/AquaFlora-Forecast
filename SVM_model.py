import math

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, max_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from joblib import dump, load
import serial
# Cactei = 0, Pancake plant = 1, Sanseveria = 2, Succulent = 3

class PlantWaterNeedPredictor:
    def __init__(self, sensor_data_path, com_port, baud_rate):
        self.sensor_data = pd.read_csv(sensor_data_path, sep=';')
        self.labelEncoder = LabelEncoder()
        self.com = com_port
        self.baud = baud_rate
        self.x = serial.Serial(self.com, self.baud, timeout=0.1)

    def data_sensors(self):
        while self.x.isOpen():
            data = str(self.x.readline().decode('utf-8')).rstrip()
            if data:
                values = data.split(',')
                values.pop()
                if len(values) == 4:
                    return values

    def preprocess_data(self):
        self.sensor_data['Waterneed'] = pd.to_numeric(self.sensor_data['Waterneed'])
        self.sensor_data['Plant Type'] = self.labelEncoder.fit_transform(self.sensor_data['Plant Type'])

    def train_model(self):
        X = self.sensor_data.drop(columns=['Unnamed: 0', 'Waterneed', 'Time'])
        y = self.sensor_data['Waterneed']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ''''
        parameters = {'kernel': ('rbf', 'sigmoid'), 'C': [0.1, 1, 10], 'gamma': ('auto', 'scale')}
        svr = SVR()
        clf = GridSearchCV(svr, parameters, scoring='neg_mean_squared_error', verbose=14)
        clf.fit(X_train, y_train)
        print(clf.best_estimator_)
        print(clf.best_score_)
        print(clf.best_score_)
        '''

        clf = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10, gamma='auto'))
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mxe = max_error(y_test, y_pred)
        print(f"R2 Score: {r2:.2f}, MAE: {mae:.2f}, MSE: {mse:.2f}, Max Error: {mxe:.2f}")
        return clf

    def save_model(self,model: Pipeline, model_name: str = 'regression') -> str:
        # Create a name for the file that the model will be saved in
        filename = f"{model_name}.joblib"
        # Save the model to the file
        dump(model, filename=filename)
        # Print to check saving went well
        print(f"ML Model {model_name} was saved as '{filename}'.")
        # Return the filename
        return filename

    def prediction(self, clf):
        while True:
            sensor_values = self.data_sensors()
            LDR_data, temp_data, humi_data, soil_data = map(float, sensor_values)
            continue_data = [[LDR_data, temp_data, humi_data, soil_data, 2]]
            prediction = clf.predict(continue_data)
            day_prediction = math.floor(prediction*7)
            print(f"Predicted water need: {day_prediction:.2f} days")
            return day_prediction
    def load_model(self,model_file: str = 'regression.joblib') -> SVR:
        # Load the model
        # print('loading model')
        model = load(model_file)
        # Returns the loaded model
        return model

    def predict_for_main(self):
        self.preprocess_data()
        model = self.load_model()
        prediction = self.prediction(model)
        data = self.data_sensors()
        return prediction, data
if __name__ == '__main__':
    sensor_data_path = "C:/Users/EmmaS/Documents/M7-Python/Final Project/csv_files/PlantDataLabels.csv"
    com_port = "COM7"
    baud_rate = 9600

    predictor = PlantWaterNeedPredictor(sensor_data_path, com_port, baud_rate)
    predictor.preprocess_data()
    # model = predictor.train_model()
    # predictor.save_model(model)

