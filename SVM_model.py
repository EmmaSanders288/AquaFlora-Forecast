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
        # Initialize the PlantWaterNeedPredictor with sensor data path, serial connection details, and encoder
        self.sensor_data = pd.read_csv(sensor_data_path, sep=';')
        self.labelEncoder = LabelEncoder()
        self.com = com_port
        self.baud = baud_rate
        self.x = serial.Serial(self.com, self.baud, timeout=0.1)

    def data_sensors(self):
        # Read sensor data from serial connection
        while self.x.isOpen():
            data = str(self.x.readline().decode('utf-8')).rstrip()
            if data:
                values = data.split(',')
                values.pop()
                if len(values) == 4:
                    return values

    def preprocess_data(self):
        # Preprocess sensor data
        self.sensor_data['Waterneed'] = pd.to_numeric(self.sensor_data['Waterneed'])
        self.sensor_data['Plant Type'] = self.labelEncoder.fit_transform(self.sensor_data['Plant Type'])

    def train_model(self):
        # Train SVR model
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

    def save_model(self, model: Pipeline, model_name: str = 'regression') -> str:
        # Save the trained model to a file using joblib
        filename = f"{model_name}.joblib"
        dump(model, filename=filename)
        print(f"ML Model {model_name} was saved as '{filename}'.")
        return filename

    def get_category(self, category):
        # Map plant category to numerical value
        if category == 'Cactus':
            self.category_number = 0
        elif category == 'Chinese_money_plant':
            self.category_number = 1
        elif category == 'Sansieveria':
            self.category_number = 2
        elif category == 'Succulents':
            self.category_number = 3
        return self.category_number

    def prediction(self, clf):
        # Make water need prediction based on sensor data and plant category
        while True:
            sensor_values = self.data_sensors()
            LDR_data, temp_data, humi_data, soil_data = map(float, sensor_values)
            continue_data = [[LDR_data, temp_data, humi_data, soil_data, self.category_number]]
            prediction = clf.predict(continue_data)
            day_prediction = math.floor(prediction * 7)
            print(f"Predicted water need: {day_prediction:.2f} days")
            return day_prediction

    def load_model(self, model_file: str = 'regression.joblib') -> SVR:
        # Load a saved model from file using joblib
        model = load(model_file)
        return model

    def predict_for_main(self):
        # Main function to preprocess data, load trained model, and make prediction
        self.preprocess_data()
        model = self.load_model()
        prediction = self.prediction(model)
        data = self.data_sensors()
        return prediction, data


if __name__ == '__main__':
    # Example usage
    sensor_data_path = "C:/Users/anna/PycharmProjects/AquaFlora-Forecast/csv_files/PlantDataLabels.csv"
    com_port = "COM16"
    baud_rate = 9600

    predictor = PlantWaterNeedPredictor(sensor_data_path, com_port, baud_rate)
    predictor.preprocess_data()
    model = predictor.train_model()
    # predictor.save_model(model)
