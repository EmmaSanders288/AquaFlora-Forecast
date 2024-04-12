import pandas as pd

class PlantDataProcessor:
    def __init__(self, sensor_data_path):
        self.sensor_data = pd.read_csv(sensor_data_path, sep=';')

    def clean_data(self):
        # Remove rows with missing values
        self.sensor_data.dropna(inplace=True)

        # Filter out non-positive values
        self.sensor_data = self.sensor_data[
            (self.sensor_data["Light"] > 0) &
            (self.sensor_data["Temperature"] > 0) &
            (self.sensor_data["Humidity"] > 0) &
            (self.sensor_data["Soil_humidity"] > 0)
        ]

    def deduplicate_data(self):
        self.sensor_data.drop_duplicates(subset=['Light', 'Temperature', 'Humidity', 'Soil_humidity', 'Plant Type'], keep='last', inplace=True)

    def save_cleaned_data(self, output_path):
        self.sensor_data.to_csv(output_path, sep=';')

if __name__ == "__main__":
    sensor_data_path = "C:/Users/anna/PycharmProjects/AquaFlora-Forecast/SensorData.csv"
    output_path = "C:/Users/anna/PycharmProjects/AquaFlora-Forecast/PlantDataLabels.csv"

    processor = PlantDataProcessor(sensor_data_path)
    processor.clean_data()
    processor.deduplicate_data()
    processor.save_cleaned_data(output_path)
