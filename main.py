from CNN_images import PlantClassifier  # Importing PlantClassifier from CNN_images module
from screen import Screen  # Importing Screen class from screen module
from SVM_model import PlantWaterNeedPredictor  # Importing PlantWaterNeedPredictor from SVM_model module
import math  # Importing math module for mathematical operations


def main():
    user = 'Emma'
    # Setting file paths based on user
    if user == 'Emma':
        url = 'C:/Users/EmmaS/Documents/M7-Python/Final Project'
    elif user == 'Anna':
        url = 'C:/Users/anna/PycharmProjects/AquaFlora-Forecast'

    # File path for test image
    filepath = url + '/data/for_recognition/succulent.jpeg'
    # File path for sensor data
    sensor_data_path = url + '/csv_files/PlantDataLabels.csv'

    # Communication settings for the PlantWaterNeedPredictor
    com_port = "COM7"
    baud_rate = 9600

    # Initialize instances of PlantClassifier, PlantWaterNeedPredictor, and Screen
    plant_recognizer = PlantClassifier(['Chinese_money_plant', 'Sansieveria', 'Cactus', 'Succulents'], filepath)
    plant_need_predictor = PlantWaterNeedPredictor(sensor_data_path, com_port, baud_rate)
    screen = Screen(filepath)

    # Predict plant category
    category = plant_recognizer.main_for_category_prediction(filepath)
    print(category)
    # Get plant category for PlantWaterNeedPredictor
    plant_need_predictor.get_category(category)

    # Main loop for continuous monitoring and displaying plant data
    while True:
        # Predict plant's water needs
        data = plant_need_predictor.predict_for_main()
        days = data[0]  # Days till watering
        # Calculate light intensity percentage
        light = math.floor((float(data[1][0])/1023)*100)
        temperature = data[1][1]  # Temperature
        humidity = data[1][2]  # Humidity
        soil_humidity = data[1][3]  # Soil Humidity

        # Display plant data on the screen
        screen.main_loop(category, light, temperature, humidity, soil_humidity, days)


if __name__ == '__main__':
    main()  # Execute the main function
