from Plant_Recognizer import Plant_recgonizer
from screen import Screen
from SVM_model import PlantWaterNeedPredictor
import math


# from saveDataToExcel import SensorDataLogger

def main():
    user = 'Emma'
    if user == 'Emma':
        url = 'C:/Users/EmmaS/Documents/M7-Python/Final Project'
    elif user == 'Anna':
        url = 'C:/Users/anna/PycharmProjects/AquaFlora-Forecast'

    filepath = url + '/data/for_recognition/dfxhgcjvk.jpeg'
    sensor_data_path = url + '/csv_files/PlantDataLabels.csv'

    com_port = "COM16"
    baud_rate = 9600

    plant_recognizer = Plant_recgonizer()
    plant_need_predictor = PlantWaterNeedPredictor(sensor_data_path, com_port, baud_rate)
    screen = Screen(filepath)

    category = plant_recognizer.main_for_category_prediction(filepath)
    print(category)
    plant_need_predictor.get_category(category)

    while True:
        data = plant_need_predictor.predict_for_main()
        days = data[0]
        light = math.floor((float(data[1][0])/1023)*100)
        temperature = data[1][1]
        humidity = data[1][2]
        soil_humidity = data[1][3]
        #print(f"Day: {days}, Light intensity: {light} lux, Temperature: {temperature}°C, Humidity: {humidity}%, Soil Humidity: {soil_humidity}%")
        screen.main_loop(category, light, temperature, humidity, soil_humidity, days)


if __name__ == '__main__':
    main()
