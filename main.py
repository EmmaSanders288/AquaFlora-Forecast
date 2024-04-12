from Plant_Recognizer import Plant_recgonizer
from screen import Screen
from SVM_model import PlantWaterNeedPredictor
# from saveDataToExcel import SensorDataLogger

def main():
    user = ''
    if user == 'Emma':
        filepath= 'C:/Users/EmmaS/Documents/M7-Python/Final Project/data/for_recognition/9cdb7a9b-afc5-4541-a736-1997efa5842a.jpg'
        sensor_data_path = "C:/Users/EmmaS/Documents/M7-Python/Final Project/csv_files/PlantDataLabels.csv"
    if user == 'Anna':
        filepath=''
        sensor_data_path=''

    com_port = "COM17"
    baud_rate = 9600

    plant_recognizer = Plant_recgonizer()
    plant_need_predictor = PlantWaterNeedPredictor(sensor_data_path, com_port, baud_rate)

    category = plant_recognizer.main_for_category_prediction(filepath)
    print(category)

    while True:
        data = plant_need_predictor.predict_for_main()
        days = data[0]
        light = data[1][0]
        temperature = data[1][1]
        humidity = data[1][2]
        soil_humidity = data[1][3]
        print(f"Day: {days}, Light intensity: {light} lux, Temperature: {temperature}°C, Humidity: {humidity}%, Soil Humidity: {soil_humidity}%")



if __name__ == '__main__':
    main()