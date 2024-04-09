import serial
import time
import csv

with open('SensorData.csv', mode='a', newline= '') as sensor_file:
    sensor_writer = csv.writer(sensor_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    sensor_writer.writerow(["Light", "Temperature", "Humidity", "Soil_humidity",'Plant Type' ,"Time"])

plant_type = 'Sanseveria'

com = "COM5"
baud = 9600

x = serial.Serial(com, baud, timeout=0.1)

while x.isOpen() == True:
    data = str(x.readline().decode('utf-8')).rstrip()
    if data:

        values = data.split(',')  # Split the data by comma
        values.pop()
        if len(values) == 4:  # Ensure there are 4 values
            print(values)
            with open('SensorData.csv', mode='a', newline= '') as sensor_file:
                sensor_writer = csv.writer(sensor_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                sensor_writer.writerow([float(val) for val in values] +[str(plant_type)] [str(time.asctime())])
