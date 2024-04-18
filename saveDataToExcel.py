import serial
import time
import csv

class SensorDataLogger:
    def __init__(self, com_port, baud_rate, plant_type, file_name='SensorData.csv'):
        # Initialize the SensorDataLogger with serial connection details, plant type, and file name
        self.plant_type = plant_type
        self.file_name = file_name
        self.serial_connection = serial.Serial(com_port, baud_rate, timeout=0.1)
        self.sensor_writer = None
        self.setup_csv()

    def setup_csv(self):
        # Setup CSV file and writer, write header row
        with open(self.file_name, mode='a', newline='') as file:
            self.sensor_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            self.sensor_writer.writerow(["Light", "Temperature", "Humidity", "Soil_humidity", 'Plant Type', "Time"])

    def log_data(self):
        # Read data from serial connection and log it to CSV file
        if self.serial_connection.isOpen():
            data = str(self.serial_connection.readline().decode('utf-8')).rstrip()
            if data:
                values = data.split(',')
                if len(values) == 4:  # Ensure there are 4 values
                    print(values)
                    with open(self.file_name, mode='a', newline='') as file:
                        self.sensor_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        self.sensor_writer.writerow([float(val) for val in values] + [self.plant_type, time.asctime()])

# Usage
logger = SensorDataLogger(com_port="COM5", baud_rate=9600, plant_type='Sanseveria')
while True:
    logger.log_data()
