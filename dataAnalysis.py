import pandas as pd
import csv

''''
plantData = pd.read_csv("C:/Users/anna/PycharmProjects/AquaFlora-Forecast/SensorData.csv", sep=';')
print(plantData.dtypes)
print(plantData.describe())
print(plantData.shape)
plantData.dropna()
print(plantData.shape)
plantData = plantData[plantData["Light"] > 0]
plantData = plantData[plantData["Temperature"] > 0]
plantData = plantData[plantData["Humidity"] > 0]
plantData = plantData[plantData["Soil_humidity"] > 0]
print(plantData.shape)
new_plantData = plantData.drop_duplicates(subset=['Light', 'Temperature', 'Humidity', 'Soil_humidity', 'Plant Type'], keep='last')
print(new_plantData.shape)
'''

# new_plantData.to_csv("C:/Users/anna/PycharmProjects/AquaFlora-Forecast/PlantData.csv", sep=';')

labelData = pd.read_csv("C:/Users/anna/PycharmProjects/AquaFlora-Forecast/PlantData.csv", sep=';')
print(labelData.shape)
new_labelData = labelData.drop(columns=['Kolom1', 'Kolom2'])
print(new_labelData.shape)
new_labelData.to_csv("C:/Users/anna/PycharmProjects/AquaFlora-Forecast/PlantDataLabels.csv", sep=';')

