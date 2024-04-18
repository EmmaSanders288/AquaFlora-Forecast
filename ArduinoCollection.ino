#include "DHTStable.h"
// Define pin configurations
#define LDRpin A0         // Pin where the LDR and resistor are connected
#define DHT11_PIN 7       // Pin to connect the temperature & humidity sensor
#define SOIL_PIN A5       // Pin to connect the soil moisture sensor

// Initialize DHT object
DHTStable DHT;

// Variables to store sensor readings
int LDRValue = 0;            // Result of reading the analog pin
const int dry = 600; // value for dry sensor
const int wet = 180; // value for wet sensor

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
}

void loop() {
  // Read LDR sensor value
  LDRValue = analogRead(LDRpin);

  // Read temperature and humidity from DHT11 sensor
  float chk = DHT.read11(DHT11_PIN);
  float humidity = DHT.getHumidity();
  float temperature = DHT.getTemperature();

  // Read soil moisture value
  float soilHumidity = analogRead(SOIL_PIN);
  int percentageSoilHumididy = map(soilHumidity, wet, dry, 100, 0); 
  // printDataToRead(LDRValue, humidity, temperature, soilHumidity);
  printDataToSend(LDRValue, humidity, temperature, percentageSoilHumididy);
  delay(500);                    // Delay before next reading
}

void printDataToRead(float LDRValue, float humidity, float temperature, float percentageSoilHumidity){
  // Print sensor readings
  Serial.print("Light = ");
  Serial.println(LDRValue);      // Print the LDR value
  Serial.print("Temperature = ");
  Serial.println(temperature);   // Print temperature
  Serial.print("Humidity = ");
  Serial.println(humidity);      // Print humidity
  Serial.print("Soil humidity = ");
  Serial.println(percentageSoilHumidity); // Print soil humidity

}

void printDataToSend(float LDRValue, float humidity, float temperature, float percentageSoilHumidity){
  Serial.print(LDRValue); 
  Serial.print(","); 
  Serial.print(temperature); 
  Serial.print(","); 
  Serial.print(humidity); 
  Serial.print(",");
  Serial.print(percentageSoilHumidity); 
  Serial.print(",");  
  Serial.println();
}