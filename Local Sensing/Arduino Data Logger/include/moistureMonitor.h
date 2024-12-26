#ifndef moistureMonitor_H
#define moistureMonitor_H

#include <Arduino.h>

#define l1digital 3
#define l2digital 4
#define u1digital 5
#define u2digital 6

#define l1analog A0
#define l2analog A1
#define u1analog A2
#define u2analog A3

float readMoisture(int sensorPin, int dryValue = 0, int wetValue = 1023, float minPercent = 0.0, float maxPercent = 100.0) {
  int rawValue = analogRead(sensorPin);
  float mappedValue = map(rawValue, dryValue, wetValue, minPercent * 100, maxPercent * 100) / 100.0;
  return constrain(mappedValue, minPercent, maxPercent);
}
float getMoistureAverage(){
    float l1 = readMoisture(l1analog); 
    float l2 = readMoisture(l2analog);
    float u1 = readMoisture(u1analog);
    float u2 = readMoisture(u2analog);
    
    unsigned long value = (l1 + l2 + u1 + u2)/4;
    return value;
}
#endif 