#ifndef COMPILE_SERIAL_H
#define COMPILE_SERIAL_H
#include <Arduino.h>

int time = 0;

void initializeSerial() {
  Serial.begin(9600);
}
String compileSerial(float flow, float moisture, float moveX, float moveY, float moveZ) {
    time = time + 1;
    String value = String(time) + "," + String(flow) + "," + String(moisture) + ',' + String(moveX) + ',' + String(moveY) + ',' + String(moveZ);
    return value;
}


#endif