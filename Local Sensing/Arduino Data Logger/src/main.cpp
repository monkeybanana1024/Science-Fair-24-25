#include <Arduino.h>
#include "compileSerial.h"
#include "moistureMonitor.h"
#include "seismicMonitor.h"

String timestamp = "2024-11-02 10:15:30";
float waterFlowRate = 50.0;
int moisture = 75;

bool isSending = false;

#define l1digital 2
#define l2digital 3
#define u1digital 4
#define u2digital 5 

SeismicMonitor seismicMonitor;

void setup() {
  Serial.begin(9600);
  pinMode(l1digital, OUTPUT);
  pinMode(l2digital, OUTPUT);
  pinMode(u1digital, OUTPUT);
  pinMode(u2digital, OUTPUT);

  if (!seismicMonitor.initSeismicMonitor()) {
    Serial.println("Failed to initialize seismic monitor");
    // Handle the error as appropriate
  }
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (input == "START") {
      isSending = true;
    } else if (input == "STOP") {
      isSending = false;
    } else if (input.startsWith("FLOW:")) {
      // Update water flow rate from Python GUI
      waterFlowRate = input.substring(5).toFloat();
    }
  }

  if (isSending) {

    digitalWrite(l1digital, HIGH);
    digitalWrite(l2digital, HIGH);
    digitalWrite(u1digital, HIGH);
    digitalWrite(u2digital, HIGH);

    Serial.println(compileSerial(

      waterFlowRate, 
      getMoistureAverage(), 
      seismicMonitor.movementX(), 
      seismicMonitor.movementY(), 
      seismicMonitor.movementZ()
      
      ));

    digitalWrite(l1digital, LOW);
    digitalWrite(l2digital, LOW);
    digitalWrite(u1digital, LOW);
    digitalWrite(u2digital, LOW);

    delay(100);  
    }
}
