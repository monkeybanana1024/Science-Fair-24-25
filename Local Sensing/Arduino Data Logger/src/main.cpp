#include <Arduino.h>
#include "compileSerial.h"
#include "moistureMonitor.h"
#include "seismicMonitor.h"

// Constants for flow rate sensor
#define FLOW_SENSOR_PIN 2  // The pin to which the flow sensor is connected

// FlowRateSensor class definition
class FlowRateSensor {
public:
    FlowRateSensor(int pin)
        : sensorPin(pin),
          flowRate(0.0),
          lastPulseTime(0),
          calibrationFactor(7.5) {}

    // Initialize the sensor pin and attach interrupt
    void begin() {
        pinMode(sensorPin, INPUT);
        attachInterrupt(digitalPinToInterrupt(sensorPin), pulseInterrupt, FALLING);
    }

    // Update flow rate based on pulse count
    void updateFlowRate() {
        unsigned long currentTime = millis();
        if (currentTime - lastPulseTime >= 100) {  // Update every 100 milliseconds
            flowRate = (FlowRateSensor::pulseCount / calibrationFactor); // Access static pulseCount
            FlowRateSensor::pulseCount = 0;  // Reset pulse count after updating the rate
            lastPulseTime = currentTime;
        }
    }

    // Get the current flow rate
    float getFlowRate() const {
        return flowRate;
    }

    // Static interrupt handler
    static void pulseInterrupt() {
        FlowRateSensor::pulseCount++;  // Increment pulse count
    }

private:
    int sensorPin;              // Pin connected to the flow sensor
    static volatile int pulseCount;  // Static count of pulses
    float flowRate;             // Flow rate in L/min or similar
    unsigned long lastPulseTime;  // Last time flow rate was updated
    float calibrationFactor;    // Calibration factor for the sensor
};

// Initialize the static pulse count variable
volatile int FlowRateSensor::pulseCount = 0;

// Global instance of FlowRateSensor
FlowRateSensor flowSensor(FLOW_SENSOR_PIN);

bool isSending = false;

// Pin definitions for other sensors and devices
#define l1digital 3
#define l2digital 4
#define u1digital 5
#define u2digital 6

// Corrected SeismicMonitor instantiation
SeismicMonitor seismicMonitor;

void setup() {
    Serial.begin(115200);  // Initialize serial communication
    pinMode(l1digital, OUTPUT);
    pinMode(l2digital, OUTPUT);
    pinMode(u1digital, OUTPUT);
    pinMode(u2digital, OUTPUT);

    if (!seismicMonitor.initSeismicMonitor()) {
        Serial.println("Failed to initialize seismic monitor");
        // Handle the error as appropriate
    }

    flowSensor.begin();  // Initialize flow sensor and attach interrupt
}

void loop() {
    if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        input.trim();
        
        if (input == "START") {
            isSending = true;
        } else if (input == "STOP") {
            isSending = false;
        }
    }

    if (isSending) {
        flowSensor.updateFlowRate();  // Update flow rate every 100 milliseconds

        // Control digital pins for other sensors or devices
        digitalWrite(l1digital, HIGH);
        digitalWrite(l2digital, HIGH);
        digitalWrite(u1digital, HIGH);
        digitalWrite(u2digital, HIGH);

        // Print data to serial monitor
        Serial.println(compileSerial(
            flowSensor.getFlowRate(),    // Flow rate data
            getMoistureAverage(),        // Moisture data
            seismicMonitor.movementX(),  // Seismic X data
            seismicMonitor.movementY(),  // Seismic Y data
            seismicMonitor.movementZ()   // Seismic Z data
        ));

        // Reset control pins
        digitalWrite(l1digital, LOW);
        digitalWrite(l2digital, LOW);
        digitalWrite(u1digital, LOW);
        digitalWrite(u2digital, LOW);

        delay(100);  // Small delay for better readability in serial monitor
    }
}
