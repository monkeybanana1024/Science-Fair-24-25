#ifndef SEISMIC_MONITOR_H
#define SEISMIC_MONITOR_H

#include <Arduino.h>
#include <Wire.h>
#include <MPU6050.h>

class SeismicMonitor {
private:
    MPU6050 mpu;
    const float ACCEL_SCALE = 16384.0;  // For +/- 2g range
    const float GYRO_SCALE = 131.0;     // For +/- 250 deg/s range

    float mapFloat(float x, float in_min, float in_max, float out_min, float out_max) {
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
    }

public:
    SeismicMonitor() {}

    bool initSeismicMonitor() {
        Wire.begin();
        mpu.initialize();
        
        if (!mpu.testConnection()) {
            Serial.println("MPU6050 connection failed");
            return false;
        }

        mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);
        mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);

        return true;
    }

    float movementX() {
        int16_t ax, ay, az, gx, gy, gz;
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

        float accel = ax / ACCEL_SCALE;
        float tilt = gx / GYRO_SCALE;

        // Combine acceleration and tilt
        float combined = sqrt(accel * accel + tilt * tilt);

        // Map the combined value to 0-5 range
        return mapFloat(combined, 0, 2, 0, 5                                                                );  // Assuming max combined value of 2
    }

    float movementY() {
        int16_t ax, ay, az, gx, gy, gz;
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

        float accel = ay / ACCEL_SCALE;
        float tilt = gy / GYRO_SCALE;

        float combined = sqrt(accel * accel + tilt * tilt);
        return mapFloat(combined, 0, 2, 0, 5);
    }

    float movementZ() {
        int16_t ax, ay, az, gx, gy, gz;
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

        float accel = az / ACCEL_SCALE;
        float tilt = gz / GYRO_SCALE;

        float combined = sqrt(accel * accel + tilt * tilt);
        return mapFloat(combined, 0, 2, 0, 5);
    }
};

#endif