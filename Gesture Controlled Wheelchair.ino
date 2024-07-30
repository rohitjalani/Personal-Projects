// Include required libraries
#include "Wire.h"
#include "I2Cdev.h"
#include "MPU6050.h"

// Initialize the MPU6050 object and create variables to hold the sensor data
MPU6050 mpu;
int16_t ax, ay, az;
int16_t gx, gy, gz;

// Define the motor connections
// Motor A connections
int enA = 9;
int in1 = 8;
int in2 = 7;
// Motor B connections
int enB = 3;
int in3 = 5;
int in4 = 4;

// Define a custom data structure to hold the mapped sensor data
struct MyData {
byte X;
byte Y;
};

// Create an instance of the custom data structure
MyData data;

// Set up the Arduino board
void setup()
{
// Set the motor pins as outputs
pinMode(enA, OUTPUT);
pinMode(enB, OUTPUT);
pinMode(in1, OUTPUT);
pinMode(in2, OUTPUT);
pinMode(in3, OUTPUT);
pinMode(in4, OUTPUT);

// Turn off the motors - initial state
digitalWrite(in1, LOW);
digitalWrite(in2, LOW);
digitalWrite(in3, LOW);
digitalWrite(in4, LOW);

// Start the serial communication
Serial.begin(115200);

// Start the I2C communication with the MPU6050 sensor
Wire.begin();

// Initialize the MPU6050 sensor
mpu.initialize();

// Set the built-in LED pin as an output (commented out)
// pinMode(LED_BUILTIN, OUTPUT);
}

// Main loop function
void loop()
{
  // Read the sensor data from the MPU6050
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // Map the sensor data to a custom data structure with byte values between 0 and 255
  data.X = map(ax, -17000, 17000, 0, 255 ); // Map the X axis data
  data.Y = map(ay, -17000, 17000, 0, 255); // Map the Y axis data

  // Add a delay to prevent the code from running too quickly
  delay(500);

  // Print the mapped sensor data to the serial monitor
  Serial.print("Axis X = ");
  Serial.print(data.X);
  Serial.print(" ");
  Serial.print("Axis Y = ");
  Serial.println(data.Y);

  // Control the motors based on the mapped sensor data
  if (data.Y < 80) 
  { // Gesture: Down (FORWARD MOVEMENT)
    Serial.println("gesture 1");
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);
    analogWrite(enA, 150);
    analogWrite(enB, 150);
  }

  if (data.Y > 145) 
  { // Gesture: Up (BACKWARD MOVEMENT)
  //digitalWrite(LED_BUILTIN, HIGH);
    Serial.println("gesture 2");
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
    analogWrite(enA, 150);
    analogWrite(enB, 150);
  }

  if (data.X > 155) 
  { // Gesture: Left (FORWARD MOVEMENT)
    Serial.println("gesture 3");
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);
    analogWrite(enA, 150);
    analogWrite(enA, 150);
    analogWrite(enB, 150);
  }

  if (data.X < 80) 
  {//gesture : right ( BACKWARD MOVEMENT)
    Serial.println("gesture 4");
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
    analogWrite(enA, 150);
    analogWrite(enB, 150);
  }

  if (data.X > 100 && data.X < 170 && data.Y > 80 && data.Y < 130) 
  { //gesture : little bit down (FORWARD MOVEMENT)
    Serial.println("gesture 5");
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);
    analogWrite(enA, 150);
    analogWrite(enB, 150);
  }
}