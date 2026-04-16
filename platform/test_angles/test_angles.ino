#include <Servo.h>

int HORIZONTAL_PIN = 4;
int VERTICAL_PIN = 3;

Servo horiz;
Servo vert;

void setup() {
  horiz.attach(HORIZONTAL_PIN, 550, 2450);
  vert.attach(VERTICAL_PIN, 550, 2450);
  Serial.begin(115200);
  //   // left
  // for(int i = 45; i >= 0; i--) {
  //   horiz.write(i);
  //   delay(1000);
  //   Serial.println(String("Left ") + i);
  // } // 0
  // //right
  // for(int i = 135; i <= 180; i++) {
  //   horiz.write(i);
  //   delay(1000);
  //   Serial.println(String("Right ") + i);
  // } // 180
  //top
  // for(int i = 60; i >= 39; i--) {
  //   vert.write(i);
  //   delay(1000);
  //   Serial.println(String("Top ") + i);
  // }  // 40
  //bottom
  // horiz.write(180);
  // for(int i = 135; i <= 140; i++) {
  //   vert.write(i);
  //   delay(1000); 
  //   Serial.println(String("Bottom ") + i);
  // } // 168
  vert.writeMicroseconds(1500);
  for(int i = 1500; i >= 1000; i--) {
    vert.writeMicroseconds(i);
    Serial.println(String("Bottom ") + i);
  } // 1000 2250
}

void loop() {
  delay(10000);
}
