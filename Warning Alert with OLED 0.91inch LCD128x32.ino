#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 32 // OLED display height, in pixels

const int LED_PIN = D0; 

// Declaration for an SSD1306 display connected to I2C (SDA, SCL pins)
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

void setup() {
  Serial.begin(9600);
   pinMode(LED_PIN, OUTPUT);
  

  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { // Address 0x3D for 128x64
    Serial.println(F("SSD1306 allocation failed"));
    for(;;);
  }
  delay(2000);
  display.clearDisplay();

  display.setTextSize(1.3);
  display.setTextColor(WHITE);
  display.setCursor(0, 12);
  // Display static text
  display.println("Normal");
  display.display(); 
}


void loop() {

  if (Serial.available() > 0) {
     digitalWrite(LED_PIN, HIGH);
     alert();
     while (Serial.available() > 0){
       blink();
     }
     
     //blink();
    String data = Serial.readStringUntil('\n');
     Serial.println(data);
    if (data == "on") {
      digitalWrite(LED_PIN, HIGH);
      //digitalWrite(BUZZER_PIN, HIGH);
    } else {
      digitalWrite(LED_PIN, LOW);
      clearalert();
      //digitalWrite(BUZZER_PIN, LOW);
    }
  }
  else{
    digitalWrite(LED_PIN, LOW);
    clearalert();

  }

 
}

void blink(){
   // Blink "display" with a one-second interval
   display.invertDisplay(true);
  delay(500);
  display.invertDisplay(false);
  delay(500);
  
}
void alert(){
  display.clearDisplay();

  display.setTextSize(1.3);
  display.setTextColor(WHITE);
  display.setCursor(0, 12);
  // Display static text
  display.println("Alert : Drowning!");
  display.display();
}
void clearalert(){
  display.clearDisplay();

  display.setTextSize(1.3);
  display.setTextColor(WHITE);
  display.setCursor(0, 12);
  // Display static text
  display.println("Normal");
  display.display();
}
