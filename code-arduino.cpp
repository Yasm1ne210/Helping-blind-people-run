//This code is the one used in the Arduino IDE to establish the connection with the rest of our components and link the hardware part to the software part.

const int buzzerPin = 3;  // Piezo buzzer connected to pin 3

const int TONE_LEFT = 1000;      // D4
const int TONE_RIGHT = 500;     // D5
const int TONE_OBSTACLE = 880;  // A5

const int SHORT_BEEP = 150;
const int OBSTACLE_BEEP = 120;

String lastCommand = "";
bool hasBeeped = false;

void setup() {
  Serial.begin(9600);
  pinMode(buzzerPin, OUTPUT);

  // Startup tones
  tone(buzzerPin, 523, 150); delay(200);
  tone(buzzerPin, 659, 150); delay(200);
  tone(buzzerPin, 784, 150); delay(300);

  Serial.println("System ready for directional and obstacle commands");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    // Only beep if command is new or has changed
    if (command != lastCommand) {
      lastCommand = command;
      hasBeeped = false; 
    }
  }

  if (!hasBeeped) {
    if (lastCommand == "LEFT") {
      tone(buzzerPin, TONE_LEFT, SHORT_BEEP);
      hasBeeped = true;
    }
    else if (lastCommand == "RIGHT") {
      tone(buzzerPin, TONE_RIGHT, SHORT_BEEP);
      hasBeeped = true;
    }
    else if (lastCommand.indexOf("CAUTION") >= 0) {
      tone(buzzerPin, TONE_OBSTACLE, OBSTACLE_BEEP);
      hasBeeped = true;
    }
    // All other commands = no sound
  }
}

