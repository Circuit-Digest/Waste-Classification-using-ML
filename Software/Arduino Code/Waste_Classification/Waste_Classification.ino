#include <AccelStepper.h>
#include <ESP32Servo.h>

#define GRIPPER_PIN 7  // Pin for the servo controlling the gripper
#define WRIST_PIN 1    // Pin for the servo controlling wrist rotation

// Define the stepper motors and the pins they will use
AccelStepper stepper1(1, 21, 20); // (Type:driver, STEP, DIR)
AccelStepper stepper2(1, 19, 18);
AccelStepper stepper3(1, 22, 23);

Servo gripperServo;  // create servo object to control a servo
Servo wristServo;    // Servo object for the wrist rotation

const float theta1AngleToSteps = 14.55;  // Conversion factor for theta1
const float theta2AngleToSteps = 11.43;  // Conversion factor for theta2
const float zAngleToSteps = 400;         // Conversion factor for phi
const int gripperCloseAngle = 140;
const int gripperOpenAngle = 80;

//int initialise = 0;

void setup() {
    Serial.begin(115200); // USB communication for debugging
    // Stepper motors max speed and acceleration
    stepper1.setMaxSpeed(4000);
    stepper1.setAcceleration(2000);
    stepper2.setMaxSpeed(4000);
    stepper2.setAcceleration(2000);
    stepper3.setMaxSpeed(4000);
    stepper3.setAcceleration(2000);

    gripperServo.attach(GRIPPER_PIN);
    gripperServo.write(gripperOpenAngle); // Open gripper by default

    wristServo.attach(WRIST_PIN); // Pin for wrist control
    wristServo.write(61); // Default wrist angle*/
}

String ID = "NULL";

void loop() {   
    if (Serial.available()) {
        String command = Serial.readStringUntil('\n');
        command.trim();
        if (command == "BIODEGRADABLE" || command == "PAPER" || command == "CLOTH")
        ID = "organic";
        if (command == "CARDBOARD" || command == "GLASS" || command == "METAL" || command == "PLASTIC")
        ID = "inorganic";
        executeCommand(ID);

        ID = "NULL";
    }   
}

void executeCommand(String ID) {
  
    //pick up the organic object and drop command
    if(ID.equalsIgnoreCase("organic")){
        moveToPosition(0,510,-70);      
        gripperServo.write(gripperCloseAngle); // Close gripper
        moveToPosition(0,510,0);  
        moveToPosition(-120,0,0);   //before moveToPosition(150,500,-70);
        gripperServo.write(gripperOpenAngle); // Open gripper
       
        moveToPosition(0,0,0); //HOME POSITION FOR ORGANIC 
      //clear serial buffer
      while(Serial.available()){
        Serial.read(); //read and discard anything in buffer
      }  
      Serial.println("READY");
    }

    //pick up the in-organic object and drop command
    else if(ID.equalsIgnoreCase("inorganic")){
      moveToPosition(0,510,-70);      
      gripperServo.write(gripperCloseAngle); // Close gripper
      moveToPosition(0,510,0);
      moveToPosition(120,1020,0);   //before moveToPosition(150,500,-70);
      gripperServo.write(gripperOpenAngle); // Open gripper 

      moveToPosition(0,1020,0);  //home position for in-organic
      //clear serial buffer
      while(Serial.available()){
        Serial.read(); //read and discard anything in buffer
      }
      Serial.println("READY");       
    }
    
    // Unknown command
    else {
        Serial.println("Unknown command");
    }
}

// Function to move to position
void moveToPosition(int x, int y, int z) {
  
    stepper1.moveTo(x * theta1AngleToSteps);
    stepper2.moveTo(y * theta2AngleToSteps);
    stepper3.moveTo(z * zAngleToSteps);
    
    while (stepper1.isRunning() || stepper2.isRunning() || stepper3.isRunning()) {
        stepper1.run();
        stepper2.run();
        stepper3.run();
    }
}

void setGripperZ(int gripperZ){
    wristServo.write(gripperZ); // Set wrist position
}
