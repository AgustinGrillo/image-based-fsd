// PID motor position control.
// Motor used in this code https://www.pololu.com/product/2824  (Note: At 12 V maximum speed is 100 RPM=600 deg/s
// Driver used in this code https://core-electronics.com.au/vnh5019-motor-driver-carrier.html

// Thanks to Brett Beauregard for his nice PID library http://brettbeauregard.com/blog/2011/04/improving-the-beginners-pid-introduction/

//#include <PinChangeInt.h> This library is to allow more pins to attach interrupts
#include <Wire.h>
#include <PID_v1.h>
#include <Servo.h>
#include <util/atomic.h> // this library includes the ATOMIC_BLOCK macro.






//Motor Steer
Servo steer;




//MOTOR RIGHT
#define encodPinA1_R      2                       // Quadrature encoder A pin
#define encodPinB1_R      8                       // Quadrature encoder B pin
#define M_R               5                       // PWM outputs to Motor Driver module
#define INA_R             7
#define INB_R             4
#define PPR_R         800.0                      // Pulses per Revolution Output Shaft (One Edges, One channels)   (3200 is both edges, both channels)


//MOTOR LEFT
#define encodPinA1_L      3                       // Quadrature encoder A pin
#define encodPinB1_L     10                       // Quadrature encoder B pin
#define M_L               6                       // PWM outputs to Motor Driver module (Before was pin 9 but the pwm was not working)
#define INA_L            13 
#define INB_L            12
#define PPR_L         800.0                      // Pulses per Revolution Output Shaft (One Edges, One channels)   (3200 is both edges, both channels)

#define debug_serial    0
#define OFFSET_SERVO 94

unsigned long lastTime,now;
uint8_t sel_motor_id=10;


//MOTOR RIGHT

double kp_R =0.01, ki_R =1 , kd_R =0;             // modify for optimal performance
double input_R = 0, output_R = 0, setpoint_R = 0;
volatile long encoderPos_R = 0,last_pos_R=0;
PID motorR_PID(&input_R, &output_R, &setpoint_R, kp_R, ki_R, kd_R,DIRECT);


//MOTOR LEFT

double kp_L =0.01, ki_L =1 , kd_L =0;             // modify for optimal performance
double input_L = 0, output_L = 0, setpoint_L = 0;
volatile long encoderPos_L = 0,last_pos_L = 0;
PID motorL_PID(&input_L, &output_L, &setpoint_L, kp_L, ki_L, kd_L,DIRECT);


//MOTOR STEER

int16_t angle_steer = 0;   //Percentage steer goes from 0 to 270 for the 20kg servo

//Note angle_steer with the steering attached to the vehicle SHOULD NOT EXCEED -35 or 35

void setup() {

  //MOTOR STEER

  steer.attach(11);
  steer.write(map( OFFSET_SERVO, 0, 270, 0, 180));
  //steer.writeMicroseconds(value);


  
  
  //MOTOR RIGHT
  pinMode(encodPinA1_R, INPUT_PULLUP);                  // quadrature encoder input A
  pinMode(encodPinB1_R, INPUT_PULLUP);                  // quadrature encoder input B
  pinMode(INA_R, OUTPUT);
  pinMode(INB_R, OUTPUT);
  attachInterrupt(0, encoder_R, RISING);               // update encoder position (0 maps interrupt to pin 2 automatically (?)) use digitalPinToInterrupt(pin) instead
  //TCCR1B = TCCR1B & 0b11111000 | 1;                   // To prevent Motor Noise  (Modifies frequency of PWM)
  
  motorR_PID.SetMode(AUTOMATIC);
  motorR_PID.SetSampleTime(250);
  motorR_PID.SetOutputLimits(-255, 255);


  //MOTOR LEFT

  pinMode(encodPinA1_L, INPUT_PULLUP);                  // quadrature encoder input A
  pinMode(encodPinB1_L, INPUT_PULLUP);                  // quadrature encoder input B
  pinMode(INA_L, OUTPUT);
  pinMode(INB_L, OUTPUT);
  attachInterrupt(1, encoder_L, RISING);               // update encoder position (0 maps interrupt to pin 2 automatically (?)) use digitalPinToInterrupt(pin) instead
  //TCCR1B = TCCR1B & 0b11111000 | 1;                   // To prevent Motor Noise
  
  motorL_PID.SetMode(AUTOMATIC);
  motorL_PID.SetSampleTime(250);
  motorL_PID.SetOutputLimits(-255, 255);

 
  Wire.begin(0x71);                // join i2c bus with address #8 for Arduino
  Wire.onRequest(requestEvent); // register events
  Wire.onReceive(receiveEvent);
  
  if (debug_serial==1)
  {
  Serial.begin (9600);
  }
  
  setpoint_L = (double)0;
  setpoint_R = (double)0; 
  
}

void loop() {
  now = millis();
  int timeChange = (now - lastTime);

  if(timeChange>=100 ) //Maybe this value can be higher for less noisy measurement in speed
  {

    //Motor RIGHT
    
    input_R = (360.0*1000*(encoderPos_R-last_pos_R)) /(PPR_R*(now - lastTime)); // EL 1000 es para pasar de milis a segundos y PP_R son los pulsos por revolucion para el eje de salida que seria CPR/4 
    
    //Motor LEFT
    
    input_L = (360.0*1000*(encoderPos_L-last_pos_L)) /(PPR_L*(now - lastTime)); // EL 1000 es para pasar de milis a segundos y PP_R son los pulsos por revolucion para el eje de salida que seria CPR/4 
    
    lastTime=now;
    last_pos_R=encoderPos_R;
    last_pos_L=encoderPos_L;
    
    if (debug_serial==1)
    {
    
    Serial.print ("Speed_R = ");
    Serial.print(double(input_R)); //Print Speed in Deg/sec
    Serial.print (",   Setpoint_R = ");
    Serial.print(setpoint_R);
    
    Serial.print ("      Speed_L = ");
    Serial.print(double(input_L)); //Print Speed in Deg/sec
    Serial.print (",   Setpoint_L = ");
    Serial.print(setpoint_L);
    
    Serial.print ("      Steer = ");
    Serial.println(angle_steer); //Print Speed in Deg/sec
    
    
    }
  }
  
    

  motorR_PID.Compute();                                    // calculate new output
  pwmOut(output_R, M_R, INA_R, INB_R);// drive L298N H-Bridge module

  motorL_PID.Compute();
  pwmOut(output_L,M_L, INA_L, INB_L);// drive L298N H-Bridge module


  if (angle_steer > 35)
  {
    angle_steer = 35;
  }
  else if (angle_steer < -35)
  {
    angle_steer = -35;
  }
  
 
  steer.write(map(angle_steer + OFFSET_SERVO, 0, 270, 0, 180));
  
  
  delay(10);
}

void pwmOut(int out, uint8_t motor, uint8_t IN_A, uint8_t IN_B) {                                // to H-Bridge board
  if (out > 0) {
    
    digitalWrite(IN_A,HIGH);
    digitalWrite(IN_B,LOW);
    analogWrite(motor, out);                             // drive motor CW

  }
  else {
    
    digitalWrite(IN_A,LOW);
    digitalWrite(IN_B,HIGH);
    analogWrite(motor, abs(out));                        // drive motor CCW
  }
}

void encoder_R()  {                                     // pulse and direction, direct port reading to save cycles (PINB is the port in arduino of PINS 8 to 13 
  if (PINB & 0b00000001)    encoderPos_R++;             // if(digitalRead(encodPinB1)==HIGH)   count --;
  else                      encoderPos_R--;             // if(digitalRead(encodPinB1)==LOW)   count ++;
}

void encoder_L()  {                                     // pulse and direction, direct port reading to save cycles (PINB is the port in arduino of PINS 8 to 13 
  if (PINB & 0b00000100)    encoderPos_L++;             // if(digitalRead(encodPinB1)==HIGH)   count --;
  else                      encoderPos_L--;             // if(digitalRead(encodPinB1)==LOW)   count ++;
}


void requestEvent() {
  int32_t s = 0;

  if (sel_motor_id == 0)
  {
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    s = encoderPos_R;
    }
  }
  else if (sel_motor_id ==1)
  {
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    s = -1*encoderPos_L;
    }
  }
  else if (sel_motor_id == 2)
  {
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    s = angle_steer;
    }
  } 

  uint8_t a = s;
  uint8_t b = s >> 8;
  uint8_t c = s >> 16;
  uint8_t d = s >> 24;
  
  Wire.write(d);
  Wire.write(c);  
  Wire.write(b);
  Wire.write(a);

}


// function that executes whenever data is received from master
// this function is registered as an event, see setup()
void receiveEvent(int howMany)
{
  uint8_t motor_id,cmd_a, cmd_b,duration,i,dummy;
  //uint16_t command_abs = 0;
  int16_t command = 0;

  if (howMany == 1)
  {
      sel_motor_id = Wire.read();
  }
  else if (howMany == 4)
  {
    motor_id = Wire.read();
    cmd_a = Wire.read();
    cmd_b = Wire.read();
    duration = Wire.read();

    command= (cmd_b<<8) + cmd_a;
    //command = read_command(command_abs, dir);

    
    if (motor_id == 0)
    {
      
      setpoint_R = command;
      
    }
    else if (motor_id ==1)
    {
      setpoint_L = -1*command;
      
    }
    else if (motor_id == 2)
    {
      angle_steer = command;
      //Serial.println(angle_steer);

    }
   

         
  }
  else
  {
      for(i=0;i<howMany;i++)
      {
        dummy = Wire.read();
      }

  }
      


}
