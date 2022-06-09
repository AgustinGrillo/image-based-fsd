#ifndef FSCPP__JOINT_H
#define FSCPP__JOINT_H

#include <sstream>
#include <fscpp/i2c.h>

#include <stdlib.h>
#include <math.h>
#include <stdexcept>
#include "ros/ros.h"
#include <angles/angles.h>

#define BASE_SLAVE_ADDRESS 0x71


#define ACTUATOR_TYPE_NONE 4

#define ACTUATOR_TYPE_VELOCITY_MOTOR 0    //PID Runs in ROS
#define ACTUATOR_TYPE_POSITION_MOTOR 1    //PID Runs in ROS
#define ACTUATOR_TYPE_VELOCITY_SERVO 2    //PID Runs inside MOTOR
#define ACTUATOR_TYPE_POSITION_SERVO 3    //PID Runs inside MOTOR

#define ACTUATOR_TYPE_NONE_SERVO 5

namespace fscpp
{
	class Joint
	{
		private:
			uint8_t _motorId = 0;
			uint8_t _actuatorType = 0;
			uint8_t _getSlaveAddress();
			uint8_t _minServoValue = 0;
			uint8_t _maxServoValue = 75;
			double _previousEffort;
			double _filterAngle(double angle);
			int _angleReads = 0;
			static const int _filterPrevious = 3;
			double _previousAngles[_filterPrevious];
			void _prepareI2CWrite(uint8_t result[3], int16_t effort);
			void _prepareI2CRead(uint8_t result[4]);
		public:
			std::string name;
			Joint();
			Joint(uint8_t motorId);
			~Joint();
			double sensorResolution = 1024;
			double angleOffset = 0;
			double readRatio = 1;
			double actuatorType = 0;
			uint8_t getMotorId();
			void setMotorId(uint8_t motorId);
			void setActuatorType(uint8_t actuatorType);
			void setServoLimits(uint8_t minValue, uint8_t maxValue);
			int getActuatorType();
			double getPreviousEffort();
			void actuate(int16_t command, uint8_t duration);
			double readAngle();
	};
}

#endif
