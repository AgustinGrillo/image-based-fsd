

#include <fscpp/joint.h>


#define PI 3.14159265359
#define TAU 6.28318530718

namespace fscpp
{
	Joint::Joint()
	{
		
	}

	Joint::Joint(uint8_t motorId)
	{
		setMotorId(motorId);
	}

	Joint::~Joint()
	{

	}

	void Joint::setActuatorType(uint8_t actuatorType)
	{
		this->_actuatorType = actuatorType;
	}

	uint8_t Joint::getMotorId()
	{
		return this->_motorId;
	}

	void Joint::setMotorId(uint8_t motorId)
	{
		this->_motorId = motorId;
	}

	double Joint::_filterAngle(double angle)
	{
		_angleReads = _angleReads + 1;

		// put value at front of array
		for (int i = _filterPrevious - 1; i > 0; i--) {
			_previousAngles[i] = _previousAngles[i - 1];
		}
		_previousAngles[0] = angle;


		int filterIterations = _filterPrevious;
		if (_angleReads < _filterPrevious) {
			filterIterations = _angleReads;
		}

		double angleSum = 0;
		for (int i = 0; i < filterIterations; i++) {
			angleSum = angleSum + _previousAngles[i];
		}

		double filterResult = angleSum / (filterIterations * 1.0);

		//ROS_INFO("%f, %f, %f, %i", angle, angleSum, filterResult, filterIterations);

		return filterResult;
	}

	double Joint::readAngle()
	{
        uint32_t message;
        I2C i2cSlave = I2C(0, _getSlaveAddress());
        uint8_t result = i2cSlave.readBytes(_motorId, 4, message);


        switch (_actuatorType) {
            //case ACTUATOR_TYPE_VELOCITY_MOTOR: {
            //    break;
            //}
            //case ACTUATOR_TYPE_POSITION_MOTOR: {
            //    break;
            //}
            case ACTUATOR_TYPE_VELOCITY_MOTOR:
            case ACTUATOR_TYPE_POSITION_MOTOR:
            case ACTUATOR_TYPE_VELOCITY_SERVO:
            case ACTUATOR_TYPE_NONE:    {

                auto encoder_ticks = (int32_t) message;
                ROS_INFO("Result: [%i]; MOTOR_ID: [%i]; Encoder Ticks: [%i] \n", result, _motorId,
                         encoder_ticks);

                if (result == 1)
                {

                    double angle = TAU * (encoder_ticks/ sensorResolution);
                    angle = _filterAngle(angle);
                    angle += angleOffset;
                    angle *= readRatio;
                    return angle;
                }
                else
                {
                    //throw std::runtime_error("I2C Read Error during joint position read. Exiting for safety.");
                }
                break;
            }
            case ACTUATOR_TYPE_POSITION_SERVO:
            case ACTUATOR_TYPE_NONE_SERVO:{

                auto angle_deg = (int32_t) message;
                ROS_INFO("Result: [%i]; MOTOR_ID: [%i]; Angle: [%i] \n", result, _motorId,
                         angle_deg);

                if (result == 1)
                {

                    double angle = angles::from_degrees((double) angle_deg);
                    angle = _filterAngle(angle);
                    angle += angleOffset;
                    angle *= readRatio;
                    return angle;
                }
                else
                {
                    //throw std::runtime_error("I2C Read Error during joint position read. Exiting for safety.");
                }
                break;

            }
 
        }


	}

	void Joint::actuate(int16_t command, uint8_t duration = 15)
	{

        uint8_t data[3];
        data[2] = duration;
        _prepareI2CWrite(data, command);
        I2C i2cSlave = I2C(0, _getSlaveAddress());
        uint8_t result = i2cSlave.writeData(_motorId, data);

        ROS_INFO("Result: [%i]; MOTOR_ID: [%i]; Command: [%i]; bytes: %i, %i, %i  \n", result, _motorId, command, data[0], data[1], data[2]);

	}

	uint8_t Joint::_getSlaveAddress()
	{
		if (_motorId >= 0 && _motorId <= 2)
		{
			return BASE_SLAVE_ADDRESS;
		}
		else
		{
			ROS_ERROR("Invalid MotorID: %i", _motorId);
			return -1;
		}
	}

	void Joint::setServoLimits(uint8_t minValue, uint8_t maxValue)
	{
		this->_minServoValue = minValue;
		this->_maxServoValue = maxValue;
	}

	double Joint::getPreviousEffort() {
		return this->_previousEffort;
	}

	void Joint::_prepareI2CWrite(uint8_t result[3], int16_t command)
	{


			result[0] = command;
			result[1] = command >> 8;

			//ROS_INFO("name: %s, minServoValue: %i, maxServoValue: %i, effort: %f, magnitude: %f, servoValue: %i", name.c_str(), _minServoValue, _maxServoValue, effort, magnitude, servoValue);

	}
	
	int Joint::getActuatorType()
	{
		return _actuatorType;
	}
}
