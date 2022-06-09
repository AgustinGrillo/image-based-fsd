
#include <fscpp/fs.h>


namespace fscpp
{
	FS::FS()
	{
	    //Esto capaz puede ir en fs_hardware_interface
		//base
		base.joints[0].name = "JointBaseWheelBR";  //BR => BACK RIGHT
		base.joints[0].setMotorId(0);
        base.joints[0].sensorResolution = 800;

		base.joints[1].name = "JointBaseWheelBL";  //BL => BACK LEFT
		base.joints[1].setMotorId(1);
        base.joints[1].sensorResolution = 800;


        base.joints[2].name = "JointSteerRight";
        base.joints[2].setMotorId(2);
        base.joints[2].setServoLimits(0, 180);

        base.joints[3].name = "JointBaseWheelFR";
        base.joints[3].setMotorId(0);
        base.joints[3].sensorResolution = 800;

        base.joints[4].name = "JointBaseWheelFL";
        base.joints[4].setMotorId(1);
        base.joints[4].sensorResolution = 800;

        base.joints[5].name = "JointSteerLeft";
        base.joints[5].setMotorId(2);
        base.joints[5].setServoLimits(0, 180);



        //head.joints[0].sensorResolution = 128;
		//armRight.joints[6].setActuatorType(ACTUATOR_TYPE_SERVO);
		//armRight.joints[6].setServoLimits(0, 180);


		//armRight.joints[6].actuate(0, 15);
		//ROS_INFO("wrist set to 0");


	}

	FS::~FS()
	{

	}

	Joint FS::getJoint(std::string jointName)
	{

		int numJointsBase = sizeof(base.joints) / sizeof(base.joints[0]);
		for (int i = 0; i < numJointsBase; i++)
		{
			if (base.joints[i].name == jointName)
			{
				return base.joints[i];
			}
		}

		throw std::runtime_error("Could not find joint with name " + jointName);
	}

	void FS::setJoint(Joint joint)
	{
		bool foundJoint = false;


		int numJointsBase = sizeof(base.joints) / sizeof(base.joints[0]);
		for (int i = 0; i < numJointsBase; i++)
		{
			if (base.joints[i].name == joint.name)
			{
				foundJoint = true;
				base.joints[i] = joint;
			}
		}

		if (!foundJoint)
		{
			throw std::runtime_error("Could not find joint with name " + joint.name);
		}
	}
}
