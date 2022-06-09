
#include <fscpp/i2c.h>

namespace fscpp
{
	I2C::I2C(int bus, int address) {
		_i2cbus = bus;
		_i2caddr = address;
		snprintf(busfile, sizeof(busfile), "/dev/i2c-%d", bus);
		openfd();
	}

	I2C::~I2C() {
		close(fd);
	}

	uint8_t I2C::readBytes(uint8_t registerNumber, uint8_t bufferSize, uint32_t &position) {
		if (fd != -1)
		{
			uint8_t buff[bufferSize];

			uint8_t writeBufferSize = 1;
			uint8_t writeBuffer[writeBufferSize];
			writeBuffer[0] = registerNumber;

			if (write(fd, writeBuffer, writeBufferSize) != writeBufferSize)
			{
				ROS_ERROR("I2C slave 0x%x failed to go to register 0x%x [read_byte():write %d]", _i2caddr, registerNumber, errno);
				return (-1);
			}
			else
            {


				if (read(fd, buff, bufferSize) != bufferSize)
				{

					ROS_ERROR("Could not read from I2C slave 0x%x, register 0x%x [read_byte():read %d]", _i2caddr, registerNumber, errno);
					return (-1);
				}
				else
                {

					position = 0;
					for (int i = 0; i < bufferSize; i++)
					{
						int shift = pow(256, abs(i + 1 - bufferSize));
						position = position + (buff[i] * shift);
                        ROS_DEBUG("BUFF[%i]: %i", i, buff[i]);
						if (registerNumber == 2)
						{
							//ROS_INFO("%i: %i", i, buff[i]);
						}
					}

                    //CHEQUEAR ESTE ULTIMO PASO

					//uint32_t excessK = pow(256, bufferSize)/2;
					//position -= excessK;

					return (1);

				}
			}
		}
		else
        {
			ROS_ERROR("Device File not available. Aborting read");
			return (-1);
		}
	}

	uint8_t I2C::writeData(uint8_t registerNumber, uint8_t data[3]) {

		if (fd != -1)
		{
			uint8_t buff[4];
			buff[0] = registerNumber; //MOTOR_ID
			buff[1] = data[0]; //Command
            buff[2] = data[1]; //Command
			buff[3] = data[2]; //Duration



			int result = write(fd, buff, sizeof(buff));
			if (result != sizeof(buff))
			{
				ROS_ERROR("%s. Failed to write to I2C Slave 0x%x @ register 0x%x [write_byte():write %d]", strerror(errno), _i2caddr, registerNumber, errno);
				return (-1);
			}
			else
            {
				ROS_INFO("Wrote to I2C Slave 0x%x @ register 0x%x", _i2caddr,registerNumber);
				return result;
			}

		}
		else
        {
		    ROS_ERROR("Device File not available. Aborting write");
			return (-1);
		}

	}

	void I2C::openfd() {
		if ((fd = open(busfile, O_RDWR)) < 0) {
			ROS_ERROR("Couldn't open I2C Bus %d [openfd():open %d]", _i2cbus, errno);
		}
		if (ioctl(fd, I2C_SLAVE, _i2caddr) < 0) {
			ROS_ERROR("I2C slave %d failed [openfd():ioctl %d]", _i2caddr, errno);
		}
	}
}
