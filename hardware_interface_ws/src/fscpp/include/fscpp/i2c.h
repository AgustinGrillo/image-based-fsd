#ifndef I2C_H
#define I2C_H
#include <inttypes.h>
#include "ros/ros.h"
#include <sys/ioctl.h>
#include <errno.h>
#include <stdio.h>      /* Standard I/O functions */
#include <fcntl.h>
#include <unistd.h>
//#include <linux/i2c.h>
#include <linux/i2c-dev.h>
#include <syslog.h>		/* Syslog functionality */

#define BUFFER_SIZE 1

namespace fscpp
{
	class I2C
	{
		public:
			I2C(int, int);
			virtual ~I2C();
			uint8_t dataBuffer[BUFFER_SIZE];
			uint8_t readBytes(uint8_t registerNumber, uint8_t bufferSize, uint32_t &position);
			uint8_t writeData(uint8_t registerNumber, uint8_t data[3]);
		private:
			int _i2caddr;
			int _i2cbus;
			void openfd();
			char busfile[64];
			int fd;
	};
}

#endif
