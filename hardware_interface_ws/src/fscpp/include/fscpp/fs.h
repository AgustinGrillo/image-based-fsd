#ifndef FSCPP__FS_H
#define FSCPP__FS_H

#include <sstream>
#include <fscpp/segment.h>
#include <stdexcept>
#include <fscpp/joint.h>

namespace fscpp
{
	class FS
	{
		private:
		public:
			FS();
			~FS();

			Segment<6> base;   //The number 5 is the amount joints in that segment

			Joint getJoint(std::string jointName);
			void setJoint(fscpp::Joint joint);
	};
}

#endif
