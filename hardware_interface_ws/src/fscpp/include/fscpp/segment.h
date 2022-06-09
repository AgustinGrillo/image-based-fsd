#ifndef FSCPP__SEGMENT_H
#define FSCPP__SEGMENT_H

#include <fscpp/joint.h>

namespace fscpp
{
	template <int T> class Segment
	{
		private:

		public:
			Segment() { };
			~Segment() { };
			int size() const { return T; }
			Joint joints[T];
	};
}

#endif
