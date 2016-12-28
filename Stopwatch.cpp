

#include "Stopwatch.h"

#ifdef _MSC_VER
#include <windows.h>
#include <stdint.h>
int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
	// Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
	static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

	SYSTEMTIME  system_time;
	FILETIME    file_time;
	uint64_t    time;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	time = ((uint64_t)file_time.dwLowDateTime);
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec = (long)((time - EPOCH) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
	return 0;
}
#else
#include <sys/time.h>
#endif

double Stopwatch::_stopwatch() const
{
	struct timeval time;
	gettimeofday(&time, 0);
	return 1.0 * time.tv_sec + time.tv_usec / (double)1e6;
}
