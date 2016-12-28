#ifndef Stopwatch_h
#define Stopwatch_h

class Stopwatch {
private:
	double start;
	double _stopwatch() const;
public:
	Stopwatch() { reset(); }
	void reset() { start = _stopwatch(); }
	double read() const { return _stopwatch() - start; }
};

#endif
