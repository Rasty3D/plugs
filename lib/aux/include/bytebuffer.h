#include <stdlib.h>
#include <string.h>

#define BYTEBUFFER_STEP	256

#define BYTEBUFFER_BEG	-1
#define BYTEBUFFER_CUR	 0
#define BYTEBUFFER_END	 1

class ByteBuffer
{
private:
	unsigned long size;
	unsigned long capacity;
	unsigned long step;
	unsigned char *buffer;
	unsigned long cursor;

public:
	ByteBuffer();
	~ByteBuffer();

	unsigned long getSize();
	unsigned char *getBuffer();
	bool setCapacity(unsigned long capacity);
	bool setCapacityStep(unsigned long step);

	bool write(unsigned char value);
	bool write(unsigned char *values, unsigned long size);

	bool read(unsigned char &value);
	bool read(unsigned char *values, unsigned long size);

	unsigned long getCursor();
	bool setCursor(unsigned long pos, int direction = BYTEBUFFER_BEG);
	bool setEnd();
	void clear();

private:
	bool resize(unsigned long size);
};
