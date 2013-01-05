#include <iostream>
#include "bytebuffer.h"

int main(int argc, char *argv[])
{
	ByteBuffer byteBuffer;
	unsigned char *buffer;
	unsigned long size;
	unsigned char value;

	byteBuffer.write('H');
	byteBuffer.write('e');
	byteBuffer.write('l');
	byteBuffer.write('l');
	byteBuffer.write('o');
	byteBuffer.write((unsigned char*)", World!\0", 9);

	size = byteBuffer.getSize();
	buffer = byteBuffer.getBuffer();

	std::cout << "Size  : " << size << std::endl;
	std::cout << "Buffer: " << (char*)buffer << std::endl;

	byteBuffer.clear();
	byteBuffer.write((unsigned char*)"Blah, blah, blah\0", 17);

	byteBuffer.setCursor(0);
	while (byteBuffer.read(value))
	{
		std::cout << value;
	}
	std::cout << std::endl;

	return 0;
}
