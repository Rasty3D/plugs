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
	byteBuffer.write((unsigned char*)", World!", 8);

	byteBuffer.setCursor(320);
	byteBuffer.write('*');

	size = byteBuffer.getSize();
	buffer = byteBuffer.getBuffer();
	buffer[size] = '\0';

	std::cout << "Size  : " << size << std::endl;
	std::cout << "Buffer: " << (char*)buffer << std::endl;

	byteBuffer.setCursor(0);
	while (byteBuffer.read(value))
	{
		std::cout << value;
	}
	std::cout << std::endl;

	return 0;
}
