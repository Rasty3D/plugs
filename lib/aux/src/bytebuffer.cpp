#include "bytebuffer.h"

ByteBuffer::ByteBuffer()
{
	this->size = 0;
	this->capacity = 0;
	this->step = BYTEBUFFER_STEP;
	this->buffer = NULL;
	this->cursor = 0;
}

ByteBuffer::~ByteBuffer()
{
	if (this->capacity > 0)
		delete [] this->buffer;
}

unsigned long ByteBuffer::getSize()
{
	return this->size;
}

unsigned char *ByteBuffer::getBuffer()
{
	return this->buffer;
}

bool ByteBuffer::setCapacity(unsigned long capacity)
{
	return this->resize(capacity);
}

bool ByteBuffer::setCapacityStep(unsigned long step)
{
	if (step < 1)
		return false;

	this->step = step;
	return true;
}

bool ByteBuffer::write(unsigned char value)
{
	// Resize buffer if needed
	if (!this->resize(this->cursor + 1))
		return false;

	// Write data
	this->buffer[this->cursor] = value;
	this->cursor++;

	// Change size if the cursor went beyond
	if (this->size < this->cursor)
		this->size = this->cursor;

	return true;
}

bool ByteBuffer::write(unsigned char *values, unsigned long size)
{
	// Resize buffer if needed
	if (!this->resize(this->cursor + size))
		return false;

	// Write data
	memcpy(&this->buffer[this->cursor], values, size);
	this->cursor += size;

	// Change size if the cursor went beyond
	if (this->size < this->cursor)
		this->size = this->cursor;

	return true;
}

bool ByteBuffer::read(unsigned char &value)
{
	// I cannot read beyond the size
	if (this->cursor >= this->size)
		return false;

	// Read data
	value = this->buffer[this->cursor];
	this->cursor++;

	return true;
}

bool ByteBuffer::read(unsigned char *values, unsigned long size)
{
	// I cannot read beyond the size
	if (this->cursor >= this->size - size)
		return false;

	// Read data
	memcpy(values, &this->buffer[this->cursor], size);
	this->cursor += size;

	return true;
}

unsigned long ByteBuffer::getCursor()
{
	return this->cursor;
}

bool ByteBuffer::setCursor(unsigned long pos, int direction)
{
	// Convert from relative to absolute position
	switch (direction)
	{
	case BYTEBUFFER_BEG:
		// Do nothing
		break;
	case BYTEBUFFER_CUR:
		pos += this->cursor;
		break;
	case BYTEBUFFER_END:
		if (pos > this->size)
			return false;
		pos = this->size - pos;
		break;
	default:
		return false;
	}

	// Check capacity
	if (pos < this->capacity)
	{
		this->cursor = pos;

		if (this->size < pos)
			this->size = pos;
	}
	else	// And resize if pos is bigger than capacity
	{
		if (!this->resize(pos))
			return false;

		this->cursor = pos;
		this->size = pos;
	}

	return true;
}

bool ByteBuffer::setEnd()
{
	this->size = this->cursor;
	return true;
}

void ByteBuffer::reset()
{
	this->size = 0;
	this->cursor = 0;

	if (this->capacity > 0)
		memset(this->buffer, 0, this->capacity);
}

bool ByteBuffer::resize(unsigned long size)
{
	unsigned long newCapacity;

	// Calculate new capacity
	if ((size % this->step) == 0)
		newCapacity = (size / this->step) * this->step;
	else
		newCapacity = (size / this->step + 1) * this->step;

	if (this->capacity == 0)
	{
		this->capacity = newCapacity;
		this->buffer = new unsigned char[this->capacity];
	}
	else
	{
		unsigned char *newBuffer = new unsigned char[newCapacity];
		memcpy(newBuffer, this->buffer, this->capacity);
		delete [] this->buffer;
		this->buffer = newBuffer;
		this->capacity = newCapacity;
	}

	return true;
}
