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
	if (this->cursor >= this->size)
	{
		if (!this->resize(this->capacity + this->step))
			return false;
	}

	this->buffer[this->cursor] = value;
	this->cursor++;
	return true;
}

bool ByteBuffer::write(unsigned char *values, unsigned long size)
{
	if (this->cursor >= this->size + size)
	{
		if (!this->resize(this->capacity + size + this->step))
			return false;
	}

	memcpy(&this->buffer[this->cursor], values, size);
	this->cursor += size;
	return true;
}

bool ByteBuffer::read(unsigned char &value)
{
	if (this->cursor == this->size)
		return false;

	value = this->buffer[this->cursor];
	this->cursor++;
	return true;
}

bool ByteBuffer::read(unsigned char *values, unsigned long size)
{
	if (this->cursor >= this->size - size)
		return false;

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
		if (!this->resize(pos + this->step))
			return false;

		this->cursor = pos;
		this->size = pos;
	}

	return true;
}

bool ByteBuffer::setEnd()
{
	this->size = this->cursor;
}

void ByteBuffer::clear()
{
	this->size = 0;
	this->cursor = 0;

	if (this->capacity > 0)
		memmemset(this->buffer, 0, this->capacity);
}

bool ByteBuffer::resize(unsigned long size)
{

}
