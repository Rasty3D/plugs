/**
 * @file	memblocks.cpp
 * @author	Jesús Ortiz
 * @version	2.0
 * @date	03-Sept-2012
 * @brief	Functions to manage data allocation
 */

/*
 * INCLUDE
 */

#include "memblocks.h"


/*
 * CLASS
 */

	/* Constructor and Destructor */

// Constructor
MemBlocks::MemBlocks()
{
	this->init();
}

// Destructor
MemBlocks::~MemBlocks()
{
	this->clear();
}


	/* Configuration functions */

// Set the alignment of the data
bool MemBlocks::setAlignment(unsigned int alignment)
{
	if (this->owner != MEMBLOCK_NO_OWNER)
		return false;

	// Check alignment, usually is a power of 2
	if ((alignment & (alignment - 1)) != 0)
	{
		// TODO: Display warning message
	}

	this->alignment = alignment;
	return true;
}


	/* Block management functions */

// Add a new block
bool MemBlocks::addBlock(unsigned long size, const char *name)
{
	if (this->owner != MEMBLOCK_NO_OWNER)
		return false;

	// Init new block
	MemBlock newBlock;

	if (this->blocks.size() == 0)
	{
		newBlock.begin = 0;
	}
	else
	{
		newBlock.begin =
			this->blocks.back().begin +
			this->blocks.back().realSize;
	}

	newBlock.size = size;

	if (this->alignment <= 1)
		newBlock.realSize = size;
	else if ((size % this->alignment) == 0)
		newBlock.realSize = size;
	else
		newBlock.realSize = size + this->alignment - (size % this->alignment);

	// Add the new block
	this->blocks.push_back(newBlock);

	// Add the new name to the trie
	if (!this->trie.add(name, &this->blocks.back(), false))
	{
		this->blocks.pop_back();
		return false;
	}

	return true;
}

// Get the memory block address
unsigned char *MemBlocks::getBlock(const char *name)
{
	if (this->owner == MEMBLOCK_NO_OWNER)
		return NULL;

	MemBlock *block = (MemBlock*)this->trie.get(name);

	if (block == NULL)
		return NULL;

	if ((block->begin + block->size) > this->bufferSize)
		return NULL;

	return &this->buffer[block->begin];
}

// Get the memory block size
unsigned long MemBlocks::getBlockSize(const char *name)
{
	if (this->owner == MEMBLOCK_NO_OWNER)
		return 0;

	MemBlock *block = (MemBlock*)this->trie.get(name);

	if (block == NULL)
		return 0;

	return block->size;
}

// Get the owner of the data
int MemBlocks::getOwner()
{
	return this->owner;
}

// Allocate the memory
bool MemBlocks::allocate(int owner, bool reset)
{
	if (this->owner != MEMBLOCK_NO_OWNER)
		this->clear();

	if (owner != MEMBLOCK_OWNER_HOST && owner < 0)
		return false;

	if (this->blocks.empty())
		return false;

	this->owner = owner;
	this->bufferSize =
		this->blocks.back().begin +
		this->blocks.back().realSize;

	this->buffer = (unsigned char*)malloc(this->bufferSize, owner);

	if (reset)
		memset(this->buffer, 0, this->bufferSize, owner);

	return true;
}

// Init the variables
void MemBlocks::init()
{
	this->blocks.clear();
	this->trie.clear();
	this->alignment = 0;
	this->owner = MEMBLOCK_NO_OWNER;
	this->bufferSize = 0;
	this->buffer = NULL;
	this->atlasSize = 0;
	this->atlas = NULL;
}

// Clear the memory and reset the variables
bool MemBlocks::clear()
{
	if (this->owner == MEMBLOCK_NO_OWNER)
		return false;

	free(this->buffer, owner);
	this->init();
	return true;
}


	/* Memory Atlas management functions */

// Gets the memory atlas. If it's not already generated, generates a new one
unsigned char *MemBlocks::getMemAtlas(unsigned long &size)
{
	if (!this->genMemAtlas())
	{
		size = 0;
		return NULL;
	}

	size = this->atlasSize;
	return this->atlas;
}

// Loads a memory atlas
bool MemBlocks::loadMemAtlas(unsigned char *buffer, unsigned long size)
{


	// TODO
	return false;
}

// Generates the memory atlas
bool MemBlocks::genMemAtlas()
{
	// Calculate maximum size


	// TODO
	return false;
}
