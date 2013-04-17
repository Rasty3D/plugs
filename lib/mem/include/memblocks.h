/**
 * @file	memblocks.h
 * @author	Jesús Ortiz
 * @version	2.0
 * @date	03-Sept-2012
 * @brief	Functions to manage data allocation
 */

#ifndef MEMBLOCKS_H_
#define MEMBLOCKS_H_

/*
 * INCLUDE
 */

#include <list>

#include "trie.h"
#include "mem.h"


/*
 * DEFINES
 */

#define MEMBLOCK_NO_OWNER	-2
#define MEMBLOCK_OWNER_HOST	-1


/*
 * TYPES
 */

/** Memory block */
typedef struct
{
	unsigned long begin;	//!< Offset of the beginning of the data
	unsigned long size;		//!< Size of the data (bytes)
	unsigned long realSize;	//!< Size occupied in the memory (bytes)
}MemBlock;


/*
 * CLASS
 */

/** Class to manage the data allocation */
class MemBlocks
{
		/** @name Memory blocks */
private:
	// @{
	std::list<MemBlock> blocks;	//!< List of blocks
	Trie trie;					//!< Trie structure to manage the block names
	unsigned int alignment;		//!< Alignment
	int owner;					//!< Owner of the data (-2 no owner, -1 if host, >=0 device)
	unsigned long bufferSize;	//!< Total size in bytes of the buffer
	unsigned char *buffer;		//!< Buffer to store the data
	// @}


		/** @name Memory atlas */
private:
	// @{
	unsigned long atlasSize;	//!< Size of the atlas
	unsigned char *atlas;		//!< Buffer to store the atlas
	// @}


		/** @name Constructor and Destructor */
public:
	// @{

	/** Constructor */
	MemBlocks();

	/** Destructor */
	~MemBlocks();

	// @}


		/** @name Configuration functions */
public:
	// @{

	/** Set the alignment of the data.
	 *  The memory blocks will be align leaving
	 *  a small space between them.
	 *  @param [in]	alignment	Alignment of the data in bytes
	 *  @return					true if ok, false otherwise
	 */
	bool setAlignment(unsigned int alignment);

	// @}


		/** @name Block management functions */
public:
	// @{

	/** Add a new block
	 *  @param [in]	size	Size in bytes of the block
	 *  @param [in] name	Name of the block
	 *  @return				true if ok, false otherwise
	 */
	bool addBlock(unsigned long size, const char *name);

	/** Get the memory block address
	 *  @param [in] name	Name of the block
	 *  @return				Pointer to the memory block.
	 *  					NULL if the name is not found
	 */
	unsigned char *getBlock(const char *name);

	/** Get the memory block size
	 *  @param [in] name	Name of the block
	 *  @return				Size of the memory block.
	 *  					0 if the name is not found
	 */
	unsigned long getBlockSize(const char *name);

	/** Get the owner of the data
	 *  @return				-2 if the data is not initialized
	 */
	int getOwner();

	/** Allocate the memory
	 *  @param [in] owner	Id of the owner
	 *  @param [in] reset	If true then the memory is reset to zeros
	 *  @return				true if ok, false otherwise
	 */
	bool allocate(int owner, bool reset = false);

private:
	/** Init the variables */
	void init();

public:
	/** Clear the memory and reset the variables
	 *  @return				true if ok, false otherwise
	 */
	bool clear();

	// @}


		/** @name Memory Atlas management functions */
public:
	// @{

	/** Gets the memory atlas size
	 *  @return		Memory atlas size
	 */
	unsigned long getMemAtlasSize();

	/** Gets the memory atlas
	 *  @param [out] atlas		Buffer to store the atlas
	 *  @return					true if the size is enough,
	 *  						false otherwise
	 */
	bool getMemAtlas(unsigned char *atlas);

	/** Loads a memory atlas
	 *  @param [in]	atlas	Buffer containing the memory atlas
	 *  @param [in] size	Size of the memory atlas buffer
	 *  @return				true if ok, false otherwise
	 */
	bool loadMemAtlas(unsigned char *atlas, unsigned long size);

	/** Generates the memory atlas
	 *  @return				true if ok, false otherwise
	 */
	bool genMemAtlas();

	// @}
};

#endif	/* MEMBLOCKS_H_ */
