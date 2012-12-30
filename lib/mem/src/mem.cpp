/**
 * @file	mem.cpp
 * @author	Jesús Ortiz
 * @version	1.0
 * @date	04-Sept-2012
 * @brief	Memory functions for CPU & CUDA
 */

/*
 * INCLUDES
 */

#include "mem.h"


/*
 * FUNCTIONS
 */

// Allocate memory in device or host
void *malloc(size_t size, int owner)
{
	void *ptr;

#ifdef CUDA
	if (owner < 0)
#endif
		ptr = malloc(size);
#ifdef CUDA
	else
		ptr = cudaMalloc(size, owner);
#endif

	return ptr;
}

// Set the memory with a value
void memset(void *ptr, int value, size_t count, int owner)
{
#ifdef CUDA
	if (owner < 0)
#endif
		memset(ptr, value, count);
#ifdef CUDA
	else
		cudaMemset(ptr, value, count, owner);
#endif
}

// Free the memory in device or host
void free(void *ptr, int owner)
{
#ifdef CUDA
	if (owner < 0)
#endif
		free(ptr);
#ifdef CUDA
	else
		cudaFree(ptr, owner);
#endif
}

// Copy a memory block of data from and to host/device
void *memcpy(
	void *dst, const void *src,
	size_t count,
	int ownerDst, int ownerSrc)
{
#ifdef CUDA
	if (ownerDst < 0 && ownerSrc < 0)
#endif
		return memcpy(dst, src, count);
#ifdef CUDA
	else
		return cudaMemcpy(dst, src, count, ownerDst, ownerSrc);
#endif
}
