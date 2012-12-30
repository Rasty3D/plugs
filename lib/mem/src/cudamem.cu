/**
 * @file	cudamem.cu
 * @author	Jesús Ortiz
 * @version	2.0
 * @date	04-Sept-2012
 * @brief	Memory functions for CUDA
 */

/*
 * INCLUDES
 */

#include "cudamem.h"
#include "cudainline.h"


/*
 * FUNCTIONS
 */

// Allocate memory in device or host
void *cudaMalloc(size_t size, int owner)
{
	void *ptr;
	cudaSetDevice(owner);
	cuda_safeCall(cudaMalloc((void**)&ptr, size));
	return ptr;
}

// Set the memory with a value
void cudaMemset(void *ptr, int value, size_t count, int owner)
{
	cuda_safeCall(cudaMemset(ptr, value, count));
}

// Free the memory in device or host
void cudaFree(void *ptr, int owner)
{
	cuda_safeCall(cudaFree(ptr));
}

// Copy a memory block of data from and to host/device
void *cudaMemcpy(
	void *dst, const void *src,
	size_t count,
	int ownerDst, int ownerSrc)
{
	cudaMemcpyKind kind;

	if (ownerDst <= 0 && ownerSrc > 0)
		kind = cudaMemcpyDeviceToHost;
	else if (ownerDst > 0 && ownerSrc <= 0)
		kind = cudaMemcpyHostToDevice;
	else if (ownerDst > 0 && ownerSrc > 0)
		kind = cudaMemcpyDeviceToDevice;
	else //if (ownerDst <= 0 && ownerSrc <= 0)
		kind = cudaMemcpyHostToHost;

	cuda_safeCall(cudaMemcpy(dst, src, count, kind));

	return dst;
}
