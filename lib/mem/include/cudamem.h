/**
 * @file	cudamem.h
 * @author	Jesús Ortiz
 * @version	2.0
 * @date	04-Sept-2012
 * @brief	Memory functions for CUDA
 */

#ifndef CUDAMEM_H_
#define CUDAMEM_H_

/*
 * FUNCTIONS
 */

/** Allocate memory in device or host
 *  @param [in] size		Size in bytes to allocate
 *  @param [in] owner		Owner of the data
 *  @return 				Pointer to the allocated data
 */
void *cudaMalloc(size_t size, int owner);

/** Set the memory with a value
 *  @param [in] ptr			Pointer to the data
 *  @param [in] value		Value to fill the data
 *  @param [in] count		Number of bytes to fill
 *  @param [in] owner		Owner of the data
 */
void cudaMemset(void *ptr, int value, size_t count, int owner);

/** Free the memory in device or host
 *  @param [in] ptr			Pointer to the data
 *  @param [in] owner		Owner of the data
 */
void cudaFree(void *ptr, int owner);

/** Copy a memory block of data from and to host/device
 *  @param [in] dst			Pointer to the destination data
 *  @param [in] src			Pointer to the source data
 *  @param [in] count		Number of bytes to fill
 *  @param [in] ownerDst	Owner of the destination data
 *  @param [in] ownerSrc	Owner of the source data
 *  @return					Pointer to the destination data
 */
void *cudaMemcpy(
	void *dst, const void *src,
	size_t count,
	int ownerDst, int ownerSrc);

#endif	/* CUDAMEM_H_ */
