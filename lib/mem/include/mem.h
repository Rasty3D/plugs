/**
 * @file	mem.h
 * @author	Jesús Ortiz
 * @version	1.0
 * @date	04-Sept-2012
 * @brief	Memory functions for CPU & CUDA
 */

#ifndef MEM_H_
#define MEM_H_

/*
 * INCLUDES
 */

#include <stdlib.h>
#include <string.h>

#ifdef CUDA
#include "cudaaux.h"
#include "cudamem.h"
#endif


/*
 * FUNCTIONS
 */

/** Allocate memory in device or host
 *  @param [in] size		Size in bytes to allocate
 *  @param [in] owner		Owner of the data
 *  @return 				Pointer to the allocated data
 */
void *malloc(size_t size, int owner);

/** Set the memory with a value
 *  @param [in] ptr			Pointer to the data
 *  @param [in] value		Value to fill the data
 *  @param [in] count		Number of bytes to fill
 *  @param [in] owner		Owner of the data
 */
void memset(void *ptr, int value, size_t count, int owner);

/** Free the memory in device or host
 *  @param [in] ptr			Pointer to the data
 *  @param [in] owner		Owner of the data
 */
void free(void *ptr, int owner);

/** Copy a memory block of data from and to host/device
 *  @param [in] dst			Pointer to the destination data
 *  @param [in] src			Pointer to the source data
 *  @param [in] count		Number of bytes to fill
 *  @param [in] ownerDst	Owner of the destination data
 *  @param [in] ownerSrc	Owner of the source data
 *  @return					Pointer to the destination data
 */
void *memcpy(
	void *dst, const void *src,
	size_t count,
	int ownerDst, int ownerSrc);

#endif	/* MEM_H_ */
