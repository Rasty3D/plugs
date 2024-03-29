/**
 * @file	cudainline.h
 * @author	Jesús Ortiz
 * @version	1.0
 * @date	17-Sept-2012
 * @brief	CUDA inline functions
 * @details	The functions included in this file check for CUDA errors
 * 			and launch an exception. The caller function (scheduler, main, etc)
 * 			should catch the exception.
 */

#ifndef CUDAINLINE_H_
#define CUDAINLINE_H_


/*
 * INCLUDES
 */

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


/*
 * MACROS
 */

/** Macro to launch a CUDA function and check the returning error */
#define cuda_safeCall(err)       __cuda_safeCall      (err, __FILE__, __LINE__)
/** Macro to check the last CUDA error */
#define cuda_checkLastError(msg) __cuda_checkLastError(msg, __FILE__, __LINE__)


/*
 * CUDA functions
 */

/** Function to launch a CUDA function and check the returning error
 *  @param[in] err		Error code
 *  @param[in] file		File producing the error
 *  @param[in] line		Line of the file producing the error
 */
inline void __cuda_safeCall(
	cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
    	printf("%s(%i) : cuda_safeCall() Runtime API error %d : %s.",
    		file, line, (int)err, cudaGetErrorString(err));
        throw (int)err;
    }
}

/** Function to check the last CUDA error
 *  @param[in] errorMessage		Error message to display in case of error
 *  @param[in] file				File producing the error
 *  @param[in] line				Line of the file producing the error
 */
inline void __cuda_checkLastError(
	const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
    	printf("%s(%i) : cuda_checkLastError() CUDA error : %s : %d : %s.",
    		file, line, errorMessage, (int)err, cudaGetErrorString(err));
        throw (int)err;
    }
}

#endif	/* CUDAINLINE_H_ */
