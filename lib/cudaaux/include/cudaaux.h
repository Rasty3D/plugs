/**
 * @file	cudaaux.h
 * @author	Jesús Ortiz
 * @version	1.0
 * @date	18-Jan-2012
 * @brief	These are cuda auxiliary functions.
 * @details	This library contains cuda auxiliary functions
 * 			to display devices information, init and
 * 			finish a cuda device
 */

#ifndef CUDA_H_
#define CUDA_H_


/*
 * INCLUDES
 */

	/* General */
#include <iostream>


/*
 * CUDA functions
 */

/** Function to print the devices information. This function prints the
 *  detailed information of the cuda enabled devices present in the system and
 *  also the number of cores of the CPU(s)
 */
void cuda_getDeviceInfo();

/** Function to init a cuda device
 *  @param [in] device 	The device id starting in 1
 */
int cuda_init(int device);

/** Function to finish the current cuda device */
void cuda_finish();

/** Function to calculate the ingeter ceil of a division. This function is
 *  used commonly to calculate the minimum number of blocks you need for an
 *  specified number of thread to calculate the desired number of elements
 *  @param [in] a		Divident (typically number of elements)
 *  @param [in] b		Divisor (typically number of threads)
 *  @return				Integer ceil of the quotient (typically number of
 * 						blocks)
 */
int cuda_iDivUp(int a, int b);

/** Function to set a cuda device
 *  @param [in] device 	The device id starting in 1
 */
void cuda_setDevice(int device);

/** Gives the active cuda device
 *  @return				Device id, starting with 1
 */
int cuda_getDevice();

#endif /* CUDA_H_ */
