/**
 * @file	cudaaux.cu
 * @author	Jesús Ortiz
 * @version	1.0
 * @date	18-Jan-2012
 * @brief	These are cuda auxiliary functions.
 * @details	This library contains cuda auxiliary functions
 * 			to display devices information, init and
 * 			finish a cuda device
 */

/*
 * INCLUDES
 */

	/* CUDA */
#include "cudaaux.h"
#include <cuda_runtime_api.h>
#include "cudainline.h"

#ifdef _MSC_VER
#include <windows.h>
#endif

/*
 * NAMESPACES
 */

using namespace std;


/*
 * CUDA functions
 */

// Function to print the devices information
void cuda_getDeviceInfo()
{
	int deviceCount;
	int device;
	struct cudaDeviceProp deviceProperties;

	cudaGetDeviceCount(&deviceCount);

	cout << "Device list [" << deviceCount << "]:" << endl;

	for (device = 0; device < deviceCount; device++)
	{
		cudaGetDeviceProperties(&deviceProperties, device);

		cout << "  " << (device + 1) << ".- " << deviceProperties.name << ":" << endl;
		cout << "     Total global memory    : " << deviceProperties.totalGlobalMem << endl;
		cout << "     Shared memory per block: " << deviceProperties.sharedMemPerBlock << endl;
		cout << "     Registers per block    : " << deviceProperties.regsPerBlock << endl;
		cout << "     Warp size              : " << deviceProperties.warpSize << endl;
		cout << "     Memory pitch           : " << deviceProperties.memPitch << endl;
		cout << "     Max threads per block  : " << deviceProperties.maxThreadsPerBlock << endl;
		cout << "     Max threads dimension  : " <<
			deviceProperties.maxThreadsDim[0] << ", " <<
			deviceProperties.maxThreadsDim[1] << ", " <<
			deviceProperties.maxThreadsDim[2] << endl;
		cout << "     Max grid size          : " <<
			deviceProperties.maxGridSize[0] << ", " <<
			deviceProperties.maxGridSize[1] << ", " <<
			deviceProperties.maxGridSize[2] << endl;
		cout << "     Major                  : " << deviceProperties.major << endl;
		cout << "     Minor                  : " << deviceProperties.minor << endl;
		cout << "     Clock rate             : " << deviceProperties.clockRate << endl;
		cout << "     Texture alignment      : " << deviceProperties.textureAlignment << endl;
		cout << "     Multi processor count  : " << deviceProperties.multiProcessorCount << endl;
		cout << "     Kernel execution time  : " << deviceProperties.kernelExecTimeoutEnabled << endl;
		cout << "     Integrated             : " << deviceProperties.integrated << endl;
		cout << "     Can map host memory    : " << deviceProperties.canMapHostMemory << endl;
		cout << "     Compute mode           : " << deviceProperties.computeMode << endl;
		cout << "     Concurrent kernels     : " << deviceProperties.concurrentKernels << endl;
		cout << "     ECC enabled            : " << deviceProperties.ECCEnabled << endl;
		cout << "     PCI bus ID             : " << deviceProperties.pciBusID << endl;
		cout << "     PCI device ID          : " << deviceProperties.pciDeviceID << endl;
		cout << "     TCC driver             : " << deviceProperties.tccDriver << endl;
	}


#ifdef _MSC_VER
	SYSTEM_INFO sysinfo;
	GetSystemInfo( &sysinfo );
	cout << "Number of CPU processors: " << sysinfo.dwNumberOfProcessors << endl;
#else
	cout << "Number of CPU processors: " << sysconf(_SC_NPROCESSORS_ONLN) << endl;
#endif
}

// Function to init a cuda device
int cuda_init(int device)
{
		/* Variables */

	int deviceCount;
	cudaEvent_t wakeGPU;
	struct cudaDeviceProp deviceProperties;


		/* Get number of devices */

	cudaGetDeviceCount(&deviceCount);

	if (device >= deviceCount || device < 0)
		return 0;


		/* Init device */

	cudaSetDevice(device);


		/* Send a wakeup event */


	cuda_safeCall(cudaEventCreate(&wakeGPU));


		/* Read the properties of the selected device */

	cudaGetDevice(&device);
	cudaGetDeviceProperties(&deviceProperties, device);

	cout << "Using device " << deviceProperties.name << endl;


		/* Return ok */

	return 1;
}

// Function to calculate the ingeter ceil of a division
void cuda_finish()
{
	cudaThreadSynchronize();
}

// Function to set a cuda device
int cuda_iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

// Function to set a cuda device
void cuda_setDevice(int device)
{
	cudaSetDevice(device);
}

// Gives the active cuda device
int cuda_getDevice()
{
	int device;
	cudaGetDevice(&device);
	return device;
}
