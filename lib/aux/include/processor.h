/**
 * @file	processor.h
 * @author	Jesús Ortiz
 * @version	1.0
 * @date	07-Sept-2012
 * @brief	Computation processor (CPU or GPU)
 */

#ifndef PROCESSOR_H_
#define PROCESSOR_H_

/*
 * INCLUDE
 */

#include <pthread.h>


/*
 * DEFINES
 */

	/** @name Processor states */
// @{

/** The processor is ready to accept a work */
#define PROCESSOR_STATE_IDLE		0

/** The processor is already processing something */
#define PROCESSOR_STATE_BUSY		1

/** The processor is running, but has to stop */
#define PROCESSOR_STATE_MUSTSTOP	2

// @}


/*
 * TYPES
 */

/** Function prototype for the processor */
typedef bool (*ProcessorFunc)(void*);

/** Data structure that will be passed fo the processor function */
typedef struct
{
	int *state;				//!< Pointer to the processor state
	void *arg;				//!< Argument (pointer to any data)
	ProcessorFunc func;		//!< Pointer to the function to run
}ProcessorArg;


/*
 * CLASS
 */

/** Class to manage a computation processor */
class Processor
{
private:
	/** Processor state */
	int state;

	/** Thread handler */
    pthread_t thread;

public:
    /** Constructor */
	Processor();

	/** Destructor */
	~Processor();

	/** Launch one function in a thread */
	int launchOnce(ProcessorFunc func, void *arg);

	/** Launch a function in a loop */
	int launchLoop(ProcessorFunc func, void *arg);

	/** Stop the running thread */
	void stop();

	/** Get the state of the processor */
	int getState();

private:
	/** Thread funtion for a single function */
	static void *runOnce(void *arg);

	/** Thread function for a loop function */
	static void *runLoop(void *arg);
};

#endif	/* PROCESSOR_H_ */
