/**
 * @file	processor.cpp
 * @author	Jesús Ortiz
 * @version	1.0
 * @date	07-Sept-2012
 * @brief	Computation processor (CPU or GPU)
 */

/*
 * INCLUDE
 */

#include "processor.h"


/*
 * CLASS
 */

	/* Class to manage a computation processor */

// Constructor
Processor::Processor()
{
	this->state = PROCESSOR_STATE_IDLE;
}

// Destructor
Processor::~Processor()
{
	this->stop();
}

// Launch one function in a thread
int Processor::launchOnce(ProcessorFunc func, void *arg)
{
	if (this->state != PROCESSOR_STATE_IDLE)
		return 0;

	ProcessorArg threadArg = {&this->state, arg, func};
	this->state = PROCESSOR_STATE_BUSY;
	pthread_create(&this->thread, NULL, Processor::runOnce, (void*)&threadArg);

	return 1;
}

// Launch a function in a loop
int Processor::launchLoop(ProcessorFunc func, void *arg)
{
	if (this->state != PROCESSOR_STATE_IDLE)
		return 0;

	ProcessorArg threadArg = {&this->state, arg, func};
	this->state = PROCESSOR_STATE_BUSY;
	pthread_create(&this->thread, NULL, Processor::runLoop, (void*)&threadArg);

	return 1;
}

// Stop the running thread
void Processor::stop()
{
	if (this->state == PROCESSOR_STATE_BUSY)
	{
		this->state = PROCESSOR_STATE_MUSTSTOP;
		pthread_join(this->thread, NULL);
	}
}

// Get the state of the processor
int Processor::getState()
{
	return this->state;
}

// Thread funtion for a single function
void *Processor::runOnce(void *arg)
{
	// Get argument
	ProcessorArg *threadArg = (ProcessorArg*)arg;

	// Call function
	threadArg->func(threadArg->arg);

	// Set idle state and exit
	threadArg->state = PROCESSOR_STATE_IDLE;
	return NULL;
}

// Thread function for a loop function
void *Processor::runLoop(void *arg)
{
	// Get argument
	ProcessorArg *threadArg = (ProcessorArg*)arg;

	// Call function in a loop
	while (*threadArg->state == PROCESSOR_STATE_BUSY)
	{
		if (!threadArg->func(threadArg->arg))
			break;
	}

	// Set idle state and exit
	threadArg->state = PROCESSOR_STATE_IDLE;
	return NULL;
}
