/**
 * @file	nuxconext.cpp
 * @author	Jesús Ortiz
 * @version	1.0
 * @date	19-Apr-2013
 * @brief	NUX context
 */

/*
 * INCLUDES
 */

#include "nuxcontext.h"


/*
 * CLASSES
 */

	/* NUX context class */

// Default constructor with standard io
NuxContext::NuxContext()
{
	cout = &std::cout;
	cerr = &std::cerr;
	cin = &std::cin;
}
