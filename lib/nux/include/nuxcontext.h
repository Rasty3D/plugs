/**
 * @file	nuxconext.h
 * @author	Jesús Ortiz
 * @version	1.0
 * @date	19-Apr-2013
 * @brief	NUX context
 */

/*
 * INCLUDES
 */

#include <iostream>


/*
 * CLASSES
 */

/** NUX context class */
class NuxContext
{
		/** @name Standard iostream replacement */
	// @{
public:
	std::ostream *cout;		//!< Standard output replacement
	std::ostream *cerr;		//!< Standard error replacement
	std::istream *cin;		//!< Stantard input replacement
	// @}


public:
	/** Default constructor with standard io */
	NuxContext();
};
