/**
 * @file	test_plugin.cpp
 * @author	Jesús Ortiz
 * @version	1.0
 * @date	19-Apr-2013
 * @brief	Test program for NUX plugin class
 */

/*
 * INCLUDES
 */

#include <fstream>
#include <sstream>

#include "nuxplugin.h"


/*
 * MAIN
 */

int main(int argc, char *argv[])
{
	NuxContext context;

	/*
	std::filebuf filebufErr;
	filebufErr.open("test.txt", std::ios::out);
	context.cerr = new std::ostream(&filebufErr);*/

	/*
	std::stringbuf bufErr;
	context.cerr = new std::ostream(&bufErr);*/

	NuxPlugin plugin(&context);

	if (!plugin.load("test"))
	{
		return -1;
	}

	return 0;
}
