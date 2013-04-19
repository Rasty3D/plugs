/**
 * @file	nuxplugin.h
 * @author	Jesús Ortiz
 * @version	1.0
 * @date	19-Apr-2013
 * @brief	Plugin for NUX
 */

/*
 * INCLUDES
 */

	/* General */
#include <iostream>
#include <string>
#include <dlfcn.h>

	/* NUX */
#include "nuxcontext.h"


/*
 * CLASSES
 */

/** NuxPlugin class for NUX */
class NuxPlugin
{
private:
	NuxContext *ctxt;
	void *handle;
	std::string name;

public:
	NuxPlugin(NuxContext *context);
	~NuxPlugin();

	bool load(const char *filename);
	bool unload();

private:
	bool init();
	bool finish();

public:
	bool getMemAtlas();

	std::string getName();
	bool execute();
};
