/**
 * @file	nuxplugin.cpp
 * @author	Jesús Ortiz
 * @version	1.0
 * @date	19-Apr-2013
 * @brief	Plugin for NUX
 */

/*
 * INCLUDES
 */

#include "nuxplugin.h"


/*
 * CLASSES
 */

NuxPlugin::NuxPlugin(NuxContext *context)
{
	this->ctxt = context;
	this->handle = NULL;
	this->name = "unknown";
}

NuxPlugin::~NuxPlugin()
{
}

bool NuxPlugin::load(const char *filename)
{
	this->handle = dlopen(filename, RTLD_NOW | RTLD_LOCAL);

	if (this->handle == NULL)
	{
		*ctxt->cerr << "Error opening shader [" << dlerror() << "]\n";
		return false;
	}

	/*
	*(void**)(&this->pGetType) = dlsym(this->handle, "getType");

	if ((error = dlerror()) != NULL)
	{
		cout << "Error loading 'getType' function [" << error << "]" << endl;
		this->unload();
		return false;
	}*/

	return true;
}

bool NuxPlugin::unload()
{
	// TODO
	return false;
}

bool NuxPlugin::init()
{
	// TODO
	return false;
}

bool NuxPlugin::finish()
{
	// TODO
	return false;
}

bool NuxPlugin::getMemAtlas()
{
	// TODO
	return false;
}

std::string NuxPlugin::getName()
{
	return this->name;
}

bool NuxPlugin::execute()
{
	// TODO
	return false;
}
