#include <iostream>
#include <string.h>
#include "trie.h"

int main(int argc, char *argv[])
{
	std::cout << "==== START ====" << std::endl;

	// Trie
	Trie trie;


	std::cout << "==== ADD ELEMENTS ... ";

	// Add elements
	const int nValues = 8;
	/*
	int values[nValues] = {
		   7,     3,     4,    12,  15,  11,    5,     9};
	const char *names[nValues] = {
		"to", "tea", "ted", "ten", "a", "i", "in", "inn"};*/
	int values[nValues] = {
		   7,     3,  15,     9,     4,    12,  11,    5};
	const char *names[nValues] = {
		"to", "tea", "a", "inn", "ted", "ten", "i", "in"};

	for (int i = 0; i < nValues; i++)
	{
		if (!trie.add(names[i], (void*)&values[i], true))
		{
			std::cout << "[ERROR] Error adding [" << names[i] << "]" << std::endl;
			return -1;
		}
	}

	std::cout << "[OK]" << std::endl;


	std::cout << "==== GET ELEMENTS ... ";

	int *element;

	for (int i = 0; i < nValues; i++)
	{
		// Get elements
		element = (int*)trie.get(names[i]);

		if (element == NULL)
		{
			std::cout << "[ERROR] Error getting element" << std::endl;
			return -1;
		}

		if (*element != values[i])
		{
			std::cout << "[ERROR] Wrong element value" << std::endl;
			return -1;
		}
	}

	std::cout << "[OK]" << std::endl;


	std::cout << "==== PRINT TRIE ====" << std::endl;

	// Print trie
	std::cout << "Trie: ";
	trie.print();
	std::cout << std::endl;

	// Print trie atlas
	std::cout << "Atlas: ";
	trie.printAtlas();
	std::cout << std::endl;


	std::cout << "==== GET TRIE ATLAS ... ";

	// Get atlas trie size
	unsigned long atlasSize;
	unsigned long nElements;
	trie.getAtlasSize(atlasSize, nElements);

	if (atlasSize != 24)
	{
		std::cout << "[ERROR] Returned wrong atlas size" << std::endl;
		return -1;
	}

	if (nElements != nValues)
	{
		std::cout << "[ERROR] Returned wrong number of elements" << std::endl;
		return -1;
	}

	// Get atlas
	char *atlas = new char[atlasSize];
	void **elements = new void*[nElements];

	if (!trie.getAtlas(atlas, elements))
	{
		std::cout << "[ERROR] Error getting the atlas" << std::endl;
		return -1;
	}

	char atlasCheck[24] = {
		't', 'o', '\0', -1, 'e', 'a', '\0', -1, 'd', '\0', -1, 'n', '\0', -6,
		'a', '\0', -1, 'i', '\0', 'n', '\0', 'n', '\0', -5};

	if (memcmp(atlas, atlasCheck, 24) != 0)
	{
		std::cout << "[ERROR] The atlas is wrong" << std::endl;
		return -1;
	}

	std::cout << "[OK]" << std::endl;


	std::cout << "==== SET TRIE ATLAS ... ";

	// Clean and set atlas
	trie.clear();

	if (!trie.setAtlas(atlas, atlasSize, elements))
	{
		std::cout << "[ERROR] Error setting the atlas" << std::endl;
		return -1;
	}

	// Get elements
	for (int i = 0; i < nValues; i++)
	{
		// Get elements
		element = (int*)trie.get(names[i]);

		if (element == NULL)
		{
			std::cout << "[ERROR] Error getting element" << std::endl;
			return -1;
		}

		if (*element != values[i])
		{
			std::cout << "[ERROR] Wrong element value" << std::endl;
			return -1;
		}
	}

	// Get atlas again and check
	trie.getAtlasSize(atlasSize, nElements);

	if (atlasSize != 24)
	{
		std::cout << "[ERROR] Returned wrong atlas size" << std::endl;
		return -1;
	}
	if (!trie.getAtlas(atlas, elements))
	{
		std::cout << "[ERROR] Error getting the atlas" << std::endl;
		return -1;
	}
	if (memcmp(atlas, atlasCheck, 24) != 0)
	{
		std::cout << "[ERROR] The atlas is wrong" << std::endl;
		return -1;
	}

	std::cout << "[OK]" << std::endl;


	std::cout << "==== FINISH ====" << std::endl;

	// Return ok
	return 0;
}
