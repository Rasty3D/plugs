#include <iostream>
#include "trie.h"

int main(int argc, char *argv[])
{
	// Trie
	Trie trie;

	// Add elements
	const int nValues = 8;
	int values[nValues] = {
		   7,     3,     4,    12,  15,  11,    5,     9};
	const char *names[nValues] = {
		"to", "tea", "ted", "ten", "a", "i", "in", "inn"};

	for (int i = 0; i < nValues; i++)
	{
		if (!trie.add(names[i], (void*)&values[i], true))
		{
			std::cerr << "Error adding [" << names[i] << "]" << std::endl;
			return -1;
		}
	}

	// Get element
	int *element = (int*)trie.get("ted");

	if (element == NULL)
	{
		std::cerr << "Error getting element" << std::endl;
		return -1;
	}

	std::cout << "Value of element 'ted': " << (int)*element << std::endl;

	if (*element != 4)
	{
		std::cerr << "Wrong element value" << std::endl;
		return -1;
	}

	// Print trie
	std::cout << "Trie: ";
	trie.print();
	std::cout << std::endl;

	// Print trie atlas
	std::cout << "Atlas: ";
	trie.printAtlas();
	std::cout << std::endl;

	// Get trie size
	unsigned long nElements = 0;
	unsigned long nLeafs = 0;
	trie.getSize(nElements, nLeafs);
	std::cout << "Trie size: " << nElements << ", " << nLeafs << std::endl;

	if (nElements != 8)
	{
		std::cerr << "Returned wrong number of elements" << std::endl;
		return -1;
	}

	if (nLeafs != 10)
	{
		std::cerr << "Returned wrong number of leafs" << std::endl;
		return -1;
	}

	// Get atlas trie size
	unsigned long atlasSize = trie.getAtlasSize();
	std::cout << "Trie Atlas size: " << atlasSize << std::endl;

	if (atlasSize != 24)
	{
		std::cerr << "Returned wrong atlas size" << std::endl;
		return -1;
	}

	// Get atlas
	char *atlas = new char[atlasSize];
	if (!trie.getAtlas(atlas))
	{
		std::cerr << "Error getting the atlas" << std::endl;
		return -1;
	}

	std::cout << "Atlas: ";
	for (unsigned long i = 0; i < atlasSize; i++)
	{
		if (atlas[i] > 16)
			std::cout << "'" << atlas[i] << "'";
		else if (atlas[i] == '\0')
			std::cout << "NULL";
		else if (atlas[i] < 0)
			std::cout << (int)atlas[i];
		else
			std::cout << "ERROR";

		if (i == atlasSize - 1)
			std::cout << std::endl;
		else
			std::cout << ", ";
	}

	// Clean and set atlas
	trie.clear();
	void *elements[nValues];
	for (int i = 0; i < nValues; i++)
		elements[i] = (void*)&values[i];
	if (!trie.setAtlas(atlas, atlasSize, elements))
	{
		std::cerr << "Error setting the atlas" << std::endl;
		return -1;
	}

	// Print trie atlas
	std::cout << "Atlas: ";
	trie.printAtlas();
	std::cout << std::endl;

	// Return ok
	return 0;
}
