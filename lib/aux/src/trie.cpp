/**
 * @file	trie.cpp
 * @author	Jesús Ortiz
 * @version	1.0
 * @date	03-Sept-2012
 * @brief	Trie or Prefix Tree
 */

/*
 * INCLUDES
 */

#include "trie.h"


/*
 * CLASS
 */

	/* Constructor and destructor */

// Constructor
Trie::Trie()
{
	this->child     = NULL;
	this->next      = NULL;
	this->element   = NULL;
	this->character = '\0';
}

// Destructor
Trie::~Trie()
{
	this->clear();
}


	/* Manage elements */

// Add an element to the current position
bool Trie::add(const char *name, void *element, bool overwrite)
{
	// Check if it's a valid element
	if (element == NULL)
		return false;

	// Illegal character
	if (name[0] < 0)
		return false;

	// Last character
	if (name[1] == '\0')
	{
		// Check character
		if (this->character == '\0')
		{
			this->character = name[0];
			this->element = element;
			return true;
		}
		else if (this->character == name[0])
		{
			// The name already exists
			if (overwrite || this->element == NULL)
			{
				this->element = element;
				return true;
			}
			else
			{
				return false;
			}
		}
		else if (this->next != NULL)	// Check next brother
		{
			return this->next->add(name, element, overwrite);
		}
		else
		{
			// There is no brother with the name and it's the last character
			// -> Add new element
			this->next = new Trie;
			return this->next->add(name, element, overwrite);
		}
	}

	// Check character
	if (this->character == '\0')
	{
		// Save character
		this->character = name[0];

		if (this->child == NULL)
		{
			// Add the rest of the characters
			this->child = new Trie;
			return this->child->add(name + 1, element, overwrite);
		}
		else
		{
			// The child should be NULL
			return false;
		}
	}
	else if (this->character == name[0])
	{
		if (this->child == NULL)
		{
			// Add the rest of the characters
			this->child = new Trie;
		}

		return this->child->add(name + 1, element, overwrite);
	}
	else if (this->next != NULL)			// Check next brother
	{
		return this->next->add(name, element, overwrite);
	}
	else
	{
		// There is no brother with the name
		// -> Add the rest of the characters
		this->next = new Trie;
		return this->next->add(name, element, overwrite);
	}
}

// Get element from the current position
void *Trie::get(const char *name)
{
	// Illegal character
	if (name[0] < 0)
		return NULL;

	// Last character
	if (name[1] == '\0')
	{
		if (this->character == name[0])		// Check character
			return this->element;
		else if (this->next != NULL)		// Check next brother
			return this->next->get(name);
		else								// Element not found
			return NULL;
	}

	// Check character
	if (this->character == name[0])			// Check character
	{
		if (this->child == NULL)			// There are no more children
			return NULL;
		else
			return this->child->get(name + 1);	// Look for in the child
	}
	else if (this->next != NULL)			// Check next brother
	{
		return this->next->get(name);
	}
	else									// Element not found
	{
		return NULL;
	}
}

// Removes the reference to an element
bool Trie::remove(const char *name)
{
	// Illegal character
	if (name[0] < 0)
		return false;

	// Last character
	if (name[1] == '\0')
	{
		if (this->character == '\0')		// Check character, name not found
		{
			return false;
		}
		else if (this->character == name[0]) // Name found -> Delete reference
		{
			this->element = NULL;
			return true;
		}
		else if (this->next != NULL)		// Check next brother
		{
			return this->next->remove(name);
		}
		else // There is no brother with the name and it's the last character
		{
			return false;
		}
	}

	// Check character
	if (this->character == '\0')			// Name not found
	{
		return false;
	}
	else if (this->character == name[0])
	{
		if (this->child == NULL)			// Name not found
			return false;
		else
			return this->child->remove(name + 1);	// Remove from child
	}
	else if (this->next != NULL)			// Check next brother
	{
		return this->next->remove(name);
	}
	else									// There is no brother, not found
	{
		return false;
	}
}

// Clean tree
void Trie::clear()
{
	// Delete all children
	if (this->child != NULL)
	{
		Trie *aux = this->child;
		Trie *prevBrother;

		while (aux != NULL)
		{
			prevBrother = aux;
			aux = aux->next;
			delete prevBrother;
		}

		this->child = NULL;
	}
}

// Print tree
void Trie::print()
{
	std::cout << this->character;

	if (this->child != NULL)
		this->child->print();

	if (this->next != NULL)
		this->next->print();
}

// Print tree atlas
void Trie::printAtlas()
{
	unsigned long lastJump = 0;
	this->printAtlas(0, lastJump);
	std::cout << "-" << (lastJump + 1);
}

// Print trie atlas
void Trie::printAtlas(unsigned long depth, unsigned long &lastJump)
{
	std::cout << "'" << this->character << "', ";

	if (this->element != NULL)
		std::cout << "NULL, ";

	lastJump = depth;

	if (this->child != NULL)
		this->child->printAtlas(depth + 1, lastJump);

	if (this->next != NULL)
	{
		std::cout << "-" << (lastJump - depth + 1) << ", ";
		this->next->printAtlas(depth + 1, lastJump);
	}

	if (this->child == NULL && this->next == NULL)
		lastJump = depth;
}


	/* Other functions */

// Get the size of the tree (number of elements and leafs)
void Trie::getSize(unsigned long &nElements, unsigned long &nLeafs)
{
	if (this->element != NULL)
		nElements++;

	nLeafs++;

	if (this->child != NULL)
		this->child->getSize(nElements, nLeafs);

	if (this->next != NULL)
		this->next->getSize(nElements, nLeafs);
}

// Get atlas size
unsigned long Trie::getAtlasSize()
{
	return this->getAtlasSizeAux() + 1;
}

// Get atlas size
unsigned long Trie::getAtlasSizeAux()
{
	int size = 1;

	if (this->element != NULL)
		size++;

	if (this->child != NULL)
		size += this->child->getAtlasSizeAux();

	if (this->next != NULL)
		size += this->next->getAtlasSizeAux() + 1;

	return size;
}

// Get atlas
bool Trie::getAtlas(char *atlas)
{
	unsigned long pos = 0;
	unsigned long lastJump = 0;

	if (!this->getAtlas(atlas, pos, 0, lastJump))
		return false;

	if (lastJump > 254)
		return false;

	atlas[pos] = -(lastJump + 1);
	return true;
}

// Get atlas
bool Trie::getAtlas(
	char *atlas, unsigned long &pos,
	unsigned long depth, unsigned long &lastJump)
{
	atlas[pos++] = this->character;

	if (this->element != NULL)
		atlas[pos++] = '\0';

	lastJump = depth;

	if (this->child != NULL)
	{
		if (!this->child->getAtlas(atlas, pos, depth + 1, lastJump))
			return false;
	}

	if (this->next != NULL)
	{
		unsigned long jump = lastJump - depth + 1;

		if (jump >= 255)
			return false;

		atlas[pos++] = -jump;

		if (!this->next->getAtlas(atlas, pos, depth + 1, lastJump))
			return false;
	}

	if (this->child == NULL && this->next == NULL)
		lastJump = depth;

	return true;
}

// Set atlas
bool Trie::setAtlas(char *atlas, unsigned long atlasSize, void **elements)
{
	if (atlas[0] <= 0)
		return false;

	this->character = atlas[0];

	unsigned long atlasPos = 1;
	unsigned long elementPos = 0;
	return this->setAtlas(atlas, atlasSize, atlasPos, elements, elementPos);
}

// Set atlas
bool Trie::setAtlas(
	char *atlas, unsigned long atlasSize, unsigned long &atlasPos,
	void **elements, unsigned long &elementPos)
{
	if (atlasPos == atlasSize)
		return true;

	if (atlas[atlasPos] > 0)
	{
		this->child = new Trie;
		this->child->character = atlas[atlasPos++];
		if (!this->child->setAtlas(
				atlas, atlasSize, atlasPos, elements, elementPos))
			return false;
	}
	else if (atlas[atlasPos] == 0)
	{
		this->element = elements[elementPos++];
		atlasPos++;
	}
	else if (atlas[atlasPos] == -1)
	{
		atlasPos++;

		if (atlasPos == atlasSize)
			return true;

		this->next = new Trie;
		this->next->character = atlas[atlasPos++];
		if (!this->next->setAtlas(
				atlas, atlasSize, atlasPos, elements, elementPos))
			return false;
	}
	else // if (atlas[atlasPos] < -1)
	{
		atlas[atlasPos]++;
		return true;
	}

	return this->setAtlas(atlas, atlasSize, atlasPos, elements, elementPos);
}
