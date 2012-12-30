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
	else if (this->next != NULL)	// Check next brother
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
	// Last character
	if (name[1] == '\0')
	{
		if (this->character == name[0])		// Check character
			return this->element;
		else if (this->next != NULL)	// Check next brother
			return this->next->get(name);
		else								// Element not found
			return NULL;
	}

	// Check character
	if (this->character == name[0])			// Check character
	{
		if (this->child == NULL)				// There are no more children
			return NULL;
		else
			return this->child->get(name + 1);	// Look for in the child
	}
	else if (this->next != NULL)		// Check next brother
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
	// Last character
	if (name[1] == '\0')
	{
		if (this->character == '\0')	// Check character, name not found
		{
			return false;
		}
		else if (this->character == name[0])  // Name found -> Delete reference
		{
			this->element = NULL;
			return true;
		}
		else if (this->next != NULL)	  // Check next brother
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
		if (this->child == NULL)					// Name not found
			return false;
		else
			return this->child->remove(name + 1);	// Remove from child
	}
	else if (this->next != NULL)		// Check next brother
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