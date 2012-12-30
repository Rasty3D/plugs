/**
 * @file	trie.h
 * @author	Jesús Ortiz
 * @version	1.0
 * @date	03-Sept-2012
 * @brief	Trie or Prefix Tree
 */

#ifndef TRIE_H_
#define TRIE_H_

/*
 * INCLUDES
 */

#include <iostream>
#include <stdlib.h>


/*
 * CLASS
 */

/** Trie or Prefix Tree */
class Trie
{
		/** @name Leaf information */
private:
	// @{

	/** First child of the leaf (NULL if there is no children) */
	Trie *child;

	/** Next brother (NULL if it's the last brother) */
	Trie *next;

	/** Pointer to the element of the leaf (NULL if no element) */
	void *element;

	/** Key of the leaf */
	char character;

	// @}

		/** @name Constructor and destructor */
public:
	// @{

	/** Constructor */
	Trie();

	/** Destructor */
	~Trie();

	// @}

		/** @name Manage elements */
public:
	// @{

	/** Add an element to the current position
	 *  @param [in] name		Name of the element
	 *  @param [in] element		Pointer to the element
	 *  @param [in] overwrite	If the element already exists and
	 *  						the flag is true then overwrites
	 *  						the element
	 *  @return					true if the element was inserted,
	 *  						false otherwise
	 */
	bool add(const char *name, void *element, bool overwrite);

	/** Get element from the current position
	 *  @param [in] name		Name of the element
	 *  @return					Pointer to the element. If the element was not
	 *  						found, the return is NULL
	 */
	void *get(const char *name);

	/** Removes the reference to an element. This function only removes the
	 *  reference to the element, it doesn't actually remove the element
	 *  from the tree, because there could be common parts with other elements
	 *  @param [in] name		Name of the element
	 *  @return					true if the element was removed,
	 *  						false otherwise
	 */
	bool remove(const char *name);

	/** Clean tree */
	void clear();

	/** Print tree */
	void print();

	// @}
};

#endif	/* TRIE_H_ */
