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

	// @}


		/** @name Other functions */
public:
	// @{

	/** Print tree */
	void print();

	/** Print tree atlas */
	void printAtlas();

private:
		/** Print tree atlas
	 * @param [in]	depth		Tree depth for jumps
	 * @param [out]	lastJump	Last jump depth
	 */
	void printAtlas(unsigned long depth, unsigned long &lastJump);

public:
	/** Get the size of the tree (number of elements and leafs)
	 * @param [out]	nElements	Number of elements
	 * @param [out]	nLeafs		Number of leafs
	 */
	void getSize(unsigned long &nElements, unsigned long &nLeafs);

	/** Get atlas size
	 * @return	Atlas size
	 */
	unsigned long getAtlasSize();

private:
	/** Get atlas size
	 * @return	Atlas size
	 */
	unsigned long getAtlasSizeAux();

public:
	/** Get atlas
	 * @param [out]	Buffer with the atlas
	 * @return		true if the atlas was generated properly,
	 * 				false otherwise
	 */
	bool getAtlas(char *atlas);

private:
	/** Get atlas
	 * @param [out]	atlas		Buffer with the atlas
	 * @param [out]	pos			Position in the atlas buffer (will finish with
	 * 							the atlas size)
	 * @param [in]	depth		Tree depth for jumps
	 * #param [out] lastJump	Last jump depth
	 * @return					true if the atlas was generated properly,
	 * 							false otherwise
	 */
	bool getAtlas(
		char *atlas, unsigned long &pos,
		unsigned long depth, unsigned long &lastJump);

public:
	/** Set atlas
	 *
	 */
	bool setAtlas(char *atlas, unsigned long atlasSize, void **elements);

private:
	/** Set atlas
	 *
	 */
	bool setAtlas(
		char *atlas, unsigned long atlasSize, unsigned long &atlasPos,
		void **elements, unsigned long &elementPos);

	// @}
};

#endif	/* TRIE_H_ */
