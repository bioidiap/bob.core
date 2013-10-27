/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 27 Oct 19:32:01 2013 
 *
 * @brief Includes for local compilation
 */

#include <Python.h>

/**
 * The MT-19937 RNG python type
 */
extern PyTypeObject PyBoostMt19937_Type;

/**
 * Registers the type into the given module
 */
void PyBoostMt19937_Register(PyObject* module);
