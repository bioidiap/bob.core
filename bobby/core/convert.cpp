/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 16 Oct 17:40:24 2013
 *
 * @brief Pythonic bindings to C++ constructs on bob.core
 */

#include <blitz.array/cppapi.h>

static PyMethodDef convert_methods[] = {
    {0}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC init_array(void)
{
  PyObject* m;

  m = Py_InitModule3("convert", convert_methods,
      "bob::core::array::convert bindings");

  /* imports blitz.array C-API (and NumPy's as well) */
  import_blitz_array();
}
