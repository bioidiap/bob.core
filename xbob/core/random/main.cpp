/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 25 Oct 16:54:55 2013
 *
 * @brief Bindings to boost::random
 */

#include <blitz.array/cppapi.h>
#include <mt19937.h>
//#include <uniform.h>

static PyMethodDef module_methods[] = {
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr,
"boost::random classes and methods"
);

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC init_random(void)
{
  PyBoostMt19937_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBoostMt19937_Type) < 0) return;

  PyObject* m; 
  m = Py_InitModule3("_random", module_methods, module_docstr);

  /* register the types to python */
  PyBoostMt19937_Register(m);
}
