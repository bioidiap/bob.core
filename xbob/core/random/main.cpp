/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 25 Oct 16:54:55 2013
 *
 * @brief Bindings to boost::random
 */

#define XBOB_CORE_RANDOM_MODULE
#include <xbob.core/random.h>

static PyMethodDef module_methods[] = {
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr,
"boost::random classes and methods"
);

#define ENTRY_FUNCTION_INNER(a) init ## a
#define ENTRY_FUNCTION(a) ENTRY_FUNCTION_INNER(a)

PyMODINIT_FUNC ENTRY_FUNCTION(XBOB_CORE_RANDOM_MODULE_NAME) (void) {

  PyBoostMt19937_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBoostMt19937_Type) < 0) return;

  PyObject* m = Py_InitModule3("_library", module_methods, module_docstr);

  /* register the types to python */
  Py_INCREF(&PyBoostMt19937_Type);
  PyModule_AddObject(m, "mt19937", (PyObject *)&PyBoostMt19937_Type);
  Py_INCREF(&PyBoostUniform_Type);
  PyModule_AddObject(m, "uniform", (PyObject *)&PyBoostUniform_Type);

  static void* PyXbobCoreRandom_API[PyXbobCoreRandom_API_pointers];

  /*****************************************
   * Bindings for xbob.core.random.mt19937 *
   *****************************************/

  PyXbobCoreRandom_API[PyBoostMt19937_Type_NUM] = (void *)&PyBoostMt19937_Type;
  PyXbobCoreRandom_API[PyBoostMt19937_Check_NUM] = (void *)PyBoostMt19937_Check;
  PyXbobCoreRandom_API[PyBoostMt19937_Converter_NUM] = (void *)PyBoostMt19937_Converter;
  PyXbobCoreRandom_API[PyBoostMt19937_SimpleNew_NUM] = (void *)PyBoostMt19937_SimpleNew;
  PyXbobCoreRandom_API[PyBoostMt19937_NewWithSeed_NUM] = (void *)PyBoostMt19937_NewWithSeed;

  /*****************************************
   * Bindings for xbob.core.random.uniform *
   *****************************************/

  PyXbobCoreRandom_API[PyBoostUniform_Type_NUM] = (void *)&PyBoostUniform_Type;
  PyXbobCoreRandom_API[PyBoostUniform_Check_NUM] = (void *)PyBoostUniform_Check;
  PyXbobCoreRandom_API[PyBoostUniform_Converter_NUM] = (void *)PyBoostUniform_Converter;
  PyXbobCoreRandom_API[PyBoostUniform_SimpleNew_NUM] = (void *)PyBoostUniform_SimpleNew;

#if PY_VERSION_HEX >= 0x02070000

  /* defines the PyCapsule */

  PyObject* c_api_object = PyCapsule_New((void *)PyXbobCoreRandom_API, 
      BOOST_PP_STRINGIZE(XBOB_CORE_RANDOM_MODULE_PREFIX) "." BOOST_PP_STRINGIZE(XBOB_CORE_RANDOM_MODULE_NAME) "._C_API", 0);

#else

  PyObject* c_api_object = PyCObject_FromVoidPtr((void *)PyXbobCoreRandom_API, 0);

#endif

  if (c_api_object) PyModule_AddObject(m, "_C_API", c_api_object);

}
