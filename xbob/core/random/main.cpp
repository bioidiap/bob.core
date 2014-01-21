/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 25 Oct 16:54:55 2013
 *
 * @brief Bindings to boost::random
 */

#define XBOB_CORE_RANDOM_MODULE
#include <xbob.core/random.h>

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <xbob.blitz/capi.h>

static PyMethodDef module_methods[] = {
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr,
"boost::random classes and methods"
);

#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  XBOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  module_methods, 
  0, 0, 0, 0
};
#endif

int PyXbobCoreRandom_APIVersion = XBOB_CORE_API_VERSION;

PyMODINIT_FUNC XBOB_EXT_ENTRY_NAME (void) {

  PyBoostMt19937_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBoostMt19937_Type) < 0) return
# if PY_VERSION_HEX >= 0x03000000
    0
# endif
    ;

  PyBoostUniform_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBoostUniform_Type) < 0) return
# if PY_VERSION_HEX >= 0x03000000
    0
# endif
    ;

  PyBoostNormal_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBoostNormal_Type) < 0) return
# if PY_VERSION_HEX >= 0x03000000
    0
# endif
    ;

  PyBoostLogNormal_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBoostLogNormal_Type) < 0) return
# if PY_VERSION_HEX >= 0x03000000
    0
# endif
    ;

  PyBoostGamma_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBoostGamma_Type) < 0) return
# if PY_VERSION_HEX >= 0x03000000
    0
# endif
    ;

  PyBoostBinomial_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBoostBinomial_Type) < 0) return
# if PY_VERSION_HEX >= 0x03000000
    0
# endif
    ;

# if PY_VERSION_HEX >= 0x03000000
  PyObject* m = PyModule_Create(&module_definition);
  if (!m) return 0;
# else
  PyObject* m = Py_InitModule3(XBOB_EXT_MODULE_NAME, 
      module_methods, module_docstr);
  if (!m) return;
# endif

  /* register some constants */
  PyModule_AddIntConstant(m, "__api_version__", XBOB_CORE_API_VERSION);
  PyModule_AddStringConstant(m, "__version__", XBOB_EXT_MODULE_VERSION);

  /* register the types to python */
  Py_INCREF(&PyBoostMt19937_Type);
  PyModule_AddObject(m, "mt19937", (PyObject *)&PyBoostMt19937_Type);

  Py_INCREF(&PyBoostUniform_Type);
  PyModule_AddObject(m, "uniform", (PyObject *)&PyBoostUniform_Type);

  Py_INCREF(&PyBoostNormal_Type);
  PyModule_AddObject(m, "normal", (PyObject *)&PyBoostNormal_Type);

  Py_INCREF(&PyBoostLogNormal_Type);
  PyModule_AddObject(m, "lognormal", (PyObject *)&PyBoostLogNormal_Type);

  Py_INCREF(&PyBoostGamma_Type);
  PyModule_AddObject(m, "gamma", (PyObject *)&PyBoostGamma_Type);

  Py_INCREF(&PyBoostBinomial_Type);
  PyModule_AddObject(m, "binomial", (PyObject *)&PyBoostBinomial_Type);

  static void* PyXbobCoreRandom_API[PyXbobCoreRandom_API_pointers];

  /* exhaustive list of C APIs */
  PyXbobCoreRandom_API[PyXbobCoreRandom_APIVersion_NUM] = (void *)&PyXbobCoreRandom_APIVersion;

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

  /****************************************
   * Bindings for xbob.core.random.normal *
   ****************************************/

  PyXbobCoreRandom_API[PyBoostNormal_Type_NUM] = (void *)&PyBoostNormal_Type;
  PyXbobCoreRandom_API[PyBoostNormal_Check_NUM] = (void *)PyBoostNormal_Check;
  PyXbobCoreRandom_API[PyBoostNormal_Converter_NUM] = (void *)PyBoostNormal_Converter;
  PyXbobCoreRandom_API[PyBoostNormal_SimpleNew_NUM] = (void *)PyBoostNormal_SimpleNew;

  /*******************************************
   * Bindings for xbob.core.random.lognormal *
   *******************************************/

  PyXbobCoreRandom_API[PyBoostLogNormal_Type_NUM] = (void *)&PyBoostLogNormal_Type;
  PyXbobCoreRandom_API[PyBoostLogNormal_Check_NUM] = (void *)PyBoostLogNormal_Check;
  PyXbobCoreRandom_API[PyBoostLogNormal_Converter_NUM] = (void *)PyBoostLogNormal_Converter;
  PyXbobCoreRandom_API[PyBoostLogNormal_SimpleNew_NUM] = (void *)PyBoostLogNormal_SimpleNew;

  /***************************************
   * Bindings for xbob.core.random.gamma *
   ***************************************/

  PyXbobCoreRandom_API[PyBoostGamma_Type_NUM] = (void *)&PyBoostGamma_Type;
  PyXbobCoreRandom_API[PyBoostGamma_Check_NUM] = (void *)PyBoostGamma_Check;
  PyXbobCoreRandom_API[PyBoostGamma_Converter_NUM] = (void *)PyBoostGamma_Converter;
  PyXbobCoreRandom_API[PyBoostGamma_SimpleNew_NUM] = (void *)PyBoostGamma_SimpleNew;

  /******************************************
   * Bindings for xbob.core.random.binomial *
   ******************************************/

  PyXbobCoreRandom_API[PyBoostBinomial_Type_NUM] = (void *)&PyBoostBinomial_Type;
  PyXbobCoreRandom_API[PyBoostBinomial_Check_NUM] = (void *)PyBoostBinomial_Check;
  PyXbobCoreRandom_API[PyBoostBinomial_Converter_NUM] = (void *)PyBoostBinomial_Converter;
  PyXbobCoreRandom_API[PyBoostBinomial_SimpleNew_NUM] = (void *)PyBoostBinomial_SimpleNew;

#if PY_VERSION_HEX >= 0x02070000

  /* defines the PyCapsule */

  PyObject* c_api_object = PyCapsule_New((void *)PyXbobCoreRandom_API, 
      XBOB_EXT_MODULE_PREFIX "." XBOB_EXT_MODULE_NAME "._C_API", 0);

#else

  PyObject* c_api_object = PyCObject_FromVoidPtr((void *)PyXbobCoreRandom_API, 0);

#endif

  if (c_api_object) PyModule_AddObject(m, "_C_API", c_api_object);
  
  /* imports the NumPy C-API */
  import_array();

  /* imports blitz.array C-API */
  import_xbob_blitz();

# if PY_VERSION_HEX >= 0x03000000
  return m;
# endif

}
