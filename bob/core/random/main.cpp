/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 25 Oct 16:54:55 2013
 *
 * @brief Bindings to boost::random
 */

#define BOB_CORE_RANDOM_MODULE
#include <bob.core/random_api.h>

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>

static PyMethodDef module_methods[] = {
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr,
"boost::random classes and methods"
);

#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  BOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  module_methods,
  0, 0, 0, 0
};
#endif

int PyBobCoreRandom_APIVersion = BOB_CORE_API_VERSION;

static PyObject* create_module (void) {

  PyBoostMt19937_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBoostMt19937_Type) < 0) return 0;

  PyBoostUniform_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBoostUniform_Type) < 0) return 0;

  PyBoostNormal_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBoostNormal_Type) < 0) return 0;

  PyBoostLogNormal_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBoostLogNormal_Type) < 0) return 0;

  PyBoostGamma_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBoostGamma_Type) < 0) return 0;

  PyBoostBinomial_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBoostBinomial_Type) < 0) return 0;

  PyBoostDiscrete_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBoostDiscrete_Type) < 0) return 0;

# if PY_VERSION_HEX >= 0x03000000
  PyObject* m = PyModule_Create(&module_definition);
# else
  PyObject* m = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
# endif
  if (!m) return 0;
  auto m_ = make_safe(m);

  /* register the types to python */
  Py_INCREF(&PyBoostMt19937_Type);
  if (PyModule_AddObject(m, "mt19937", (PyObject *)&PyBoostMt19937_Type) < 0) return 0;

  Py_INCREF(&PyBoostUniform_Type);
  if (PyModule_AddObject(m, "uniform", (PyObject *)&PyBoostUniform_Type) < 0) return 0;

  Py_INCREF(&PyBoostNormal_Type);
  if (PyModule_AddObject(m, "normal", (PyObject *)&PyBoostNormal_Type) < 0) return 0;

  Py_INCREF(&PyBoostLogNormal_Type);
  if (PyModule_AddObject(m, "lognormal", (PyObject *)&PyBoostLogNormal_Type) < 0) return 0;

  Py_INCREF(&PyBoostGamma_Type);
  if (PyModule_AddObject(m, "gamma", (PyObject *)&PyBoostGamma_Type) < 0) return 0;

  Py_INCREF(&PyBoostBinomial_Type);
  if (PyModule_AddObject(m, "binomial", (PyObject *)&PyBoostBinomial_Type) < 0) return 0;

  Py_INCREF(&PyBoostDiscrete_Type);
  if (PyModule_AddObject(m, "discrete", (PyObject *)&PyBoostDiscrete_Type) < 0) return 0;

  static void* PyBobCoreRandom_API[PyBobCoreRandom_API_pointers];

  /* exhaustive list of C APIs */
  PyBobCoreRandom_API[PyBobCoreRandom_APIVersion_NUM] = (void *)&PyBobCoreRandom_APIVersion;

  /*****************************************
   * Bindings for bob.core.random.mt19937 *
   *****************************************/

  PyBobCoreRandom_API[PyBoostMt19937_Type_NUM] = (void *)&PyBoostMt19937_Type;
  PyBobCoreRandom_API[PyBoostMt19937_Check_NUM] = (void *)PyBoostMt19937_Check;
  PyBobCoreRandom_API[PyBoostMt19937_Converter_NUM] = (void *)PyBoostMt19937_Converter;
  PyBobCoreRandom_API[PyBoostMt19937_SimpleNew_NUM] = (void *)PyBoostMt19937_SimpleNew;
  PyBobCoreRandom_API[PyBoostMt19937_NewWithSeed_NUM] = (void *)PyBoostMt19937_NewWithSeed;

  /*****************************************
   * Bindings for bob.core.random.uniform *
   *****************************************/

  PyBobCoreRandom_API[PyBoostUniform_Type_NUM] = (void *)&PyBoostUniform_Type;
  PyBobCoreRandom_API[PyBoostUniform_Check_NUM] = (void *)PyBoostUniform_Check;
  PyBobCoreRandom_API[PyBoostUniform_Converter_NUM] = (void *)PyBoostUniform_Converter;
  PyBobCoreRandom_API[PyBoostUniform_SimpleNew_NUM] = (void *)PyBoostUniform_SimpleNew;

  /****************************************
   * Bindings for bob.core.random.normal *
   ****************************************/

  PyBobCoreRandom_API[PyBoostNormal_Type_NUM] = (void *)&PyBoostNormal_Type;
  PyBobCoreRandom_API[PyBoostNormal_Check_NUM] = (void *)PyBoostNormal_Check;
  PyBobCoreRandom_API[PyBoostNormal_Converter_NUM] = (void *)PyBoostNormal_Converter;
  PyBobCoreRandom_API[PyBoostNormal_SimpleNew_NUM] = (void *)PyBoostNormal_SimpleNew;

  /*******************************************
   * Bindings for bob.core.random.lognormal *
   *******************************************/

  PyBobCoreRandom_API[PyBoostLogNormal_Type_NUM] = (void *)&PyBoostLogNormal_Type;
  PyBobCoreRandom_API[PyBoostLogNormal_Check_NUM] = (void *)PyBoostLogNormal_Check;
  PyBobCoreRandom_API[PyBoostLogNormal_Converter_NUM] = (void *)PyBoostLogNormal_Converter;
  PyBobCoreRandom_API[PyBoostLogNormal_SimpleNew_NUM] = (void *)PyBoostLogNormal_SimpleNew;

  /***************************************
   * Bindings for bob.core.random.gamma *
   ***************************************/

  PyBobCoreRandom_API[PyBoostGamma_Type_NUM] = (void *)&PyBoostGamma_Type;
  PyBobCoreRandom_API[PyBoostGamma_Check_NUM] = (void *)PyBoostGamma_Check;
  PyBobCoreRandom_API[PyBoostGamma_Converter_NUM] = (void *)PyBoostGamma_Converter;
  PyBobCoreRandom_API[PyBoostGamma_SimpleNew_NUM] = (void *)PyBoostGamma_SimpleNew;

  /******************************************
   * Bindings for bob.core.random.binomial *
   ******************************************/

  PyBobCoreRandom_API[PyBoostBinomial_Type_NUM] = (void *)&PyBoostBinomial_Type;
  PyBobCoreRandom_API[PyBoostBinomial_Check_NUM] = (void *)PyBoostBinomial_Check;
  PyBobCoreRandom_API[PyBoostBinomial_Converter_NUM] = (void *)PyBoostBinomial_Converter;
  PyBobCoreRandom_API[PyBoostBinomial_SimpleNew_NUM] = (void *)PyBoostBinomial_SimpleNew;

  /******************************************
   * Bindings for bob.core.random.discrete *
   ******************************************/

  PyBobCoreRandom_API[PyBoostDiscrete_Type_NUM] = (void *)&PyBoostDiscrete_Type;
  PyBobCoreRandom_API[PyBoostDiscrete_Check_NUM] = (void *)PyBoostDiscrete_Check;
  PyBobCoreRandom_API[PyBoostDiscrete_Converter_NUM] = (void *)PyBoostDiscrete_Converter;
  PyBobCoreRandom_API[PyBoostDiscrete_SimpleNew_NUM] = (void *)PyBoostDiscrete_SimpleNew;

#if PY_VERSION_HEX >= 0x02070000

  /* defines the PyCapsule */

  PyObject* c_api_object = PyCapsule_New((void *)PyBobCoreRandom_API,
      BOB_EXT_MODULE_PREFIX "." BOB_EXT_MODULE_NAME "._C_API", 0);

#else

  PyObject* c_api_object = PyCObject_FromVoidPtr((void *)PyBobCoreRandom_API, 0);

#endif

  if (c_api_object) PyModule_AddObject(m, "_C_API", c_api_object);

  /* imports dependencies */
  if (import_bob_blitz() < 0) return 0;

  return Py_BuildValue("O", m);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
