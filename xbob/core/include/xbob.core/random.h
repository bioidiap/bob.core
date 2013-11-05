/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 30 Oct 07:40:47 2013 
 *
 * @brief C/C++-API for the random module
 */

#ifndef XBOB_CORE_RANDOM_H
#define XBOB_CORE_RANDOM_H

#include <xbob.core/config.h>
#include <boost/preprocessor/stringize.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/random.hpp>
#include <Python.h>

#define XBOB_CORE_RANDOM_MODULE_PREFIX xbob.core.random
#define XBOB_CORE_RANDOM_MODULE_NAME _library

/*******************
 * C API functions *
 *******************/

#define PyXbobCoreRandom_APIVersion_NUM 0
#define PyXbobCoreRandom_APIVersion_TYPE int

/*****************************************
 * Bindings for xbob.core.random.mt19937 *
 *****************************************/

/* Type definition for PyBoostMt19937Object */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  boost::random::mt19937* rng;

} PyBoostMt19937Object;

#define PyBoostMt19937_Type_NUM 1
#define PyBoostMt19937_Type_TYPE PyTypeObject

#define PyBoostMt19937_Check_NUM 2
#define PyBoostMt19937_Check_RET int
#define PyBoostMt19937_Check_PROTO (PyObject* o)

#define PyBoostMt19937_Converter_NUM 3
#define PyBoostMt19937_Converter_RET int
#define PyBoostMt19937_Converter_PROTO (PyObject* o, PyBoostMt19937Object** a)

#define PyBoostMt19937_SimpleNew_NUM 4
#define PyBoostMt19937_SimpleNew_RET PyObject*
#define PyBoostMt19937_SimpleNew_PROTO ()

#define PyBoostMt19937_NewWithSeed_NUM 5
#define PyBoostMt19937_NewWithSeed_RET PyObject*
#define PyBoostMt19937_NewWithSeed_PROTO (Py_ssize_t seed)

/*****************************************
 * Bindings for xbob.core.random.uniform *
 *****************************************/

/* Type definition for PyBoostUniformObject */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  int type_num;
  boost::shared_ptr<void> distro;

} PyBoostUniformObject;

#define PyBoostUniform_Type_NUM 6
#define PyBoostUniform_Type_TYPE PyTypeObject

#define PyBoostUniform_Check_NUM 7
#define PyBoostUniform_Check_RET int
#define PyBoostUniform_Check_PROTO (PyObject* o)

#define PyBoostUniform_Converter_NUM 8
#define PyBoostUniform_Converter_RET int
#define PyBoostUniform_Converter_PROTO (PyObject* o, PyBoostUniformObject** a)

#define PyBoostUniform_SimpleNew_NUM 9
#define PyBoostUniform_SimpleNew_RET PyObject*
#define PyBoostUniform_SimpleNew_PROTO (int type_num, PyObject* min, PyObject* max)

/****************************************
 * Bindings for xbob.core.random.normal *
 ****************************************/

/* Type definition for PyBoostNormalObject */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  int type_num;
  boost::shared_ptr<void> distro;

} PyBoostNormalObject;

#define PyBoostNormal_Type_NUM 10
#define PyBoostNormal_Type_TYPE PyTypeObject

#define PyBoostNormal_Check_NUM 11
#define PyBoostNormal_Check_RET int
#define PyBoostNormal_Check_PROTO (PyObject* o)

#define PyBoostNormal_Converter_NUM 12
#define PyBoostNormal_Converter_RET int
#define PyBoostNormal_Converter_PROTO (PyObject* o, PyBoostNormalObject** a)

#define PyBoostNormal_SimpleNew_NUM 13
#define PyBoostNormal_SimpleNew_RET PyObject*
#define PyBoostNormal_SimpleNew_PROTO (int type_num, PyObject* mean, PyObject* sigma)

/*******************************************
 * Bindings for xbob.core.random.lognormal *
 *******************************************/

/* Type definition for PyBoostLogNormalObject */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  int type_num;
  boost::shared_ptr<void> distro;

} PyBoostLogNormalObject;

#define PyBoostLogNormal_Type_NUM 14
#define PyBoostLogNormal_Type_TYPE PyTypeObject

#define PyBoostLogNormal_Check_NUM 15
#define PyBoostLogNormal_Check_RET int
#define PyBoostLogNormal_Check_PROTO (PyObject* o)

#define PyBoostLogNormal_Converter_NUM 16
#define PyBoostLogNormal_Converter_RET int
#define PyBoostLogNormal_Converter_PROTO (PyObject* o, PyBoostLogNormalObject** a)

#define PyBoostLogNormal_SimpleNew_NUM 17
#define PyBoostLogNormal_SimpleNew_RET PyObject*
#define PyBoostLogNormal_SimpleNew_PROTO (int type_num, PyObject* mean, PyObject* sigma)

/***************************************
 * Bindings for xbob.core.random.gamma *
 ***************************************/

/* Type definition for PyBoostGammaObject */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  int type_num;
  boost::shared_ptr<void> distro;

} PyBoostGammaObject;

#define PyBoostGamma_Type_NUM 18
#define PyBoostGamma_Type_TYPE PyTypeObject

#define PyBoostGamma_Check_NUM 19
#define PyBoostGamma_Check_RET int
#define PyBoostGamma_Check_PROTO (PyObject* o)

#define PyBoostGamma_Converter_NUM 20
#define PyBoostGamma_Converter_RET int
#define PyBoostGamma_Converter_PROTO (PyObject* o, PyBoostGammaObject** a)

#define PyBoostGamma_SimpleNew_NUM 21
#define PyBoostGamma_SimpleNew_RET PyObject*
#define PyBoostGamma_SimpleNew_PROTO (int type_num, PyObject* alpha, PyObject* beta)

/******************************************
 * Bindings for xbob.core.random.binomial *
 ******************************************/

/* Type definition for PyBoostBinomialObject */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  int type_num;
  boost::shared_ptr<void> distro;

} PyBoostBinomialObject;

#define PyBoostBinomial_Type_NUM 22
#define PyBoostBinomial_Type_TYPE PyTypeObject

#define PyBoostBinomial_Check_NUM 23
#define PyBoostBinomial_Check_RET int
#define PyBoostBinomial_Check_PROTO (PyObject* o)

#define PyBoostBinomial_Converter_NUM 24
#define PyBoostBinomial_Converter_RET int
#define PyBoostBinomial_Converter_PROTO (PyObject* o, PyBoostBinomialObject** a)

#define PyBoostBinomial_SimpleNew_NUM 25
#define PyBoostBinomial_SimpleNew_RET PyObject*
#define PyBoostBinomial_SimpleNew_PROTO (int type_num, PyObject* alpha, PyObject* beta)

/* Total number of C API pointers */
#define PyXbobCoreRandom_API_pointers 26

#ifdef XBOB_CORE_RANDOM_MODULE

  /* This section is used when compiling `xbob.core.random' itself */

  extern int PyXbobCoreRandom_APIVersion;

  /*****************************************
   * Bindings for xbob.core.random.mt19937 *
   *****************************************/

  extern PyBoostMt19937_Type_TYPE PyBoostMt19937_Type;

  PyBoostMt19937_Check_RET PyBoostMt19937_Check PyBoostMt19937_Check_PROTO;

  PyBoostMt19937_Converter_RET PyBoostMt19937_Converter PyBoostMt19937_Converter_PROTO;

  PyBoostMt19937_SimpleNew_RET PyBoostMt19937_SimpleNew PyBoostMt19937_SimpleNew_PROTO;

  PyBoostMt19937_NewWithSeed_RET PyBoostMt19937_NewWithSeed PyBoostMt19937_NewWithSeed_PROTO;

  /*****************************************
   * Bindings for xbob.core.random.uniform *
   *****************************************/

  extern PyBoostUniform_Type_TYPE PyBoostUniform_Type;

  PyBoostUniform_Check_RET PyBoostUniform_Check PyBoostUniform_Check_PROTO;

  PyBoostUniform_Converter_RET PyBoostUniform_Converter PyBoostUniform_Converter_PROTO;

  PyBoostUniform_SimpleNew_RET PyBoostUniform_SimpleNew PyBoostUniform_SimpleNew_PROTO;

  /****************************************
   * Bindings for xbob.core.random.normal *
   ****************************************/

  extern PyBoostNormal_Type_TYPE PyBoostNormal_Type;

  PyBoostNormal_Check_RET PyBoostNormal_Check PyBoostNormal_Check_PROTO;

  PyBoostNormal_Converter_RET PyBoostNormal_Converter PyBoostNormal_Converter_PROTO;

  PyBoostNormal_SimpleNew_RET PyBoostNormal_SimpleNew PyBoostNormal_SimpleNew_PROTO;

  /*******************************************
   * Bindings for xbob.core.random.lognormal *
   *******************************************/

  extern PyBoostLogNormal_Type_TYPE PyBoostLogNormal_Type;

  PyBoostLogNormal_Check_RET PyBoostLogNormal_Check PyBoostLogNormal_Check_PROTO;

  PyBoostLogNormal_Converter_RET PyBoostLogNormal_Converter PyBoostLogNormal_Converter_PROTO;

  PyBoostLogNormal_SimpleNew_RET PyBoostLogNormal_SimpleNew PyBoostLogNormal_SimpleNew_PROTO;

  /***************************************
   * Bindings for xbob.core.random.gamma *
   ***************************************/

  extern PyBoostGamma_Type_TYPE PyBoostGamma_Type;

  PyBoostGamma_Check_RET PyBoostGamma_Check PyBoostGamma_Check_PROTO;

  PyBoostGamma_Converter_RET PyBoostGamma_Converter PyBoostGamma_Converter_PROTO;

  PyBoostGamma_SimpleNew_RET PyBoostGamma_SimpleNew PyBoostGamma_SimpleNew_PROTO;

  /******************************************
   * Bindings for xbob.core.random.binomial *
   ******************************************/

  extern PyBoostBinomial_Type_TYPE PyBoostBinomial_Type;

  PyBoostBinomial_Check_RET PyBoostBinomial_Check PyBoostBinomial_Check_PROTO;

  PyBoostBinomial_Converter_RET PyBoostBinomial_Converter PyBoostBinomial_Converter_PROTO;

  PyBoostBinomial_SimpleNew_RET PyBoostBinomial_SimpleNew PyBoostBinomial_SimpleNew_PROTO;

#else

  /* This section is used in modules that use `blitz.array's' C-API */

/************************************************************************
 * Macros to avoid symbol collision and allow for separate compilation. *
 * We pig-back on symbols already defined for NumPy and apply the same  *
 * set of rules here, creating our own API symbol names.                *
 ************************************************************************/

#  if defined(PY_ARRAY_UNIQUE_SYMBOL)
#    define XBOB_CORE_RANDOM_MAKE_API_NAME_INNER(a) XBOB_CORE_RANDOM_ ## a
#    define XBOB_CORE_RANDOM_MAKE_API_NAME(a) XBOB_CORE_RANDOM_MAKE_API_NAME_INNER(a)
#    define PyBlitzArray_API XBOB_CORE_RANDOM_MAKE_API_NAME(PY_ARRAY_UNIQUE_SYMBOL)
#  endif

#  if defined(NO_IMPORT_ARRAY)
  extern void **PyXbobCoreRandom_API;
#  else
#    if defined(PY_ARRAY_UNIQUE_SYMBOL)
  void **PyXbobCoreRandom_API;
#    else
  static void **PyXbobCoreRandom_API=NULL;
#    endif
#  endif

#define PyXbobCoreRandom_APIVersion (*(PyXbobCoreRandom_APIVersion_TYPE *)PyXbobCoreRandom_API[PyXbobCoreRandom_APIVersion_NUM])

  static void **PyXbobCoreRandom_API;

  /*****************************************
   * Bindings for xbob.core.random.mt19937 *
   *****************************************/

# define PyBoostMt19937_Type (*(PyBoostMt19937_Type_TYPE *)PyXbobCoreRandom_API[PyBoostMt19937_Type_NUM])

# define PyBoostMt19937_Check (*(PyBoostMt19937_Check_RET (*)PyBoostMt19937_Check_PROTO) PyXbobCoreRandom_API[PyBoostMt19937_Check_NUM])

# define PyBoostMt19937_Converter (*(PyBoostMt19937_Converter_RET (*)PyBoostMt19937_Converter_PROTO) PyXbobCoreRandom_API[PyBoostMt19937_Converter_NUM])

# define PyBoostMt19937_SimpleNew (*(PyBoostMt19937_SimpleNew_RET (*)PyBoostMt19937_SimpleNew_PROTO) PyXbobCoreRandom_API[PyBoostMt19937_SimpleNew_NUM])

# define PyBoostMt19937_NewWithSeed (*(PyBoostMt19937_NewWithSeed_RET (*)PyBoostMt19937_NewWithSeed_PROTO) PyXbobCoreRandom_API[PyBoostMt19937_NewWithSeed_NUM])

  /*****************************************
   * Bindings for xbob.core.random.uniform *
   *****************************************/

# define PyBoostUniform_Type (*(PyBoostUniform_Type_TYPE *)PyXbobCoreRandom_API[PyBoostUniform_Type_NUM])

# define PyBoostUniform_Check (*(PyBoostUniform_Check_RET (*)PyBoostUniform_Check_PROTO) PyXbobCoreRandom_API[PyBoostUniform_Check_NUM])

# define PyBoostUniform_Converter (*(PyBoostUniform_Converter_RET (*)PyBoostUniform_Converter_PROTO) PyXbobCoreRandom_API[PyBoostUniform_Converter_NUM])

# define PyBoostUniform_SimpleNew (*(PyBoostUniform_SimpleNew_RET (*)PyBoostUniform_SimpleNew_PROTO) PyXbobCoreRandom_API[PyBoostUniform_SimpleNew_NUM])

  /****************************************
   * Bindings for xbob.core.random.normal *
   ****************************************/

# define PyBoostNormal_Type (*(PyBoostNormal_Type_TYPE *)PyXbobCoreRandom_API[PyBoostNormal_Type_NUM])

# define PyBoostNormal_Check (*(PyBoostNormal_Check_RET (*)PyBoostNormal_Check_PROTO) PyXbobCoreRandom_API[PyBoostNormal_Check_NUM])

# define PyBoostNormal_Converter (*(PyBoostNormal_Converter_RET (*)PyBoostNormal_Converter_PROTO) PyXbobCoreRandom_API[PyBoostNormal_Converter_NUM])

# define PyBoostNormal_SimpleNew (*(PyBoostNormal_SimpleNew_RET (*)PyBoostNormal_SimpleNew_PROTO) PyXbobCoreRandom_API[PyBoostNormal_SimpleNew_NUM])

  /*******************************************
   * Bindings for xbob.core.random.lognormal *
   *******************************************/

# define PyBoostLogNormal_Type (*(PyBoostLogNormal_Type_TYPE *)PyXbobCoreRandom_API[PyBoostLogNormal_Type_NUM])

# define PyBoostLogNormal_Check (*(PyBoostLogNormal_Check_RET (*)PyBoostLogNormal_Check_PROTO) PyXbobCoreRandom_API[PyBoostLogNormal_Check_NUM])

# define PyBoostLogNormal_Converter (*(PyBoostLogNormal_Converter_RET (*)PyBoostLogNormal_Converter_PROTO) PyXbobCoreRandom_API[PyBoostLogNormal_Converter_NUM])

# define PyBoostLogNormal_SimpleNew (*(PyBoostLogNormal_SimpleNew_RET (*)PyBoostLogNormal_SimpleNew_PROTO) PyXbobCoreRandom_API[PyBoostLogNormal_SimpleNew_NUM])

  /***************************************
   * Bindings for xbob.core.random.gamma *
   ***************************************/

# define PyBoostGamma_Type (*(PyBoostGamma_Type_TYPE *)PyXbobCoreRandom_API[PyBoostGamma_Type_NUM])

# define PyBoostGamma_Check (*(PyBoostGamma_Check_RET (*)PyBoostGamma_Check_PROTO) PyXbobCoreRandom_API[PyBoostGamma_Check_NUM])

# define PyBoostGamma_Converter (*(PyBoostGamma_Converter_RET (*)PyBoostGamma_Converter_PROTO) PyXbobCoreRandom_API[PyBoostGamma_Converter_NUM])

# define PyBoostGamma_SimpleNew (*(PyBoostGamma_SimpleNew_RET (*)PyBoostGamma_SimpleNew_PROTO) PyXbobCoreRandom_API[PyBoostGamma_SimpleNew_NUM])

  /******************************************
   * Bindings for xbob.core.random.binomial *
   ******************************************/

# define PyBoostBinomial_Type (*(PyBoostBinomial_Type_TYPE *)PyXbobCoreRandom_API[PyBoostBinomial_Type_NUM])

# define PyBoostBinomial_Check (*(PyBoostBinomial_Check_RET (*)PyBoostBinomial_Check_PROTO) PyXbobCoreRandom_API[PyBoostBinomial_Check_NUM])

# define PyBoostBinomial_Converter (*(PyBoostBinomial_Converter_RET (*)PyBoostBinomial_Converter_PROTO) PyXbobCoreRandom_API[PyBoostBinomial_Converter_NUM])

# define PyBoostBinomial_SimpleNew (*(PyBoostBinomial_SimpleNew_RET (*)PyBoostBinomial_SimpleNew_PROTO) PyXbobCoreRandom_API[PyBoostBinomial_SimpleNew_NUM])

  /**
   * Returns -1 on error, 0 on success. PyCapsule_Import will set an exception
   * if there's an error.
   */
  static int import_xbob_core_random(void) {

#if PY_VERSION_HEX >= 0x02070000

    /* New Python API support for library loading */

    PyXbobCoreRandom_API = (void **)PyCapsule_Import(BOOST_PP_STRINGIZE(XBOB_CORE_RANDOM_MODULE_PREFIX) "." BOOST_PP_STRINGIZE(XBOB_CORE_RANDOM_MODULE_NAME) "._C_API", 0);

    if (!PyXbobCoreRandom_API) return -1;

#else

    /* Old-style Python API support for library loading */

    PyObject *c_api_object;
    PyObject *module;

    module = PyImport_ImportModule(BOOST_PP_STRINGIZE(XBOB_CORE_RANDOM_MODULE_PREFIX) "." BOOST_PP_STRINGIZE(XBOB_CORE_RANDOM_MODULE_NAME));

    if (module == NULL) return -1;

    c_api_object = PyObject_GetAttrString(module, "_C_API");

    if (c_api_object == NULL) {
      Py_DECREF(module);
      return -1;
    }

    if (PyCObject_Check(c_api_object)) {
      PyXbobCoreRandom_API = (void **)PyCObject_AsVoidPtr(c_api_object);
    }

    Py_DECREF(c_api_object);
    Py_DECREF(module);

#endif
    
    /* Checks that the imported version matches the compiled version */
    int imported_version = *(int*)PyXbobCoreRandom_API[PyXbobCoreRandom_APIVersion_NUM];

    if (XBOB_CORE_API_VERSION != imported_version) {
      PyErr_Format(PyExc_RuntimeError, "%s.%s import error: you compiled against API version 0x%04x, but are now importing an API with version 0x%04x which is not compatible - check your Python runtime environment for errors", BOOST_PP_STRINGIZE(XBOB_CORE_RANDOM_MODULE_PREFIX), BOOST_PP_STRINGIZE(XBOB_CORE_RANDOM_MODULE_NAME), XBOB_CORE_API_VERSION, imported_version);
      return -1;
    }

    /* If you get to this point, all is good */
    return 0;

  }

#endif /* XBOB_CORE_RANDOM_MODULE */

#endif /* XBOB_CORE_RANDOM_H */
