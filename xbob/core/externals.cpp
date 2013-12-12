/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu  7 Nov 13:50:16 2013
 *
 * @brief Binds configuration information available from bob
 */

#include <Python.h>

#include <xbob.core/config.h>

#include <bob/config.h>

#include <string>
#include <cstdlib>
#include <blitz/blitz.h>
#include <boost/preprocessor/stringize.hpp>
#include <boost/version.hpp>
#include <boost/format.hpp>
#if WITH_PERFTOOLS
#include <google/tcmalloc.h>
#endif

extern "C" {

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

}

static int dict_set(PyObject* d, const char* key, const char* value) {
  PyObject* v = Py_BuildValue("s", value);
  if (!v) return 0;
  int retval = PyDict_SetItemString(d, key, v);
  Py_DECREF(v);
  if (retval == 0) return 1; //all good
  return 0; //a problem occurred
}

static int dict_steal(PyObject* d, const char* key, PyObject* value) {
  if (!value) return 0;
  int retval = PyDict_SetItemString(d, key, value);
  Py_DECREF(value);
  if (retval == 0) return 1; //all good
  return 0; //a problem occurred
}

/**
 * Describes the version of Boost libraries installed
 */
static PyObject* boost_version() {
  boost::format f("%d.%d.%d");
  f % (BOOST_VERSION / 100000);
  f % (BOOST_VERSION / 100 % 1000);
  f % (BOOST_VERSION % 100);
  return Py_BuildValue("s", f.str().c_str());
}

/**
 * Describes the compiler version
 */
static PyObject* compiler_version() {
# if defined(__GNUC__) && !defined(__llvm__)
  boost::format f("%s.%s.%s");
  f % BOOST_PP_STRINGIZE(__GNUC__);
  f % BOOST_PP_STRINGIZE(__GNUC_MINOR__);
  f % BOOST_PP_STRINGIZE(__GNUC_PATCHLEVEL__);
  return Py_BuildValue("ss", "gcc", f.str().c_str());
# elif defined(__llvm__) && !defined(__clang__)
  return Py_BuildValue("ss", "llvm-gcc", __VERSION__);
# elif defined(__clang__)
  return Py_BuildValue("ss", "clang", __clang_version__);
# else
  return Py_BuildValue("s", "unsupported");
# endif
}

/**
 * Python version with which we compiled the extensions
 */
static PyObject* python_version() {
  boost::format f("%s.%s.%s");
  f % BOOST_PP_STRINGIZE(PY_MAJOR_VERSION);
  f % BOOST_PP_STRINGIZE(PY_MINOR_VERSION);
  f % BOOST_PP_STRINGIZE(PY_MICRO_VERSION);
  return Py_BuildValue("s", f.str().c_str());
}

/**
 * Numpy version
 */
static PyObject* numpy_version() {
  return Py_BuildValue("s", BOOST_PP_STRINGIZE(NPY_VERSION));
}

/**
 * Google profiler version, if available
 */
static PyObject* perftools_version() {
#if WITH_PERFTOOLS
  boost::format f("%s.%s.%s");
  f % BOOST_PP_STRINGIZE(TC_VERSION_MAJOR);
  f % BOOST_PP_STRINGIZE(TC_VERSION_MINOR);
  if (std::strlen(TC_VERSION_PATCH) == 0) f % "0";
  else f % BOOST_PP_STRINGIZE(TC_VERSION_PATCH);
  return Py_BuildValue("s", f.str().c_str());
#else
  return Py_BuildValue("s", "unavailable");
#endif
}

/**
 * Bob version, API version and platform
 */
static PyObject* bob_version() {
  return Py_BuildValue("sis", BOB_VERSION, BOB_API_VERSION, BOB_PLATFORM);
}

static PyObject* build_version_dictionary() {

  PyObject* retval = PyDict_New();
  if (!retval) return 0;

  if (!dict_set(retval, "Blitz++", BZ_VERSION)) {
    Py_DECREF(retval);
    return 0;
  }

  if (!dict_steal(retval, "Boost", boost_version())) {
    Py_DECREF(retval);
    return 0;
  }

  if (!dict_steal(retval, "Compiler", compiler_version())) {
    Py_DECREF(retval);
    return 0;
  }

  if (!dict_steal(retval, "Python", python_version())) {
    Py_DECREF(retval);
    return 0;
  }

  if (!dict_steal(retval, "NumPy", numpy_version())) {
    Py_DECREF(retval);
    return 0;
  }

  if (!dict_steal(retval, "Google Perftools", perftools_version())) {
    Py_DECREF(retval);
    return 0;
  }

  if (!dict_steal(retval, "Bob", bob_version())) {
    Py_DECREF(retval);
    return 0;
  }

  return retval;
}

static PyMethodDef module_methods[] = {
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr,
"Information about software used to compile the C++ Bob API"
);

PyMODINIT_FUNC XBOB_EXT_ENTRY_NAME (void) {

  PyObject* m = Py_InitModule3(XBOB_EXT_MODULE_NAME, module_methods, module_docstr);

  /* register some constants */
  PyModule_AddIntConstant(m, "__api_version__", XBOB_CORE_API_VERSION);
  PyModule_AddStringConstant(m, "__version__", XBOB_EXT_MODULE_VERSION);
  PyModule_AddObject(m, "versions", build_version_dictionary());

  /* imports the NumPy C-API */
  import_array();

}
