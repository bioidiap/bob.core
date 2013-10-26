/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 25 Oct 16:54:55 2013
 *
 * @brief Bindings to boost::random
 */

#include <blitz.array/cppapi.h>
#include <boost/random.hpp>

#if BOOST_VERSION >= 104700
#include <boost/random/discrete_distribution.hpp>
#include <bob/python/ndarray.h>
#endif

PyDoc_STRVAR(s_prefix_str, "xbob.core");
PyDoc_STRVAR(s_module_str, "random");
PyDoc_STRVAR(s_mt19937_str, "mt19937");

/* Type definition for PyBoostMt19937Object */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  boost::random::mt19937* rng;

} PyBoostMt19937Object;

/* How to create a new PyBoostMt19937Object */
PyObject* PyBoostMt19937_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBoostMt19937Object* self = (PyBoostMt19937Object*)type->tp_alloc(type, 0);

  self->rng = 0;

  return reinterpret_cast<PyObject*>(self);
}

/* How to delete a PyBoostMt19937Object */
void PyBoostMt19937_Delete (PyBoostMt19937Object* o) {

  delete o->rng;
  o->ob_type->tp_free((PyObject*)o);

}

/**
 * Formal initialization of a BoostMt19937 object
 */
static int PyBoostMt19937__init__(PyBoostMt19937Object* self, PyObject *args,
    PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"seed", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* seed = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist)) return -1;

  /* Checks seed and, if it is set, try to convert it into a know format in
   * which the RNG can be initialized with */
  if (seed) {
    Py_ssize_t s_seed = PyNumber_AsSsize_t(seed, PyExc_ValueError);
    if (PyErr_Occurred()) return -1;
    self->rng = new boost::random::mt19937(s_seed);
  }
  else {
    self->rng = new boost::random::mt19937;
  }

  return 0; ///< SUCCESS
}

PyDoc_STRVAR(s_BoostMt19937__doc__,
"A Mersenne-Twister Random Number Generator (RNG)\n\
\n\
A Random Number Generator (RNG) based on the work 'Mersenne Twister: A\n\
623-dimensionally equidistributed uniform pseudo-random number\n\
generator, Makoto Matsumoto and Takuji Nishimura, ACM Transactions\n\
on Modeling and Computer Simulation: Special Issue on Uniform Random\n\
Number Generation, Vol. 8, No. 1, January 1998, pp. 3-30'\n\
"
);

/* Sets the seed on a random number generator */
static PyObject* PyBoostMt19937_seed(PyBoostMt19937Object* self, 
    PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"seed", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* seed = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &seed)) return 0;

  Py_ssize_t s_seed = PyNumber_AsSsize_t(seed, PyExc_ValueError);
  if (PyErr_Occurred()) return 0;
  self->rng->seed(s_seed);

  Py_RETURN_NONE;
}

PyDoc_STRVAR(s_seed_str, "seed");
PyDoc_STRVAR(s_seed_doc,
"seed(x) -> None\n\
\n\
Sets the seed for this random number generator\n\
\n\
This method sets the seed for this random number generator. The input\n\
value needs to be convertible to a long integer.\n\
"
);

static PyMethodDef PyBoostMt19937_methods[] = {
    {
      s_seed_str,
      (PyCFunction)PyBoostMt19937_seed,
      METH_VARARGS|METH_KEYWORDS,
      s_seed_doc,
    },
    {0}  /* Sentinel */
};

static PyObject* PyBoostMt19937_RichCompare(PyBoostMt19937Object* self,
    PyBoostMt19937Object* other, int op) {

  switch (op) {
    case Py_EQ:
      if (*(self->rng) == *(other->rng)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    case Py_NE:
      if (*(self->rng) != *(other->rng)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    default:
      return Py_NotImplemented;
  }

}

static PyObject* PyBoostMt19937_Repr(PyBoostMt19937Object* self) {
  return PyString_FromFormat("%s.%s.%s()", s_prefix_str, s_module_str,
      s_mt19937_str);
}

PyTypeObject PyBoostMt19937_Type = {
    PyObject_HEAD_INIT(0)
    0,                                          /*ob_size*/
    "xbob.core.random.mt19937",                 /*tp_name*/
    sizeof(PyBoostMt19937Object),               /*tp_basicsize*/
    0,                                          /*tp_itemsize*/
    (destructor)PyBoostMt19937_Delete,          /*tp_dealloc*/
    0,                                          /*tp_print*/
    0,                                          /*tp_getattr*/
    0,                                          /*tp_setattr*/
    0,                                          /*tp_compare*/
    (reprfunc)PyBoostMt19937_Repr,              /*tp_repr*/
    0,                                          /*tp_as_number*/
    0,                                          /*tp_as_sequence*/
    0,                                          /*tp_as_mapping*/
    0,                                          /*tp_hash */
    0,                                          /*tp_call*/
    (reprfunc)PyBoostMt19937_Repr,              /*tp_str*/
    0,                                          /*tp_getattro*/
    0,                                          /*tp_setattro*/
    0,                                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /*tp_flags*/
    s_BoostMt19937__doc__,                      /* tp_doc */
    0,		                                      /* tp_traverse */
    0,		                                      /* tp_clear */
    (richcmpfunc)PyBoostMt19937_RichCompare,    /* tp_richcompare */
    0,		                                      /* tp_weaklistoffset */
    0,		                                      /* tp_iter */
    0,		                                      /* tp_iternext */
    PyBoostMt19937_methods,                     /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PyBoostMt19937__init__,           /* tp_init */
    0,                                          /* tp_alloc */
    PyBoostMt19937_New,                         /* tp_new */
};

static PyMethodDef random_methods[] = {
    {0}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC init_random(void)
{
  PyBoostMt19937_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBoostMt19937_Type) < 0) return;

  PyObject* m; 
  m = Py_InitModule3("_random", random_methods, 
      "boost::random classes and methods");

  /* register the type object to python */
  Py_INCREF(&PyBoostMt19937_Type);
  PyModule_AddObject(m, s_mt19937_str, (PyObject *)&PyBoostMt19937_Type);
}
