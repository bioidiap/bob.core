/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 27 Oct 09:01:04 2013
 *
 * @brief Bindings for the MT19937 random number generator
 */

#define BOB_CORE_RANDOM_MODULE
#include <bob.core/random_api.h>

PyDoc_STRVAR(s_mt19937_str, BOB_EXT_MODULE_PREFIX ".mt19937");

PyDoc_STRVAR(s_mt19937_doc,
"mt19937([seed]) -> new random number generator\n\
\n\
A Mersenne-Twister Random Number Generator (RNG)\n\
\n\
Constructor parameters:\n\
\n\
seed\n\
  [optional] A integral value determining the initial seed\n\
\n\
A Random Number Generator (RNG) based on the work \"*Mersenne Twister:\n\
A 623-dimensionally equidistributed uniform pseudo-random number\n\
generator, Makoto Matsumoto and Takuji Nishimura, ACM Transactions\n\
on Modeling and Computer Simulation: Special Issue on Uniform Random\n\
Number Generation, Vol. 8, No. 1, January 1998, pp. 3-30*\"\n\
\n\
Objects of this class support comparison operators such as ``==``\n\
or ``!=`` and setting the seed with the method ``seed(int)``. Two\n\
random number generators are equal if they are at the same state -\n\
i.e. they have been initialized with the same seed and have been\n\
called the same number of times for number generation.\n\
"
);

/* How to create a new PyBoostMt19937Object */
static PyObject* PyBoostMt19937_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBoostMt19937Object* self = (PyBoostMt19937Object*)type->tp_alloc(type, 0);

  self->rng = 0;

  return reinterpret_cast<PyObject*>(self);
}

PyObject* PyBoostMt19937_SimpleNew () {

  PyBoostMt19937Object* retval = (PyBoostMt19937Object*)PyBoostMt19937_New(&PyBoostMt19937_Type, 0, 0);

  if (!retval) return 0;

  retval->rng = new boost::mt19937;

  return reinterpret_cast<PyObject*>(retval);

}

PyObject* PyBoostMt19937_NewWithSeed (Py_ssize_t seed) {

  PyBoostMt19937Object* retval = (PyBoostMt19937Object*)PyBoostMt19937_New(&PyBoostMt19937_Type, 0, 0);

  if (!retval) return 0;

  retval->rng = new boost::mt19937(seed);

  return reinterpret_cast<PyObject*>(retval);

}

int PyBoostMt19937_Check(PyObject* o) {
  if (!o) return 0;
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBoostMt19937_Type));
}

int PyBoostMt19937_Converter(PyObject* o, PyBoostMt19937Object** a) {
  if (!PyBoostMt19937_Check(o)) return 0;
  Py_INCREF(o);
  (*a) = reinterpret_cast<PyBoostMt19937Object*>(o);
  return 1;
}

static void PyBoostMt19937_Delete (PyBoostMt19937Object* o) {

  delete o->rng;
  Py_TYPE(o)->tp_free((PyObject*)o);

}

/* The __init__(self) method */
static int PyBoostMt19937_Init(PyBoostMt19937Object* self, PyObject *args,
    PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"seed", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* seed = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &seed)) return -1;

  /* Checks seed and, if it is set, try to convert it into a know format in
   * which the RNG can be initialized with */
  if (seed) {
    Py_ssize_t s_seed = PyNumber_AsSsize_t(seed, PyExc_ValueError);
    if (PyErr_Occurred()) return -1;
    self->rng = new boost::mt19937(s_seed);
  }
  else {
    self->rng = new boost::mt19937;
  }

  return 0; ///< SUCCESS
}

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
This method sets the seed for this random number generator. The\n\
input value needs to be convertible to a long integer.\n\
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
    PyObject* other, int op) {

  if (!PyBoostMt19937_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'",
        s_seed_str, other->ob_type->tp_name);
    return 0;
  }

  PyBoostMt19937Object* other_ = reinterpret_cast<PyBoostMt19937Object*>(other);

  switch (op) {
    case Py_EQ:
      if (*(self->rng) == *(other_->rng)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    case Py_NE:
      if (*(self->rng) != *(other_->rng)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }

}

static PyObject* PyBoostMt19937_Repr(PyBoostMt19937Object* self) {
  return PyUnicode_FromFormat("%s()", Py_TYPE(self)->tp_name);
}

PyTypeObject PyBoostMt19937_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    s_mt19937_str,                              /*tp_name*/
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
    s_mt19937_doc,                              /* tp_doc */
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
    (initproc)PyBoostMt19937_Init,              /* tp_init */
    0,                                          /* tp_alloc */
    PyBoostMt19937_New,                         /* tp_new */
};
