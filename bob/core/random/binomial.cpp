/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 27 Oct 09:02:32 2013
 *
 * @brief Binomial distributions (with integers or floating point numbers)
 */

#define BOB_CORE_RANDOM_MODULE
#include <bob.core/random_api.h>
#include <bob.blitz/cppapi.h>
#include <boost/make_shared.hpp>

#include <bob.core/random.h>

PyDoc_STRVAR(s_binomial_str, BOB_EXT_MODULE_PREFIX ".binomial");

PyDoc_STRVAR(s_binomial_doc,
"binomial(dtype, [t=1.0, p=0.5]]) -> new binomial distribution\n\
\n\
Models a random binomial distribution\n\
\n\
This distribution class models a binomial random distribution.\n\
Such a distribution produces random numbers :math:`x` distributed\n\
with the probability density function\n\
:math:`{{t}\\choose{k}}p^k(1-p)^{t-k}`,\n\
where ``t`` and ``p`` are parameters of the distribution.\n\
\n\
.. warning::\n\
\n\
   This distribution requires that :math:`t >=0` and\n\
   that :math:`0 <= p <= 1`.\n\
\n\
"
);

/* How to create a new PyBoostBinomialObject */
static PyObject* PyBoostBinomial_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBoostBinomialObject* self = (PyBoostBinomialObject*)type->tp_alloc(type, 0);
  self->type_num = NPY_NOTYPE;
  self->distro.reset();

  return reinterpret_cast<PyObject*>(self);
}

/* How to delete a PyBoostBinomialObject */
static void PyBoostBinomial_Delete (PyBoostBinomialObject* o) {

  o->distro.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);

}

template <typename T>
boost::shared_ptr<void> make_binomial(PyObject* t, PyObject* p) {
  T ct = 1.;
  if (t) ct = PyBlitzArrayCxx_AsCScalar<T>(t);
  if (ct < 0) {
    PyErr_SetString(PyExc_ValueError, "parameter t must be >= 0");
    return boost::shared_ptr<void>();
  }
  T cp = 0.5;
  if (p) cp = PyBlitzArrayCxx_AsCScalar<T>(p);
  if (cp < 0.0 || cp > 1.0) {
    PyErr_SetString(PyExc_ValueError, "parameter p must lie in the interval [0.0, 1.0]");
    return boost::shared_ptr<void>();
  }
  return boost::make_shared<bob::core::random::binomial_distribution<int64_t,T>>(ct, cp);
}

PyObject* PyBoostBinomial_SimpleNew (int type_num, PyObject* t, PyObject* p) {

  PyBoostBinomialObject* retval = (PyBoostBinomialObject*)PyBoostBinomial_New(&PyBoostBinomial_Type, 0, 0);

  if (!retval) return 0;

  retval->type_num = type_num;

  switch(type_num) {
    case NPY_FLOAT32:
      retval->distro = make_binomial<float>(t, p);
      break;
    case NPY_FLOAT64:
      retval->distro = make_binomial<double>(t, p);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot create %s(T) with T having an unsupported numpy type number of %d (it only supports numpy.float32 or numpy.float64)", Py_TYPE(retval)->tp_name, retval->type_num);
      Py_DECREF(retval);
      return 0;
  }

  if (!retval->distro) { // a problem occurred
    Py_DECREF(retval);
    return 0;
  }

  return reinterpret_cast<PyObject*>(retval);

}

/* Implements the __init__(self) function */
static
int PyBoostBinomial_Init(PyBoostBinomialObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"dtype", "t", "p", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* t = 0;
  PyObject* p = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|OO", kwlist, &PyBlitzArray_TypenumConverter, &self->type_num, &t, &p)) return -1; ///< FAILURE

  switch(self->type_num) {
    case NPY_FLOAT32:
      self->distro = make_binomial<float>(t, p);
      break;
    case NPY_FLOAT64:
      self->distro = make_binomial<double>(t, p);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot create %s(T) with T having an unsupported numpy type number of %d (it only supports numpy.float32 or numpy.float64)", Py_TYPE(self)->tp_name, self->type_num);
      return -1;
  }

  if (!self->distro) { // a problem occurred
    return -1;
  }

  return 0; ///< SUCCESS
}

int PyBoostBinomial_Check(PyObject* o) {
  if (!o) return 0;
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBoostBinomial_Type));
}

int PyBoostBinomial_Converter(PyObject* o, PyBoostBinomialObject** a) {
  if (!PyBoostBinomial_Check(o)) return 0;
  Py_INCREF(o);
  (*a) = reinterpret_cast<PyBoostBinomialObject*>(o);
  return 1;
}

template <typename T> PyObject* get_t(PyBoostBinomialObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<bob::core::random::binomial_distribution<int64_t,T>>(self->distro)->t());
}

/**
 * Accesses the t value
 */
static PyObject* PyBoostBinomial_GetT(PyBoostBinomialObject* self) {
  switch (self->type_num) {
    case NPY_FLOAT32:
      return get_t<float>(self);
    case NPY_FLOAT64:
      return get_t<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get parameter `t` of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
}

template <typename T> PyObject* get_p(PyBoostBinomialObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<bob::core::random::binomial_distribution<int64_t,T>>(self->distro)->p());
}

/**
 * Accesses the p value
 */
static PyObject* PyBoostBinomial_GetP(PyBoostBinomialObject* self) {
  switch (self->type_num) {
    case NPY_FLOAT32:
      return get_p<float>(self);
    case NPY_FLOAT64:
      return get_p<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get parameter `p` of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
}

/**
 * Accesses the datatype
 */
static PyObject* PyBoostBinomial_GetDtype(PyBoostBinomialObject* self) {
  return reinterpret_cast<PyObject*>(PyArray_DescrFromType(self->type_num));
}

template <typename T> PyObject* reset(PyBoostBinomialObject* self) {
  boost::static_pointer_cast<bob::core::random::binomial_distribution<int64_t,T>>(self->distro)->reset();
  Py_RETURN_NONE;
}

/**
 * Resets the distribution - this is a noop for binomial distributions, here
 * only for compatibility reasons
 */
static PyObject* PyBoostBinomial_Reset(PyBoostBinomialObject* self) {
  switch (self->type_num) {
    case NPY_FLOAT32:
      return reset<float>(self);
    case NPY_FLOAT64:
      return reset<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot reset %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
}

template <typename T> PyObject* call(PyBoostBinomialObject* self, PyBoostMt19937Object* rng) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<bob::core::random::binomial_distribution<int64_t,T>>(self->distro)->operator()(*rng->rng));
}

/**
 * Calling a PyBoostBinomialObject to generate a random number
 */
static
PyObject* PyBoostBinomial_Call(PyBoostBinomialObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"rng", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBoostMt19937Object* rng = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist, &PyBoostMt19937_Type, &rng)) return 0; ///< FAILURE

  switch(self->type_num) {
    case NPY_FLOAT32:
      return call<float>(self, rng);
      break;
    case NPY_FLOAT64:
      return call<double>(self, rng);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot call %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
  }

  return 0; ///< FAILURE
}

PyDoc_STRVAR(s_reset_str, "reset");
PyDoc_STRVAR(s_reset_doc,
"x.reset() -> None\n\
\n\
After calling this method, subsequent uses of the distribution do\n\
not depend on values produced by any random number generator prior\n\
to invoking reset.\n\
"
);

static PyMethodDef PyBoostBinomial_methods[] = {
    {
      s_reset_str,
      (PyCFunction)PyBoostBinomial_Reset,
      METH_NOARGS,
      s_reset_doc,
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(s_dtype_str, "dtype");
PyDoc_STRVAR(s_dtype_doc,
"x.dtype -> numpy dtype\n\
\n\
The type of scalars produced by this binomial distribution.\n\
"
);

PyDoc_STRVAR(s_t_str, "t");
PyDoc_STRVAR(s_t_doc,
"x.t -> scalar\n\
\n\
This value corresponds to the parameter ``t`` of the distribution.\n\
"
);

PyDoc_STRVAR(s_p_str, "p");
PyDoc_STRVAR(s_p_doc,
"x.p -> scalar\n\
\n\
This value corresponds to the parameter ``p`` of the distribution.\n\
"
);

static PyGetSetDef PyBoostBinomial_getseters[] = {
    {
      s_dtype_str,
      (getter)PyBoostBinomial_GetDtype,
      0,
      s_dtype_doc,
      0,
    },
    {
      s_t_str,
      (getter)PyBoostBinomial_GetT,
      0,
      s_t_doc,
      0,
    },
    {
      s_p_str,
      (getter)PyBoostBinomial_GetP,
      0,
      s_p_doc,
      0,
    },
    {0}  /* Sentinel */
};

/**
 * Converts a scalar, that will be stolen, into a str/bytes
 */
static PyObject* scalar_to_bytes(PyObject* s) {
# if PY_VERSION_HEX >= 0x03000000
  PyObject* b = PyObject_Bytes(s);
# else
  PyObject* b = PyObject_Str(s);
# endif
  Py_DECREF(s);
  return b;
}

/**
 * Accesses the char* buffer on a str/bytes object
 */
static const char* bytes_to_charp(PyObject* s) {
# if PY_VERSION_HEX >= 0x03000000
  return PyBytes_AS_STRING(s);
# else
  return PyString_AS_STRING(s);
# endif
}

/**
 * String representation and print out
 */
static PyObject* PyBoostBinomial_Repr(PyBoostBinomialObject* self) {

  PyObject* t = PyBoostBinomial_GetT(self);
  if (!t) return 0;
  PyObject* p = PyBoostBinomial_GetP(self);
  if (!p) return 0;

  PyObject* st = scalar_to_bytes(t);
  if (!st) return 0;
  PyObject* sp = scalar_to_bytes(p);
  if (!sp) return 0;

  PyObject* retval =
# if PY_VERSION_HEX >= 0x03000000
    PyUnicode_FromFormat
#else
    PyString_FromFormat
#endif
      (
       "%s(dtype='%s', t=%s, p=%s)",
       Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(self->type_num),
       bytes_to_charp(st), bytes_to_charp(sp)
      );

  Py_DECREF(st);
  Py_DECREF(sp);

  return retval;

}

PyTypeObject PyBoostBinomial_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    s_binomial_str,                               /*tp_name*/
    sizeof(PyBoostBinomialObject),                /*tp_basicsize*/
    0,                                            /*tp_itemsize*/
    (destructor)PyBoostBinomial_Delete,           /*tp_dealloc*/
    0,                                            /*tp_print*/
    0,                                            /*tp_getattr*/
    0,                                            /*tp_setattr*/
    0,                                            /*tp_compare*/
    (reprfunc)PyBoostBinomial_Repr,               /*tp_repr*/
    0,                                            /*tp_as_number*/
    0,                                            /*tp_as_sequence*/
    0,                                            /*tp_as_mapping*/
    0,                                            /*tp_hash */
    (ternaryfunc)PyBoostBinomial_Call,            /*tp_call*/
    (reprfunc)PyBoostBinomial_Repr,               /*tp_str*/
    0,                                            /*tp_getattro*/
    0,                                            /*tp_setattro*/
    0,                                            /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /*tp_flags*/
    s_binomial_doc,                               /* tp_doc */
    0,		                                        /* tp_traverse */
    0,		                                        /* tp_clear */
    0,                                            /* tp_richcompare */
    0,		                                        /* tp_weaklistoffset */
    0,		                                        /* tp_iter */
    0,		                                        /* tp_iternext */
    PyBoostBinomial_methods,                      /* tp_methods */
    0,                                            /* tp_members */
    PyBoostBinomial_getseters,                    /* tp_getset */
    0,                                            /* tp_base */
    0,                                            /* tp_dict */
    0,                                            /* tp_descr_get */
    0,                                            /* tp_descr_set */
    0,                                            /* tp_dictoffset */
    (initproc)PyBoostBinomial_Init,               /* tp_init */
    0,                                            /* tp_alloc */
    PyBoostBinomial_New,                          /* tp_new */
};
