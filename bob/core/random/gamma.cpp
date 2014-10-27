/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 27 Oct 09:02:32 2013
 *
 * @brief Gamma distributions (with integers or floating point numbers)
 */

#define BOB_CORE_RANDOM_MODULE
#include <bob.core/random_api.h>
#include <bob.blitz/cppapi.h>
#include <boost/make_shared.hpp>

#include <bob.core/random.h>
#include <boost/random.hpp>

PyDoc_STRVAR(s_gamma_str, BOB_EXT_MODULE_PREFIX ".gamma");

PyDoc_STRVAR(s_gamma_doc,
"gamma(dtype, [alpha=1.]]) -> new gamma distribution\n\
\n\
Models a random gamma distribution\n\
\n\
This distribution class models a gamma random distribution.\n\
Such a distribution produces random numbers :math:`x` distributed\n\
with the probability density function\n\
:math:`p(x) = x^{\\alpha-1}\\frac{e^{-x}}{\\Gamma(\\alpha)}`,\n\
where the ``alpha`` (:math:`\\alpha`) is a parameter of the\n\
distribution.\n\
\n\
"
);

/* How to create a new PyBoostGammaObject */
static PyObject* PyBoostGamma_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBoostGammaObject* self = (PyBoostGammaObject*)type->tp_alloc(type, 0);
  self->type_num = NPY_NOTYPE;
  self->distro.reset();

  return reinterpret_cast<PyObject*>(self);
}

/* How to delete a PyBoostGammaObject */
static void PyBoostGamma_Delete (PyBoostGammaObject* o) {

  o->distro.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);

}

template <typename T>
boost::shared_ptr<void> make_gamma(PyObject* alpha) {
  T calpha = 1.;
  if (alpha) calpha = PyBlitzArrayCxx_AsCScalar<T>(alpha);
  return boost::make_shared<bob::core::random::gamma_distribution<T>>(calpha);
}

PyObject* PyBoostGamma_SimpleNew (int type_num, PyObject* alpha) {

  PyBoostGammaObject* retval = (PyBoostGammaObject*)PyBoostGamma_New(&PyBoostGamma_Type, 0, 0);

  if (!retval) return 0;

  retval->type_num = type_num;

  switch(type_num) {
    case NPY_FLOAT32:
      retval->distro = make_gamma<float>(alpha);
      break;
    case NPY_FLOAT64:
      retval->distro = make_gamma<double>(alpha);
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
int PyBoostGamma_Init(PyBoostGammaObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"dtype", "alpha", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* alpha = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O", kwlist, &PyBlitzArray_TypenumConverter, &self->type_num, &alpha)) return -1; ///< FAILURE

  switch(self->type_num) {
    case NPY_FLOAT32:
      self->distro = make_gamma<float>(alpha);
      break;
    case NPY_FLOAT64:
      self->distro = make_gamma<double>(alpha);
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

int PyBoostGamma_Check(PyObject* o) {
  if (!o) return 0;
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBoostGamma_Type));
}

int PyBoostGamma_Converter(PyObject* o, PyBoostGammaObject** a) {
  if (!PyBoostGamma_Check(o)) return 0;
  Py_INCREF(o);
  (*a) = reinterpret_cast<PyBoostGammaObject*>(o);
  return 1;
}

template <typename T> PyObject* get_alpha(PyBoostGammaObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<bob::core::random::gamma_distribution<T>>(self->distro)->alpha());
}

/**
 * Accesses the alpha value
 */
static PyObject* PyBoostGamma_GetAlpha(PyBoostGammaObject* self) {
  switch (self->type_num) {
    case NPY_FLOAT32:
      return get_alpha<float>(self);
    case NPY_FLOAT64:
      return get_alpha<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get alpha parameter of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
}

/**
 * Accesses the datatype
 */
static PyObject* PyBoostGamma_GetDtype(PyBoostGammaObject* self) {
  return reinterpret_cast<PyObject*>(PyArray_DescrFromType(self->type_num));
}

template <typename T> PyObject* reset(PyBoostGammaObject* self) {
  boost::static_pointer_cast<bob::core::random::gamma_distribution<T>>(self->distro)->reset();
  Py_RETURN_NONE;
}

/**
 * Resets the distribution - this is a noop for gamma distributions, here
 * only for compatibility reasons
 */
static PyObject* PyBoostGamma_Reset(PyBoostGammaObject* self) {
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

template <typename T> PyObject* call(PyBoostGammaObject* self, PyBoostMt19937Object* rng) {
  typedef bob::core::random::gamma_distribution<T> distro_t;
  return PyBlitzArrayCxx_FromCScalar((*boost::static_pointer_cast<distro_t>(self->distro))(*rng->rng));
}

/**
 * Calling a PyBoostGammaObject to generate a random number
 */
static
PyObject* PyBoostGamma_Call(PyBoostGammaObject* self, PyObject *args, PyObject* kwds) {

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

static PyMethodDef PyBoostGamma_methods[] = {
    {
      s_reset_str,
      (PyCFunction)PyBoostGamma_Reset,
      METH_NOARGS,
      s_reset_doc,
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(s_dtype_str, "dtype");
PyDoc_STRVAR(s_dtype_doc,
"x.dtype -> numpy dtype\n\
\n\
The type of scalars produced by this gamma distribution.\n\
"
);

PyDoc_STRVAR(s_alpha_str, "alpha");
PyDoc_STRVAR(s_alpha_doc,
"x.alpha -> scalar\n\
\n\
This value corresponds to the alpha parameter the\n\
distribution current has.\n\
"
);

static PyGetSetDef PyBoostGamma_getseters[] = {
    {
      s_dtype_str,
      (getter)PyBoostGamma_GetDtype,
      0,
      s_dtype_doc,
      0,
    },
    {
      s_alpha_str,
      (getter)PyBoostGamma_GetAlpha,
      0,
      s_alpha_doc,
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
static PyObject* PyBoostGamma_Repr(PyBoostGammaObject* self) {

  PyObject* alpha = PyBoostGamma_GetAlpha(self);
  if (!alpha) return 0;

  PyObject* salpha = scalar_to_bytes(alpha);
  if (!salpha) return 0;

  PyObject* retval =
# if PY_VERSION_HEX >= 0x03000000
    PyUnicode_FromFormat
#else
    PyString_FromFormat
#endif
      (
       "%s(dtype='%s', alpha=%s)",
       Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(self->type_num),
       bytes_to_charp(salpha)
       );

  Py_DECREF(salpha);

  return retval;

}

PyTypeObject PyBoostGamma_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    s_gamma_str,                                /*tp_name*/
    sizeof(PyBoostGammaObject),                 /*tp_basicsize*/
    0,                                          /*tp_itemsize*/
    (destructor)PyBoostGamma_Delete,            /*tp_dealloc*/
    0,                                          /*tp_print*/
    0,                                          /*tp_getattr*/
    0,                                          /*tp_setattr*/
    0,                                          /*tp_compare*/
    (reprfunc)PyBoostGamma_Repr,                /*tp_repr*/
    0,                                          /*tp_as_number*/
    0,                                          /*tp_as_sequence*/
    0,                                          /*tp_as_mapping*/
    0,                                          /*tp_hash */
    (ternaryfunc)PyBoostGamma_Call,             /*tp_call*/
    (reprfunc)PyBoostGamma_Repr,                /*tp_str*/
    0,                                          /*tp_getattro*/
    0,                                          /*tp_setattro*/
    0,                                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /*tp_flags*/
    s_gamma_doc,                                /* tp_doc */
    0,		                                      /* tp_traverse */
    0,		                                      /* tp_clear */
    0,                                          /* tp_richcompare */
    0,		                                      /* tp_weaklistoffset */
    0,		                                      /* tp_iter */
    0,		                                      /* tp_iternext */
    PyBoostGamma_methods,                       /* tp_methods */
    0,                                          /* tp_members */
    PyBoostGamma_getseters,                     /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PyBoostGamma_Init,                /* tp_init */
    0,                                          /* tp_alloc */
    PyBoostGamma_New,                           /* tp_new */
};
