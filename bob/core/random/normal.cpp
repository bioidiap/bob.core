/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 27 Oct 09:02:32 2013
 *
 * @brief Normal distributions (with integers or floating point numbers)
 */

#define BOB_CORE_RANDOM_MODULE
#include <bob.core/random_api.h>
#include <bob.blitz/cppapi.h>
#include <boost/make_shared.hpp>

#include <bob.core/random.h>

PyDoc_STRVAR(s_normal_str, BOB_EXT_MODULE_PREFIX ".normal");

/* How to create a new PyBoostNormalObject */
static PyObject* PyBoostNormal_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBoostNormalObject* self = (PyBoostNormalObject*)type->tp_alloc(type, 0);
  self->type_num = NPY_NOTYPE;
  self->distro.reset();

  return reinterpret_cast<PyObject*>(self);
}

/* How to delete a PyBoostNormalObject */
static void PyBoostNormal_Delete (PyBoostNormalObject* o) {

  o->distro.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);

}

template <typename T>
boost::shared_ptr<void> make_normal(PyObject* mean, PyObject* sigma) {
  T cmean = 0.;
  if (mean) cmean = PyBlitzArrayCxx_AsCScalar<T>(mean);
  T csigma = 1.;
  if (sigma) csigma = PyBlitzArrayCxx_AsCScalar<T>(sigma);
  return boost::make_shared<bob::core::random::normal_distribution<T>>(cmean, csigma);
}

PyObject* PyBoostNormal_SimpleNew (int type_num, PyObject* mean, PyObject* sigma) {

  PyBoostNormalObject* retval = (PyBoostNormalObject*)PyBoostNormal_New(&PyBoostNormal_Type, 0, 0);

  if (!retval) return 0;

  retval->type_num = type_num;

  switch(type_num) {
    case NPY_FLOAT32:
      retval->distro = make_normal<float>(mean, sigma);
      break;
    case NPY_FLOAT64:
      retval->distro = make_normal<double>(mean, sigma);
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
int PyBoostNormal_Init(PyBoostNormalObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"dtype", "mean", "sigma", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* mean = 0;
  PyObject* sigma = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|OO", kwlist, &PyBlitzArray_TypenumConverter, &self->type_num, &mean, &sigma)) return -1; ///< FAILURE

  switch(self->type_num) {
    case NPY_FLOAT32:
      self->distro = make_normal<float>(mean, sigma);
      break;
    case NPY_FLOAT64:
      self->distro = make_normal<double>(mean, sigma);
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

int PyBoostNormal_Check(PyObject* o) {
  if (!o) return 0;
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBoostNormal_Type));
}

int PyBoostNormal_Converter(PyObject* o, PyBoostNormalObject** a) {
  if (!PyBoostNormal_Check(o)) return 0;
  Py_INCREF(o);
  (*a) = reinterpret_cast<PyBoostNormalObject*>(o);
  return 1;
}

template <typename T> PyObject* get_mean(PyBoostNormalObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<bob::core::random::normal_distribution<T>>(self->distro)->mean());
}

/**
 * Accesses the mean value
 */
static PyObject* PyBoostNormal_GetMean(PyBoostNormalObject* self) {
  switch (self->type_num) {
    case NPY_FLOAT32:
      return get_mean<float>(self);
    case NPY_FLOAT64:
      return get_mean<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get mean of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
}

template <typename T> PyObject* get_sigma(PyBoostNormalObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<bob::core::random::normal_distribution<T>>(self->distro)->sigma());
}

/**
 * Accesses the sigma value
 */
static PyObject* PyBoostNormal_GetSigma(PyBoostNormalObject* self) {
  switch (self->type_num) {
    case NPY_FLOAT32:
      return get_sigma<float>(self);
    case NPY_FLOAT64:
      return get_sigma<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get sigma of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
}

/**
 * Accesses the datatype
 */
static PyObject* PyBoostNormal_GetDtype(PyBoostNormalObject* self) {
  return reinterpret_cast<PyObject*>(PyArray_DescrFromType(self->type_num));
}

template <typename T> PyObject* reset(PyBoostNormalObject* self) {
  boost::static_pointer_cast<bob::core::random::normal_distribution<T>>(self->distro)->reset();
  Py_RETURN_NONE;
}

/**
 * Resets the distribution - this is a noop for normal distributions, here
 * only for compatibility reasons
 */
static PyObject* PyBoostNormal_Reset(PyBoostNormalObject* self) {
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

template <typename T> PyObject* call(PyBoostNormalObject* self, PyBoostMt19937Object* rng) {
  typedef bob::core::random::normal_distribution<T> distro_t;
  return PyBlitzArrayCxx_FromCScalar((*boost::static_pointer_cast<distro_t>(self->distro))(*rng->rng));
}

/**
 * Calling a PyBoostNormalObject to generate a random number
 */
static
PyObject* PyBoostNormal_Call(PyBoostNormalObject* self, PyObject *args, PyObject* kwds) {

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

static PyMethodDef PyBoostNormal_methods[] = {
    {
      s_reset_str,
      (PyCFunction)PyBoostNormal_Reset,
      METH_NOARGS,
      s_reset_doc,
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(s_dtype_str, "dtype");
PyDoc_STRVAR(s_dtype_doc,
"x.dtype -> numpy dtype\n\
\n\
The type of scalars produced by this normal distribution.\n\
"
);

PyDoc_STRVAR(s_mean_str, "mean");
PyDoc_STRVAR(s_mean_doc,
"x.mean -> scalar\n\
\n\
This value corresponds to the mean value the distribution\n\
will produce.\n\
"
);

PyDoc_STRVAR(s_sigma_str, "sigma");
PyDoc_STRVAR(s_sigma_doc,
"x.sigma -> scalar\n\
\n\
This value corresponds to the standard deviation value the\n\
distribution will have.\n\
"
);

static PyGetSetDef PyBoostNormal_getseters[] = {
    {
      s_dtype_str,
      (getter)PyBoostNormal_GetDtype,
      0,
      s_dtype_doc,
      0,
    },
    {
      s_mean_str,
      (getter)PyBoostNormal_GetMean,
      0,
      s_mean_doc,
      0,
    },
    {
      s_sigma_str,
      (getter)PyBoostNormal_GetSigma,
      0,
      s_sigma_doc,
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
static PyObject* PyBoostNormal_Repr(PyBoostNormalObject* self) {

  PyObject* mean = PyBoostNormal_GetMean(self);
  if (!mean) return 0;
  PyObject* sigma = PyBoostNormal_GetSigma(self);
  if (!sigma) return 0;

  PyObject* smean = scalar_to_bytes(mean);
  if (!smean) return 0;
  PyObject* ssigma = scalar_to_bytes(sigma);
  if (!ssigma) return 0;

  PyObject* retval =
# if PY_VERSION_HEX >= 0x03000000
    PyUnicode_FromFormat
#else
    PyString_FromFormat
#endif
      (
       "%s(dtype='%s', mean=%s, sigma=%s)",
       Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(self->type_num),
       bytes_to_charp(smean), bytes_to_charp(ssigma)
      );

  Py_DECREF(smean);
  Py_DECREF(ssigma);

  return retval;

}

PyDoc_STRVAR(s_normal_doc,
"normal(dtype, [mean=0., sigma=1.]]) -> new normal distribution\n\
\n\
Models a random normal distribution\n\
\n\
This distribution class models a normal random distribution.\n\
Such a distribution produces random numbers :math:`x` distributed\n\
with the probability density function\n\
:math:`p(x) = \\frac{1}{\\sqrt{2\\pi\\sigma}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}`,\n\
where the ``mean`` (:math:`\\mu`) and ``sigma`` (:math:`\\sigma`,\n\
the standard deviation) the parameters of the distribution.\n\
\n\
"
);

PyTypeObject PyBoostNormal_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    s_normal_str,                               /*tp_name*/
    sizeof(PyBoostNormalObject),                /*tp_basicsize*/
    0,                                          /*tp_itemsize*/
    (destructor)PyBoostNormal_Delete,           /*tp_dealloc*/
    0,                                          /*tp_print*/
    0,                                          /*tp_getattr*/
    0,                                          /*tp_setattr*/
    0,                                          /*tp_compare*/
    (reprfunc)PyBoostNormal_Repr,               /*tp_repr*/
    0,                                          /*tp_as_number*/
    0,                                          /*tp_as_sequence*/
    0,                                          /*tp_as_mapping*/
    0,                                          /*tp_hash */
    (ternaryfunc)PyBoostNormal_Call,            /*tp_call*/
    (reprfunc)PyBoostNormal_Repr,               /*tp_str*/
    0,                                          /*tp_getattro*/
    0,                                          /*tp_setattro*/
    0,                                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /*tp_flags*/
    s_normal_doc,                               /* tp_doc */
    0,		                                      /* tp_traverse */
    0,		                                      /* tp_clear */
    0,                                          /* tp_richcompare */
    0,		                                      /* tp_weaklistoffset */
    0,		                                      /* tp_iter */
    0,		                                      /* tp_iternext */
    PyBoostNormal_methods,                      /* tp_methods */
    0,                                          /* tp_members */
    PyBoostNormal_getseters,                    /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PyBoostNormal_Init,               /* tp_init */
    0,                                          /* tp_alloc */
    PyBoostNormal_New,                          /* tp_new */
};
