/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 27 Oct 09:02:32 2013
 *
 * @brief LogNormal distributions (with floating point numbers)
 */

#define XBOB_CORE_RANDOM_MODULE
#include <xbob.core/random.h>
#include <blitz.array/cppapi.h>
#include <boost/make_shared.hpp>

#define NORMAL_NAME lognormal
PyDoc_STRVAR(s_lognormal_str, BOOST_PP_STRINGIZE(XBOB_CORE_RANDOM_MODULE_PREFIX) "." BOOST_PP_STRINGIZE(NORMAL_NAME));

/* How to create a new PyBoostLogNormalObject */
static PyObject* PyBoostLogNormal_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBoostLogNormalObject* self = (PyBoostLogNormalObject*)type->tp_alloc(type, 0);
  self->type_num = NPY_NOTYPE;
  self->distro.reset();

  return reinterpret_cast<PyObject*>(self);
}

/* How to delete a PyBoostLogNormalObject */
static void PyBoostLogNormal_Delete (PyBoostLogNormalObject* o) {

  o->distro.reset();
  o->ob_type->tp_free((PyObject*)o);

}

template <typename T>
boost::shared_ptr<void> make_lognormal(PyObject* mean, PyObject* sigma) {
  T cmean = 0.;
  if (mean) cmean = PyBlitzArrayCxx_AsCScalar<T>(mean);
  T csigma = 1.;
  if (sigma) csigma = PyBlitzArrayCxx_AsCScalar<T>(sigma);
  return boost::make_shared<boost::lognormal_distribution<T>>(cmean, csigma);
}

PyObject* PyBoostLogNormal_SimpleNew (int type_num, PyObject* mean, PyObject* sigma) {

  PyBoostLogNormalObject* retval = (PyBoostLogNormalObject*)PyBoostLogNormal_New(&PyBoostLogNormal_Type, 0, 0);

  if (!retval) return 0;

  retval->type_num = type_num;

  switch(type_num) {
    case NPY_FLOAT32:
      retval->distro = make_lognormal<float>(mean, sigma);
      break;
    case NPY_FLOAT64:
      retval->distro = make_lognormal<double>(mean, sigma);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot create %s(T) with T having an unsupported numpy type number of %d (it only supports numpy.float32 or numpy.float64)", s_lognormal_str, retval->type_num);
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
int PyBoostLogNormal_Init(PyBoostLogNormalObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"dtype", "mean", "sigma", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  int* type_num_p = &self->type_num;
  PyObject* m = 0;
  PyObject* s = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|OO", kwlist, &PyBlitzArray_TypenumConverter, &type_num_p, &m, &s)) return -1; ///< FAILURE

  switch(self->type_num) {
    case NPY_FLOAT32:
      self->distro = make_lognormal<float>(m, s);
      break;
    case NPY_FLOAT64:
      self->distro = make_lognormal<double>(m, s);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot create %s(T) with T having an unsupported numpy type number of %d (it only supports numpy.float32 or numpy.float64)", s_lognormal_str, self->type_num);
      return -1;
  }

  if (!self->distro) { // a problem occurred
    return -1;
  }

  return 0; ///< SUCCESS
}

int PyBoostLogNormal_Check(PyObject* o) {
  if (!o) return 0;
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBoostLogNormal_Type));
}

int PyBoostLogNormal_Converter(PyObject* o, PyBoostLogNormalObject** a) {
  if (!PyBoostLogNormal_Check(o)) return 0;
  Py_INCREF(o);
  (*a) = reinterpret_cast<PyBoostLogNormalObject*>(o);
  return 1;
}

template <typename T> PyObject* get_mean(PyBoostLogNormalObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<boost::lognormal_distribution<T>>(self->distro)->mean());
}

/**
 * Accesses the m value
 */
static PyObject* PyBoostLogNormal_GetMean(PyBoostLogNormalObject* self) {
  switch (self->type_num) {
    case NPY_FLOAT32:
      return get_mean<float>(self);
    case NPY_FLOAT64:
      return get_mean<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get m of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", s_lognormal_str, self->type_num);
      return 0;
  }
}

template <typename T> PyObject* get_sigma(PyBoostLogNormalObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<boost::lognormal_distribution<T>>(self->distro)->sigma());
}

/**
 * Accesses the s value
 */
static PyObject* PyBoostLogNormal_GetSigma(PyBoostLogNormalObject* self) {
  switch (self->type_num) {
    case NPY_FLOAT32:
      return get_sigma<float>(self);
    case NPY_FLOAT64:
      return get_sigma<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get s of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", s_lognormal_str, self->type_num);
      return 0;
  }
}

/**
 * Accesses the datatype
 */
static PyObject* PyBoostLogNormal_GetDtype(PyBoostLogNormalObject* self) {
  return reinterpret_cast<PyObject*>(PyArray_DescrFromType(self->type_num));
}

template <typename T> PyObject* reset(PyBoostLogNormalObject* self) {
  boost::static_pointer_cast<boost::lognormal_distribution<T>>(self->distro)->reset();
  Py_RETURN_NONE;
}

/**
 * Resets the distribution - this is a noop for lognormal distributions, here
 * only for compatibility reasons
 */
static PyObject* PyBoostLogNormal_Reset(PyBoostLogNormalObject* self) {
  switch (self->type_num) {
    case NPY_FLOAT32:
      return reset<float>(self);
    case NPY_FLOAT64:
      return reset<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot reset %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", s_lognormal_str, self->type_num);
      return 0;
  }
}

template <typename T> PyObject* call(PyBoostLogNormalObject* self, PyBoostMt19937Object* rng) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<boost::lognormal_distribution<T>>(self->distro)->operator()(*rng->rng));
}

/**
 * Calling a PyBoostLogNormalObject to generate a random number
 */
static 
PyObject* PyBoostLogNormal_Call(PyBoostLogNormalObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"rng", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBoostMt19937Object* rng = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist, &PyBoostMt19937_Converter, &rng)) return 0; ///< FAILURE

  switch(self->type_num) {
    case NPY_FLOAT32:
      return call<float>(self, rng);
      break;
    case NPY_FLOAT64:
      return call<double>(self, rng);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot call %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", s_lognormal_str, self->type_num);
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

static PyMethodDef PyBoostLogNormal_methods[] = {
    {
      s_reset_str,
      (PyCFunction)PyBoostLogNormal_Reset,
      METH_NOARGS,
      s_reset_doc,
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(s_dtype_str, "dtype");
PyDoc_STRVAR(s_dtype_doc, 
"x.dtype -> numpy dtype\n\
\n\
The type of scalars produced by this lognormal distribution.\n\
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

PyDoc_STRVAR(s_s_str, "sigma");
PyDoc_STRVAR(s_s_doc, 
"x.sigma -> scalar\n\
\n\
This value corresponds to the standard deviation value the\n\
distribution will have.\n\
"
);

static PyGetSetDef PyBoostLogNormal_getseters[] = {
    {
      s_dtype_str,
      (getter)PyBoostLogNormal_GetDtype,
      0,
      s_dtype_doc,
      0,
    },
    {
      s_mean_str,
      (getter)PyBoostLogNormal_GetMean,
      0,
      s_mean_doc,
      0,
    },
    {
      s_s_str,
      (getter)PyBoostLogNormal_GetSigma,
      0,
      s_s_doc,
      0,
    },
    {0}  /* Sentinel */
};

/**
 * String representation and print out
 */
static PyObject* PyBoostLogNormal_Repr(PyBoostLogNormalObject* self) {
  PyObject* mean = PyBoostLogNormal_GetMean(self);
  if (!mean) return 0;
  PyObject* sigma = PyBoostLogNormal_GetSigma(self);
  if (!sigma) return 0;
  PyObject* retval = PyUnicode_FromFormat("%s(dtype='%s', mean=%S, sigma=%S)",
      s_lognormal_str, PyBlitzArray_TypenumAsString(self->type_num), mean, sigma);
  Py_DECREF(mean);
  Py_DECREF(sigma);
  return retval;
}

PyDoc_STRVAR(s_lognormal_doc,
"lognormal(dtype, [mean=0., sigma=1.]]) -> new log-normal distribution\n\
\n\
Models a random lognormal distribution\n\
\n\
This distribution models a log-normal random distribution. Such a\n\
distribution produces random numbers ``x`` distributed with the\n\
probability density function\n\
:math:`p(x) = \\frac{1}{x \\sigma_N \\sqrt{2\\pi}} e^{\\frac{-\\left(\\log(x)-\\mu_N\\right)^2}{2\\sigma_N^2}}`, for :math:`x > 0` and :math:`\\sigma_N = \\sqrt{\\log\\left(1 + \\frac{\\sigma^2}{\\mu^2}\\right)}`,\n\
\n\
where the ``mean`` (:math:`\\mu`) and ``sigma`` (:math:`\\sigma`,\n\
the standard deviation) the parameters of the distribution.\n\
\n\
"
);

PyTypeObject PyBoostLogNormal_Type = {
    PyObject_HEAD_INIT(0)
    0,                                          /*ob_size*/
    s_lognormal_str,                            /*tp_name*/
    sizeof(PyBoostLogNormalObject),             /*tp_basicsize*/
    0,                                          /*tp_itemsize*/
    (destructor)PyBoostLogNormal_Delete,        /*tp_dealloc*/
    0,                                          /*tp_print*/
    0,                                          /*tp_getattr*/
    0,                                          /*tp_setattr*/
    0,                                          /*tp_compare*/
    (reprfunc)PyBoostLogNormal_Repr,            /*tp_repr*/
    0,                                          /*tp_as_number*/
    0,                                          /*tp_as_sequence*/
    0,                                          /*tp_as_mapping*/
    0,                                          /*tp_hash */
    (ternaryfunc)PyBoostLogNormal_Call,         /*tp_call*/
    (reprfunc)PyBoostLogNormal_Repr,            /*tp_str*/
    0,                                          /*tp_getattro*/
    0,                                          /*tp_setattro*/
    0,                                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /*tp_flags*/
    s_lognormal_doc,                            /* tp_doc */
    0,		                                      /* tp_traverse */
    0,		                                      /* tp_clear */
    0,                                          /* tp_richcompare */
    0,		                                      /* tp_weaklistoffset */
    0,		                                      /* tp_iter */
    0,		                                      /* tp_iternext */
    PyBoostLogNormal_methods,                   /* tp_methods */
    0,                                          /* tp_members */
    PyBoostLogNormal_getseters,                 /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PyBoostLogNormal_Init,            /* tp_init */
    0,                                          /* tp_alloc */
    PyBoostLogNormal_New,                       /* tp_new */
};