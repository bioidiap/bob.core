/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 27 Oct 09:02:32 2013
 *
 * @brief Gamma distributions (with integers or floating point numbers)
 */

#define XBOB_CORE_RANDOM_MODULE
#include <xbob.core/random.h>
#include <blitz.array/cppapi.h>
#include <boost/make_shared.hpp>

#define GAMMA_NAME gamma
PyDoc_STRVAR(s_gamma_str, BOOST_PP_STRINGIZE(XBOB_CORE_RANDOM_MODULE_PREFIX) "." BOOST_PP_STRINGIZE(GAMMA_NAME));

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
  o->ob_type->tp_free((PyObject*)o);

}

template <typename T>
boost::shared_ptr<void> make_gamma(PyObject* alpha, PyObject* beta) {
  T calpha = 1.;
  if (alpha) calpha = PyBlitzArrayCxx_AsCScalar<T>(alpha);
  T cbeta = 1.;
  if (beta) cbeta = PyBlitzArrayCxx_AsCScalar<T>(beta);
  return boost::make_shared<boost::gamma_distribution<T>>(calpha, cbeta);
}

PyObject* PyBoostGamma_SimpleNew (int type_num, PyObject* alpha, PyObject* beta) {

  PyBoostGammaObject* retval = (PyBoostGammaObject*)PyBoostGamma_New(&PyBoostGamma_Type, 0, 0);

  if (!retval) return 0;

  retval->type_num = type_num;

  switch(type_num) {
    case NPY_FLOAT32:
      retval->distro = make_gamma<float>(alpha, beta);
      break;
    case NPY_FLOAT64:
      retval->distro = make_gamma<double>(alpha, beta);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot create %s(T) with T having an unsupported numpy type number of %d (it only supports numpy.float32 or numpy.float64)", s_gamma_str, retval->type_num);
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
  static const char* const_kwlist[] = {"dtype", "alpha", "beta", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  int* type_num_p = &self->type_num;
  PyObject* alpha = 0;
  PyObject* beta = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|OO", kwlist, &PyBlitzArray_TypenumConverter, &type_num_p, &alpha, &beta)) return -1; ///< FAILURE

  switch(self->type_num) {
    case NPY_FLOAT32:
      self->distro = make_gamma<float>(alpha, beta);
      break;
    case NPY_FLOAT64:
      self->distro = make_gamma<double>(alpha, beta);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot create %s(T) with T having an unsupported numpy type number of %d (it only supports numpy.float32 or numpy.float64)", s_gamma_str, self->type_num);
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
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<boost::gamma_distribution<T>>(self->distro)->alpha());
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
      PyErr_Format(PyExc_NotImplementedError, "cannot get alpha parameter of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", s_gamma_str, self->type_num);
      return 0;
  }
}

template <typename T> PyObject* get_beta(PyBoostGammaObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<boost::gamma_distribution<T>>(self->distro)->beta());
}

/**
 * Accesses the beta value
 */
static PyObject* PyBoostGamma_GetBeta(PyBoostGammaObject* self) {
  switch (self->type_num) {
    case NPY_FLOAT32:
      return get_beta<float>(self);
    case NPY_FLOAT64:
      return get_beta<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get beta parameter of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", s_gamma_str, self->type_num);
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
  boost::static_pointer_cast<boost::gamma_distribution<T>>(self->distro)->reset();
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
      PyErr_Format(PyExc_NotImplementedError, "cannot reset %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", s_gamma_str, self->type_num);
      return 0;
  }
}

template <typename T> PyObject* call(PyBoostGammaObject* self, PyBoostMt19937Object* rng) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<boost::gamma_distribution<T>>(self->distro)->operator()(*rng->rng));
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

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist, &PyBoostMt19937_Converter, &rng)) return 0; ///< FAILURE

  switch(self->type_num) {
    case NPY_FLOAT32:
      return call<float>(self, rng);
      break;
    case NPY_FLOAT64:
      return call<double>(self, rng);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot call %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", s_gamma_str, self->type_num);
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

PyDoc_STRVAR(s_beta_str, "beta");
PyDoc_STRVAR(s_beta_doc, 
"x.beta -> scalar\n\
\n\
This value corresponds to the beta parameter the\n\
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
    {
      s_beta_str,
      (getter)PyBoostGamma_GetBeta,
      0,
      s_beta_doc,
      0,
    },
    {0}  /* Sentinel */
};

/**
 * String representation and print out
 */
static PyObject* PyBoostGamma_Repr(PyBoostGammaObject* self) {
  PyObject* alpha = PyBoostGamma_GetAlpha(self);
  if (!alpha) return 0;
  PyObject* beta = PyBoostGamma_GetBeta(self);
  if (!beta) return 0;
  PyObject* retval = PyUnicode_FromFormat("%s(dtype='%s', alpha=%S, beta=%S)",
      s_gamma_str, PyBlitzArray_TypenumAsString(self->type_num), alpha, beta);
  Py_DECREF(alpha);
  Py_DECREF(beta);
  return retval;
}

PyDoc_STRVAR(s_gamma_doc,
"gamma(dtype, [alpha=1., beta=1.]]) -> new gamma distribution\n\
\n\
Models a random gamma distribution\n\
\n\
This distribution class models a gamma random distribution.\n\
Such a distribution produces random numbers :math:`x` distributed\n\
with the probability density function\n\
:math:`p(x) = x^{\\alpha-1}\\frac{e^{-x}}{\\Gamma(\\alpha)}`,\n\
where the ``alpha`` (:math:`\\alpha`) and ``beta`` (:math:`\\beta`)\n\
are parameters of the distribution.\n\
\n\
"
);

PyTypeObject PyBoostGamma_Type = {
    PyObject_HEAD_INIT(0)
    0,                                          /*ob_size*/
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