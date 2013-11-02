/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 27 Oct 09:02:32 2013
 *
 * @brief Uniform distributions (with integers or floating point numbers)
 */

#define XBOB_CORE_RANDOM_MODULE
#include <xbob.core/random.h>
#include <blitz.array/cppapi.h>
#include <boost/make_shared.hpp>

#define UNIFORM_NAME uniform
PyDoc_STRVAR(s_uniform_str, BOOST_PP_STRINGIZE(XBOB_CORE_RANDOM_MODULE_NAME) "." BOOST_PP_STRINGIZE(UNIFORM_NAME));

/* How to create a new PyBoostUniformObject */
static PyObject* PyBoostUniform_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBoostUniformObject* self = (PyBoostUniformObject*)type->tp_alloc(type, 0);
  self->distro.reset();

  return reinterpret_cast<PyObject*>(self);
}

/* How to delete a PyBoostUniformObject */
static void PyBoostUniform_Delete (PyBoostUniformObject* o) {

  o->distro.reset();
  o->ob_type->tp_free((PyObject*)o);

}

template <typename T>
boost::shared_ptr<void> make_uniform_int(PyObject* min, PyObject* max) {
  T cmin = 0;
  if (min) cmin = PyBlitzArrayCxx_AsCScalar<T>(min);
  T cmax = 10;
  if (max) cmax = PyBlitzArrayCxx_AsCScalar<T>(max);
  return boost::make_shared<boost::uniform_int<T>>(cmin, cmax);
}

template <typename T>
boost::shared_ptr<void> make_uniform_real(PyObject* min, PyObject* max) {
  T cmin = 0;
  if (min) cmin = PyBlitzArrayCxx_AsCScalar<T>(min);
  T cmax = 1;
  if (max) cmax = PyBlitzArrayCxx_AsCScalar<T>(max);
  return boost::make_shared<boost::uniform_real<T>>(cmin, cmax);
}

PyObject* PyBoostUniform_SimpleNew (int type_num, PyObject* min, PyObject* max) {

  PyBoostUniformObject* retval = (PyBoostUniformObject*)PyBoostUniform_New(&PyBoostUniform_Type, 0, 0);

  retval->type_num = type_num;

  switch(type_num) {

    case NPY_UINT8:
      retval->distro = make_uniform_int<uint8_t>(min, max);
      break;
    case NPY_UINT16:
      retval->distro = make_uniform_int<uint16_t>(min, max);
      break;
    case NPY_UINT32:
      retval->distro = make_uniform_int<uint32_t>(min, max);
      break;
    case NPY_UINT64:
      retval->distro = make_uniform_int<uint64_t>(min, max);
      break;
    case NPY_INT8:
      retval->distro = make_uniform_int<int8_t>(min, max);
      break;
    case NPY_INT16:
      retval->distro = make_uniform_int<int16_t>(min, max);
      break;
    case NPY_INT32:
      retval->distro = make_uniform_int<int32_t>(min, max);
      break;
    case NPY_INT64:
      retval->distro = make_uniform_int<int64_t>(min, max);
      break;
    case NPY_FLOAT32:
      retval->distro = make_uniform_real<float>(min, max);
      break;
    case NPY_FLOAT64:
      retval->distro = make_uniform_real<double>(min, max);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot create %s(T) with T having an unsupported numpy type number of %d", s_uniform_str, retval->type_num);
      return 0;
  }

  return reinterpret_cast<PyObject*>(retval);

}

/* Implements the __init__(self) function */
static 
int PyBoostUniform_Init(PyBoostUniformObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"dtype", "min", "max", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  int* type_num_p = &self->type_num;
  PyObject* min = 0;
  PyObject* max = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|OO", kwlist, &PyBlitzArray_TypenumConverter, &type_num_p, &min, &max)) return -1; ///< FAILURE

  switch(self->type_num) {
    case NPY_UINT8:
      self->distro = make_uniform_int<uint8_t>(min, max);
      break;
    case NPY_UINT16:
      self->distro = make_uniform_int<uint16_t>(min, max);
      break;
    case NPY_UINT32:
      self->distro = make_uniform_int<uint32_t>(min, max);
      break;
    case NPY_UINT64:
      self->distro = make_uniform_int<uint64_t>(min, max);
      break;
    case NPY_INT8:
      self->distro = make_uniform_int<int8_t>(min, max);
      break;
    case NPY_INT16:
      self->distro = make_uniform_int<int16_t>(min, max);
      break;
    case NPY_INT32:
      self->distro = make_uniform_int<int32_t>(min, max);
      break;
    case NPY_INT64:
      self->distro = make_uniform_int<int64_t>(min, max);
      break;
    case NPY_FLOAT32:
      self->distro = make_uniform_real<float>(min, max);
      break;
    case NPY_FLOAT64:
      self->distro = make_uniform_real<double>(min, max);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot create %s(T) with T having an unsupported numpy type number of %d", s_uniform_str, self->type_num);
      return -1;
  }

  return 0; ///< SUCCESS
}

int PyBoostUniform_Check(PyObject* o) {
  if (!o) return 0;
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBoostUniform_Type));
}

int PyBoostUniform_Converter(PyObject* o, PyBoostUniformObject** a) {
  if (!PyBoostUniform_Check(o)) return 0;
  Py_INCREF(o);
  (*a) = reinterpret_cast<PyBoostUniformObject*>(o);
  return 1;
}

template <typename T> PyObject* get_minimum_int(PyBoostUniformObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<boost::uniform_int<T>>(self->distro)->min());
}

template <typename T> PyObject* get_minimum_real(PyBoostUniformObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<boost::uniform_real<T>>(self->distro)->min());
}

/**
 * Accesses the min value
 */
static PyObject* PyBoostUniform_GetMin(PyBoostUniformObject* self) {
  switch (self->type_num) {
    case NPY_UINT8:
      return get_minimum_int<uint8_t>(self);
    case NPY_UINT16:
      return get_minimum_int<uint16_t>(self);
    case NPY_UINT32:
      return get_minimum_int<uint32_t>(self);
    case NPY_UINT64:
      return get_minimum_int<uint64_t>(self);
    case NPY_INT8:
      return get_minimum_int<int8_t>(self);
    case NPY_INT16:
      return get_minimum_int<int16_t>(self);
    case NPY_INT32:
      return get_minimum_int<int32_t>(self);
    case NPY_INT64:
      return get_minimum_int<int64_t>(self);
    case NPY_FLOAT32:
      return get_minimum_real<float>(self);
    case NPY_FLOAT64:
      return get_minimum_real<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get minimum of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", s_uniform_str, self->type_num);
      return 0;
  }
}

template <typename T> PyObject* get_maximum_int(PyBoostUniformObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<boost::uniform_int<T>>(self->distro)->max());
}

template <typename T> PyObject* get_maximum_real(PyBoostUniformObject* self) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<boost::uniform_real<T>>(self->distro)->max());
}

/**
 * Accesses the max value
 */
static PyObject* PyBoostUniform_GetMax(PyBoostUniformObject* self) {
  switch (self->type_num) {
    case NPY_UINT8:
      return get_maximum_int<uint8_t>(self);
    case NPY_UINT16:
      return get_maximum_int<uint16_t>(self);
    case NPY_UINT32:
      return get_maximum_int<uint32_t>(self);
    case NPY_UINT64:
      return get_maximum_int<uint64_t>(self);
    case NPY_INT8:
      return get_maximum_int<int8_t>(self);
    case NPY_INT16:
      return get_maximum_int<int16_t>(self);
    case NPY_INT32:
      return get_maximum_int<int32_t>(self);
    case NPY_INT64:
      return get_maximum_int<int64_t>(self);
    case NPY_FLOAT32:
      return get_maximum_real<float>(self);
    case NPY_FLOAT64:
      return get_maximum_real<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get maximum of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", s_uniform_str, self->type_num);
      return 0;
  }
}

template <typename T> PyObject* reset_int(PyBoostUniformObject* self) {
  boost::static_pointer_cast<boost::uniform_int<T>>(self->distro)->reset();
  Py_RETURN_NONE;
}

template <typename T> PyObject* reset_real(PyBoostUniformObject* self) {
  boost::static_pointer_cast<boost::uniform_real<T>>(self->distro)->reset();
  Py_RETURN_NONE;
}

/**
 * Resets the distribution - this is a noop for uniform distributions, here
 * only for compatibility reasons
 */
static PyObject* PyBoostUniform_Reset(PyBoostUniformObject* self) {
  switch (self->type_num) {
    case NPY_UINT8:
      return reset_int<uint8_t>(self);
    case NPY_UINT16:
      return reset_int<uint16_t>(self);
    case NPY_UINT32:
      return reset_int<uint32_t>(self);
    case NPY_UINT64:
      return reset_int<uint64_t>(self);
    case NPY_INT8:
      return reset_int<int8_t>(self);
    case NPY_INT16:
      return reset_int<int16_t>(self);
    case NPY_INT32:
      return reset_int<int32_t>(self);
    case NPY_INT64:
      return reset_int<int64_t>(self);
    case NPY_FLOAT32:
      return reset_real<float>(self);
    case NPY_FLOAT64:
      return reset_real<double>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot reset %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", s_uniform_str, self->type_num);
      return 0;
  }
}

template <typename T> PyObject* call_int(PyBoostUniformObject* self, PyBoostMt19937Object* rng) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<boost::uniform_int<T>>(self->distro)->operator()(*rng->rng));
}

template <typename T> PyObject* call_real(PyBoostUniformObject* self, PyBoostMt19937Object* rng) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<boost::uniform_real<T>>(self->distro)->operator()(*rng->rng));
}

/**
 * Calling a PyBoostUniformObject to generate a random number
 */
static 
PyObject* PyBoostUniform_Call(PyBoostUniformObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"rng", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBoostMt19937Object* rng = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist, &PyBoostMt19937_Converter, &rng)) return 0; ///< FAILURE

  switch(self->type_num) {
    case NPY_UINT8:
      return call_int<uint8_t>(self, rng);
      break;
    case NPY_UINT16:
      return call_int<uint16_t>(self, rng);
      break;
    case NPY_UINT32:
      return call_int<uint32_t>(self, rng);
      break;
    case NPY_UINT64:
      return call_int<uint64_t>(self, rng);
      break;
    case NPY_INT8:
      return call_int<int8_t>(self, rng);
      break;
    case NPY_INT16:
      return call_int<int16_t>(self, rng);
      break;
    case NPY_INT32:
      return call_int<int32_t>(self, rng);
      break;
    case NPY_INT64:
      return call_int<int64_t>(self, rng);
      break;
    case NPY_FLOAT32:
      return call_real<float>(self, rng);
      break;
    case NPY_FLOAT64:
      return call_real<double>(self, rng);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot call %s(T) with T having an unsupported numpy type number of %d", s_uniform_str, self->type_num);
  }

  return 0; ///< FAILURE
}

PyDoc_STRVAR(s_reset_str, "reset");
PyDoc_STRVAR(s_reset_doc, 
"x.reset() -> None\n\
\n\
After calling this method, subsequent uses of the distribution do not\n\
depend on values produced by any random number generator prior to\n\
invoking reset.\n\
"
);

static PyMethodDef PyBoostUniform_methods[] = {
    {
      s_reset_str,
      (PyCFunction)PyBoostUniform_Reset,
      METH_NOARGS,
      s_reset_doc,
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(s_min_str, "min");
PyDoc_STRVAR(s_min_doc, 
"x.min -> scalar\n\
\n\
This value corresponds to the smallest value that the distribution\n\
can produce.\n\
"
);

PyDoc_STRVAR(s_max_str, "max");
PyDoc_STRVAR(s_max_doc, 
"x.max -> scalar\n\
\n\
This value corresponds to the largest value that the distribution\n\
can produce. The uniform distribution is bound at [min, max[.\n\
Therefore, a distribution whose maximum value is ``max`` produces\n\
values which are at most, but excluding ``max``.\n\
"
);

static PyGetSetDef PyBoostUniform_getseters[] = {
    {
      s_min_str,
      (getter)PyBoostUniform_GetMin,
      0,
      s_min_doc,
      0,
    },
    {
      s_max_str,
      (getter)PyBoostUniform_GetMax,
      0,
      s_max_doc,
      0,
    },
    {0}  /* Sentinel */
};

/**
 * String representation and print out
 */
static PyObject* PyBoostUniform_Repr(PyBoostUniformObject* self) {
  PyObject* min = PyBoostUniform_GetMin(self);
  if (!min) return 0;
  PyObject* max = PyBoostUniform_GetMax(self);
  if (!max) return 0;
  PyObject* retval = PyUnicode_FromFormat("%s(dtype='%s', min=%S, max=%S)",
      s_uniform_str, PyBlitzArray_TypenumAsString(self->type_num), min, max);
  Py_DECREF(min);
  Py_DECREF(max);
  return retval;
}

PyDoc_STRVAR(s_uniform_doc,
"Models a random uniform distribution\n\
\n\
On each invocation, it returns a random value uniformly distributed\n\
in the set of numbers [min, max[.\n\
"
);

PyTypeObject PyBoostUniform_Type = {
    PyObject_HEAD_INIT(0)
    0,                                          /*ob_size*/
    s_uniform_str,                              /*tp_name*/
    sizeof(PyBoostUniformObject),               /*tp_basicsize*/
    0,                                          /*tp_itemsize*/
    (destructor)PyBoostUniform_Delete,          /*tp_dealloc*/
    0,                                          /*tp_print*/
    0,                                          /*tp_getattr*/
    0,                                          /*tp_setattr*/
    0,                                          /*tp_compare*/
    (reprfunc)PyBoostUniform_Repr,              /*tp_repr*/
    0,                                          /*tp_as_number*/
    0,                                          /*tp_as_sequence*/
    0,                                          /*tp_as_mapping*/
    0,                                          /*tp_hash */
    (ternaryfunc)PyBoostUniform_Call,           /*tp_call*/
    (reprfunc)PyBoostUniform_Repr,              /*tp_str*/
    0,                                          /*tp_getattro*/
    0,                                          /*tp_setattro*/
    0,                                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /*tp_flags*/
    s_uniform_doc,                              /* tp_doc */
    0,		                                      /* tp_traverse */
    0,		                                      /* tp_clear */
    0,                                          /* tp_richcompare */
    0,		                                      /* tp_weaklistoffset */
    0,		                                      /* tp_iter */
    0,		                                      /* tp_iternext */
    PyBoostUniform_methods,                     /* tp_methods */
    0,                                          /* tp_members */
    PyBoostUniform_getseters,                   /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PyBoostUniform_Init,              /* tp_init */
    0,                                          /* tp_alloc */
    PyBoostUniform_New,                         /* tp_new */
};
