/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 27 Oct 09:02:32 2013
 *
 * @brief Discrete distributions (with integers or floating point numbers)
 */

#define BOB_CORE_RANDOM_MODULE
#include <bob.core/random_api.h>
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <boost/make_shared.hpp>
#include <boost/version.hpp>

#include <bob.core/random.h>

PyDoc_STRVAR(s_discrete_str, BOB_EXT_MODULE_PREFIX ".discrete");

/* How to create a new PyBoostDiscreteObject */
static PyObject* PyBoostDiscrete_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBoostDiscreteObject* self = (PyBoostDiscreteObject*)type->tp_alloc(type, 0);
  self->type_num = NPY_NOTYPE;
  self->distro.reset();

  return reinterpret_cast<PyObject*>(self);
}

/* How to delete a PyBoostDiscreteObject */
static void PyBoostDiscrete_Delete (PyBoostDiscreteObject* o) {

  o->distro.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);

}

template <typename T>
boost::shared_ptr<void> make_discrete(PyObject* probabilities) {

  std::vector<double> cxx_probabilities;

  PyObject* iterator = PyObject_GetIter(probabilities);
  if (!iterator) return boost::shared_ptr<void>();
  auto iterator_ = make_safe(iterator);

  while (PyObject* item = PyIter_Next(iterator)) {
    auto item_ = make_safe(item);
    double v = PyFloat_AsDouble(item);
    if (PyErr_Occurred()) return boost::shared_ptr<void>();
    cxx_probabilities.push_back(v);
  }


  return boost::make_shared<bob::core::random::discrete_distribution<T,double>>(cxx_probabilities);
}

PyObject* PyBoostDiscrete_SimpleNew (int type_num, PyObject* probabilities) {

  PyBoostDiscreteObject* retval = (PyBoostDiscreteObject*)PyBoostDiscrete_New(&PyBoostDiscrete_Type, 0, 0);

  if (!retval) return 0;

  retval->type_num = type_num;

  switch(type_num) {
    case NPY_UINT8:
      retval->distro = make_discrete<uint8_t>(probabilities);
      break;
    case NPY_UINT16:
      retval->distro = make_discrete<uint16_t>(probabilities);
      break;
    case NPY_UINT32:
      retval->distro = make_discrete<uint32_t>(probabilities);
      break;
    case NPY_UINT64:
      retval->distro = make_discrete<uint64_t>(probabilities);
      break;
    case NPY_INT8:
      retval->distro = make_discrete<int8_t>(probabilities);
      break;
    case NPY_INT16:
      retval->distro = make_discrete<int16_t>(probabilities);
      break;
    case NPY_INT32:
      retval->distro = make_discrete<int32_t>(probabilities);
      break;
    case NPY_INT64:
      retval->distro = make_discrete<int64_t>(probabilities);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot create %s(T) with T having an unsupported numpy type number of %d", Py_TYPE(retval)->tp_name, retval->type_num);
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
int PyBoostDiscrete_Init(PyBoostDiscreteObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"dtype", "probabilities", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* probabilities = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O", kwlist, &PyBlitzArray_TypenumConverter, &self->type_num, &probabilities)) return -1; ///< FAILURE

  switch(self->type_num) {
    case NPY_UINT8:
      self->distro = make_discrete<uint8_t>(probabilities);
      break;
    case NPY_UINT16:
      self->distro = make_discrete<uint16_t>(probabilities);
      break;
    case NPY_UINT32:
      self->distro = make_discrete<uint32_t>(probabilities);
      break;
    case NPY_UINT64:
      self->distro = make_discrete<uint64_t>(probabilities);
      break;
    case NPY_INT8:
      self->distro = make_discrete<int8_t>(probabilities);
      break;
    case NPY_INT16:
      self->distro = make_discrete<int16_t>(probabilities);
      break;
    case NPY_INT32:
      self->distro = make_discrete<int32_t>(probabilities);
      break;
    case NPY_INT64:
      self->distro = make_discrete<int64_t>(probabilities);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot create %s(T) with T having an unsupported numpy type number of %d", Py_TYPE(self)->tp_name, self->type_num);
      return -1;
  }

  if (!self->distro) { // a problem occurred
    return -1;
  }

  return 0; ///< SUCCESS
}

int PyBoostDiscrete_Check(PyObject* o) {
  if (!o) return 0;
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBoostDiscrete_Type));
}

int PyBoostDiscrete_Converter(PyObject* o, PyBoostDiscreteObject** a) {
  if (!PyBoostDiscrete_Check(o)) return 0;
  Py_INCREF(o);
  (*a) = reinterpret_cast<PyBoostDiscreteObject*>(o);
  return 1;
}

template <typename T>
PyObject* get_probabilities(PyBoostDiscreteObject* self) {
  std::vector<double> w = boost::static_pointer_cast<bob::core::random::discrete_distribution<T,double>>(self->distro)->probabilities();
  PyObject* retval = PyTuple_New(w.size());
  if (!retval) return 0;
  for (size_t k=0; k<w.size(); ++k) {
    PyTuple_SET_ITEM(retval, k, Py_BuildValue("d", w[k]));
  }
  return retval;
}

/**
 * Accesses the mean value
 */
static PyObject* PyBoostDiscrete_GetProbabilities(PyBoostDiscreteObject* self) {
  switch (self->type_num) {
    case NPY_UINT8:
      return get_probabilities<uint8_t>(self);
    case NPY_UINT16:
      return get_probabilities<uint16_t>(self);
    case NPY_UINT32:
      return get_probabilities<uint32_t>(self);
    case NPY_UINT64:
      return get_probabilities<uint64_t>(self);
    case NPY_INT8:
      return get_probabilities<int8_t>(self);
    case NPY_INT16:
      return get_probabilities<int16_t>(self);
    case NPY_INT32:
      return get_probabilities<int32_t>(self);
    case NPY_INT64:
      return get_probabilities<int64_t>(self);
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot get minimum of %s(T) with T having an unsupported numpy type number of %d (DEBUG ME)", Py_TYPE(self)->tp_name, self->type_num);
      return 0;
  }
}

/**
 * Accesses the datatype
 */
static PyObject* PyBoostDiscrete_GetDtype(PyBoostDiscreteObject* self) {
  return reinterpret_cast<PyObject*>(PyArray_DescrFromType(self->type_num));
}

template <typename T> PyObject* reset(PyBoostDiscreteObject* self) {
  boost::static_pointer_cast<bob::core::random::discrete_distribution<T,double>>(self->distro)->reset();
  Py_RETURN_NONE;
}

/**
 * Resets the distribution - this is a noop for discrete distributions, here
 * only for compatibility reasons
 */
static PyObject* PyBoostDiscrete_Reset(PyBoostDiscreteObject* self) {
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

template <typename T> PyObject* call(PyBoostDiscreteObject* self, PyBoostMt19937Object* rng) {
  return PyBlitzArrayCxx_FromCScalar(boost::static_pointer_cast<bob::core::random::discrete_distribution<T,double>>(self->distro)->operator()(*rng->rng));
}

/**
 * Calling a PyBoostDiscreteObject to generate a random number
 */
static
PyObject* PyBoostDiscrete_Call(PyBoostDiscreteObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"rng", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBoostMt19937Object* rng = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist, &PyBoostMt19937_Type, &rng)) return 0; ///< FAILURE

  switch(self->type_num) {
    case NPY_UINT8:
      return call<uint8_t>(self, rng);
      break;
    case NPY_UINT16:
      return call<uint16_t>(self, rng);
      break;
    case NPY_UINT32:
      return call<uint32_t>(self, rng);
      break;
    case NPY_UINT64:
      return call<uint64_t>(self, rng);
      break;
    case NPY_INT8:
      return call<int8_t>(self, rng);
      break;
    case NPY_INT16:
      return call<int16_t>(self, rng);
      break;
    case NPY_INT32:
      return call<int32_t>(self, rng);
      break;
    case NPY_INT64:
      return call<int64_t>(self, rng);
      break;
    default:
      PyErr_Format(PyExc_NotImplementedError, "cannot call %s(T) with T having an unsupported numpy type number of %d", Py_TYPE(self)->tp_name, self->type_num);
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

static PyMethodDef PyBoostDiscrete_methods[] = {
    {
      s_reset_str,
      (PyCFunction)PyBoostDiscrete_Reset,
      METH_NOARGS,
      s_reset_doc,
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(s_dtype_str, "dtype");
PyDoc_STRVAR(s_dtype_doc,
"x.dtype -> numpy dtype\n\
\n\
The type of scalars produced by this discrete distribution.\n\
"
);

PyDoc_STRVAR(s_probabilities_str, "probabilities");
PyDoc_STRVAR(s_probabilities_doc,
"x.probabilities -> sequence\n\
\n\
This property corresponds to the values you have set for the\n\
discrete probabilities of every entry in this distribution.\n\
"
);

static PyGetSetDef PyBoostDiscrete_getseters[] = {
    {
      s_dtype_str,
      (getter)PyBoostDiscrete_GetDtype,
      0,
      s_dtype_doc,
      0,
    },
    {
      s_probabilities_str,
      (getter)PyBoostDiscrete_GetProbabilities,
      0,
      s_probabilities_doc,
      0,
    },
    {0}  /* Sentinel */
};

#if PY_VERSION_HEX >= 0x03000000
#  define PYOBJECT_STR PyObject_Str
#else
#  define PYOBJECT_STR PyObject_Unicode
#endif

/**
 * String representation and print out
 */
static PyObject* PyBoostDiscrete_Repr(PyBoostDiscreteObject* self) {

  PyObject* probabilities = PyBoostDiscrete_GetProbabilities(self);
  if (!probabilities) return 0;
  PyObject* prob_str = PYOBJECT_STR(probabilities);
  Py_DECREF(probabilities);
  if (!prob_str) return 0;

  PyObject* retval = PyUnicode_FromFormat(
      "%s(dtype='%s' , probabilities=%U)",
      Py_TYPE(self)->tp_name,
      PyBlitzArray_TypenumAsString(self->type_num),
      prob_str
      );
  Py_DECREF(prob_str);

#if PYTHON_VERSION_HEX < 0x03000000
  if (!retval) return 0;
  PyObject* tmp = PyObject_Str(retval);
  Py_DECREF(retval);
  retval = tmp;
#endif

  return retval;

}

PyDoc_STRVAR(s_discrete_doc,
"discrete(dtype, probabilities) -> new discrete distribution\n\
\n\
Models a random discrete distribution\n\
\n\
A discrete distribution can only assume certain values, which\n\
for this class is defined as a number ``i`` in the range\n\
``[0, len(probabilities)]``. Notice that the condition\n\
:math:`\\sum(P) = 1`, with ``P = probabilities``, is\n\
enforced by normalizing the input values so that the sum\n\
over all probabilities always equals 1.\n\
\n\
"
);

PyTypeObject PyBoostDiscrete_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    s_discrete_str,                             /*tp_name*/
    sizeof(PyBoostDiscreteObject),              /*tp_basicsize*/
    0,                                          /*tp_itemsize*/
    (destructor)PyBoostDiscrete_Delete,         /*tp_dealloc*/
    0,                                          /*tp_print*/
    0,                                          /*tp_getattr*/
    0,                                          /*tp_setattr*/
    0,                                          /*tp_compare*/
    (reprfunc)PyBoostDiscrete_Repr,             /*tp_repr*/
    0,                                          /*tp_as_number*/
    0,                                          /*tp_as_sequence*/
    0,                                          /*tp_as_mapping*/
    0,                                          /*tp_hash */
    (ternaryfunc)PyBoostDiscrete_Call,          /*tp_call*/
    (reprfunc)PyBoostDiscrete_Repr,             /*tp_str*/
    0,                                          /*tp_getattro*/
    0,                                          /*tp_setattro*/
    0,                                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /*tp_flags*/
    s_discrete_doc,                             /* tp_doc */
    0,		                                      /* tp_traverse */
    0,		                                      /* tp_clear */
    0,                                          /* tp_richcompare */
    0,		                                      /* tp_weaklistoffset */
    0,		                                      /* tp_iter */
    0,		                                      /* tp_iternext */
    PyBoostDiscrete_methods,                    /* tp_methods */
    0,                                          /* tp_members */
    PyBoostDiscrete_getseters,                  /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PyBoostDiscrete_Init,             /* tp_init */
    0,                                          /* tp_alloc */
    PyBoostDiscrete_New,                        /* tp_new */
};
