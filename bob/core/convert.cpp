/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 16 Oct 17:40:24 2013
 *
 * @brief Pythonic bindings to C++ constructs on bob.core
 */

#include <bob.core/config.h>
#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>

#include <bob.core/array_convert.h>

template <typename Tdst, typename Tsrc, int N>
PyObject* inner_convert (PyBlitzArrayObject* src,
    PyObject* dst_min, PyObject* dst_max,
    PyObject* src_min, PyObject* src_max) {

  using bob::core::array::convert;
  using bob::core::array::convertFromRange;
  using bob::core::array::convertToRange;

  Tdst c_dst_min = dst_min ? PyBlitzArrayCxx_AsCScalar<Tdst>(dst_min) : 0;
  Tdst c_dst_max = dst_max ? PyBlitzArrayCxx_AsCScalar<Tdst>(dst_max) : 0;
  Tsrc c_src_min = src_min ? PyBlitzArrayCxx_AsCScalar<Tsrc>(src_min) : 0;
  Tsrc c_src_max = src_max ? PyBlitzArrayCxx_AsCScalar<Tsrc>(src_max) : 0;
  auto bz_src = PyBlitzArrayCxx_AsBlitz<Tsrc,N>(src);

  if (src_min) {

    if (dst_min) { //both src_range and dst_range are valid
      auto bz_dst = convert<Tdst,Tsrc>(*bz_src, c_dst_min, c_dst_max, c_src_min, c_src_max);
      return PyBlitzArrayCxx_NewFromArray(bz_dst);
    }

    //only src_range is valid
    auto bz_dst = convertFromRange<Tdst,Tsrc>(*bz_src, c_src_min, c_src_max);
    return PyBlitzArrayCxx_NewFromArray(bz_dst);
  }

  else if (dst_min) { //only dst_range is valid
    auto bz_dst = convertToRange<Tdst,Tsrc>(*bz_src, c_dst_min, c_dst_max);
    return PyBlitzArrayCxx_NewFromArray(bz_dst);
  }

  //use all defaults
  auto bz_dst = convert<Tdst,Tsrc>(*bz_src);
  return PyBlitzArrayCxx_NewFromArray(bz_dst);

}

template <typename Tdst, typename Tsrc>
PyObject* convert_dim (PyBlitzArrayObject* src,
    PyObject* dst_min, PyObject* dst_max,
    PyObject* src_min, PyObject* src_max) {

  PyObject* retval = 0;

  switch (src->ndim) {
    case 1:
      retval = inner_convert<Tdst, Tsrc, 1>(src, dst_min, dst_max, src_min, src_max);
      break;

    case 2:
      retval = inner_convert<Tdst, Tsrc, 2>(src, dst_min, dst_max, src_min, src_max);
      break;

    case 3:
      retval = inner_convert<Tdst, Tsrc, 3>(src, dst_min, dst_max, src_min, src_max);
      break;

    case 4:
      retval = inner_convert<Tdst, Tsrc, 4>(src, dst_min, dst_max, src_min, src_max);
      break;

    default:
      PyErr_Format(PyExc_TypeError, "conversion does not support %" PY_FORMAT_SIZE_T "d dimensions", src->ndim);

  }

  return retval;
}

template <typename T> PyObject* convert_to(PyBlitzArrayObject* src,
    PyObject* dst_min, PyObject* dst_max,
    PyObject* src_min, PyObject* src_max) {

  PyObject* retval = 0;

  switch (src->type_num) {
    case NPY_BOOL:
      retval = convert_dim<T, bool>(src, dst_min, dst_max, src_min, src_max);
      break;

    case NPY_INT8:
      retval = convert_dim<T, int8_t>(src, dst_min, dst_max, src_min, src_max);
      break;

    case NPY_INT16:
      retval = convert_dim<T, int16_t>(src, dst_min, dst_max, src_min, src_max);
      break;

    case NPY_INT32:
      retval = convert_dim<T, int32_t>(src, dst_min, dst_max, src_min, src_max);
      break;

    case NPY_INT64:
      retval = convert_dim<T, int64_t>(src, dst_min, dst_max, src_min, src_max);
      break;

    case NPY_UINT8:
      retval = convert_dim<T, uint8_t>(src, dst_min, dst_max, src_min, src_max);
      break;

    case NPY_UINT16:
      retval = convert_dim<T, uint16_t>(src, dst_min, dst_max, src_min, src_max);
      break;

    case NPY_UINT32:
      retval = convert_dim<T, uint32_t>(src, dst_min, dst_max, src_min, src_max);
      break;

    case NPY_UINT64:
      retval = convert_dim<T, uint64_t>(src, dst_min, dst_max, src_min, src_max);
      break;

    case NPY_FLOAT32:
      retval = convert_dim<T, float>(src, dst_min, dst_max, src_min, src_max);
      break;

    case NPY_FLOAT64:
      retval = convert_dim<T, double>(src, dst_min, dst_max, src_min, src_max);
      break;

    default:
      PyErr_Format(PyExc_TypeError, "conversion from `%s' (%d) is not supported", PyBlitzArray_TypenumAsString(src->type_num), src->type_num);

  }

  return retval;

}

static PyObject* py_convert(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "src",
    "dtype",
    "dest_range",
    "source_range",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* src = 0;
  int type_num = NPY_NOTYPE;
  PyObject* dst_min = 0;
  PyObject* dst_max = 0;
  PyObject* src_min = 0;
  PyObject* src_max = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&|(OO)(OO)",
        kwlist,
        &PyBlitzArray_Converter, &src,
        &PyBlitzArray_TypenumConverter, &type_num,
        &dst_min, &dst_max,
        &src_min, &src_max
        )) return 0;
  auto src_ = make_safe(src);

  PyObject* retval = 0;

  switch (type_num) {
    case NPY_UINT8:
      retval = convert_to<uint8_t>(src, dst_min, dst_max, src_min, src_max);
      break;

    case NPY_UINT16:
      retval = convert_to<uint16_t>(src, dst_min, dst_max, src_min, src_max);
      break;

    case NPY_FLOAT64:
      retval = convert_to<double>(src, dst_min, dst_max, src_min, src_max);
      break;

    default:
      PyErr_Format(PyExc_TypeError, "conversion to `%s' (%d) is not supported", PyBlitzArray_TypenumAsString(type_num), type_num);

  }

  if (!retval) return 0;

  return PyBlitzArray_NUMPY_WRAP(retval);
}

PyDoc_STRVAR(s_convert_str, "convert");
PyDoc_STRVAR(s_convert__doc__,
"convert(array, dtype, [dst_range, [src_range]]) -> array\n\
\n\
Converts array data type, with optional range squash/expansion.\n\
\n\
Function which allows to convert/rescale a array of a given type into\n\
another array of a possibly different type with re-scaling. Typically,\n\
this can be used to rescale a 16 bit precision grayscale image (2D\n\
array) into an 8 bit precision grayscale image.\n\
\n\
Keyword Parameters:\n\
\n\
  array\n\
    (array) Input array\n\
  \n\
  dtype\n\
    (object) Any object that can be convertible to a\n\
    :py:class:`numpy.dtype`. Controls the output element type for the\n\
    returned array.\n\
  \n\
  dest_range\n\
    (tuple) Determines the range to be deployed at the returned array.\n\
  \n\
  source_range\n\
    (tuple) Determines the input range that will be used for scaling.\n\
  \n\
Returns a new array with the same shape as this one, but re-scaled and\n\
with its element type as indicated by the user.\n\
"
);

static PyMethodDef module_methods[] = {
    {
      s_convert_str,
      (PyCFunction)py_convert,
      METH_VARARGS|METH_KEYWORDS,
      s_convert__doc__
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "bob::core::array::convert bindings");

#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  BOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  module_methods,
  0, 0, 0, 0
};
#endif

static PyObject* create_module (void) {

# if PY_VERSION_HEX >= 0x03000000
  PyObject* m = PyModule_Create(&module_definition);
# else
  PyObject* m = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
# endif
  if (!m) return 0;
  auto m_ = make_safe(m);

  /* imports dependencies */
  if (import_bob_blitz() < 0) return 0;

  return Py_BuildValue("O", m);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
