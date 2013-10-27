/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 27 Oct 09:02:32 2013
 *
 * @brief Uniform distributions (with integers or floating point numbers)
 */

PyDoc_STRVAR(s_uniform_str, "uniform");

/* Type definition for PyUniformObject */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  int type_num;
  boost::shared_ptr<void> distro;

} PyUniformObject;

/* How to create a new PyUniformObject */
PyObject* PyUniform_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyUniformObject* self = (PyUniformObject*)type->tp_alloc(type, 0);
  self->distro.reset();

  return reinterpret_cast<PyObject*>(self);
}

/* How to delete a PyUniformObject */
void PyUniform_Delete (PyUniformObject* o) {

  o->distro.reset();
  o->ob_type->tp_free((PyObject*)o);

}

/**
 * Formal initialization of a Uniform object
 */
static 
int PyUniform_Init(PyUniformObject* self, PyObject *args, PyObject* kwds) {

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

  switch(self->type_num) {
    case NPY_UINT8:
      delete reinterpret_cast<boost::uniform_int<uint8_t>*>(self->distro);
      break;
    case NPY_UINT16:
      delete reinterpret_cast<boost::uniform_int<uint16_t>*>(self->distro);
      break;
    case NPY_UINT32:
      delete reinterpret_cast<boost::uniform_int<uint32_t>*>(self->distro);
      break;
    case NPY_UINT64:
      delete reinterpret_cast<boost::uniform_int<uint64_t>*>(self->distro);
      break;
    case NPY_INT8:
      delete reinterpret_cast<boost::uniform_int<int8_t>*>(self->distro);
      break;
    case NPY_INT16:
      delete reinterpret_cast<boost::uniform_int<int16_t>*>(self->distro);
      break;
    case NPY_INT32:
      delete reinterpret_cast<boost::uniform_int<int32_t>*>(self->distro);
      break;
    case NPY_INT64:
      delete reinterpret_cast<boost::uniform_int<int64_t>*>(self->distro);
      break;
    case NPY_FLOAT32:
      delete reinterpret_cast<boost::uniform_real<float>*>(self->distro);
      break;
    case NPY_FLOAT64:
      delete reinterpret_cast<boost::uniform_real<double>*>(self->distro);
      break;
  self->int_distro = 0;
  self->real_distro = 0;
  return 0; ///< SUCCESS
}

PyDoc_STRVAR(s_uniform__doc__,
"Uniform distribution within a range\n\
\n\
This distribution class models a uniform random distribution.\n\
On each invocation, it returns a random value uniformly distributed\n\
in the set of numbers [min, max[. You can create \n\
");

template <typename T, typename Engine>
static void uniform_int(const char* vartype) {
  typedef boost::uniform_int<T> D;

  boost::format name("uniform_%s");
  name % vartype;

  boost::format doc("Uniform distribution within a range (integer numbers).\n\nThe distribution class %s (boost::uniform_int<%s>) models a uniform random distribution. On each invocation, it returns a random integer value uniformly distributed in the set of integer numbers {min, min+1, min+2, ..., max}.");
  doc % name.str() % vartype;

  class_<D, boost::shared_ptr<D> >(name.str().c_str(), doc.str().c_str(), no_init)
    .def(init<optional<T, T> >((arg("min")=0, arg("max")=9), "Constructs a new object of this type, 'min' and 'max' are parameters of the distribution"))
    .add_property("min", &D::min, "The minimum for this distribution")
    .add_property("max", &D::max, "The maximum for this distribution")
    .def("reset", &D::reset, (arg("self")), "This is a noop for this distribution, here only for consistency")
    .def("__call__", __call__<D,Engine>, (arg("self"), arg("rng")))
    ;
}

template <typename T, typename Engine>
static void uniform_real(const char* vartype) {
  typedef boost::uniform_real<T> D;

  boost::format name("uniform_%s");
  name % vartype;

  boost::format doc("Uniform distribution within a range (floating-point numbers).\n\nThe distribution class %s (boost::uniform_real<%s>) models a uniform random distribution. On each invocation, it returns a random floating-point value uniformly distributed in the range [min..max). The value is computed using std::numeric_limits<RealType>::digits random binary digits, i.e. the mantissa of the floating-point value is completely filled with random bits.\n\n.. note::\n   The current implementation is buggy, because it may not fill all of the mantissa with random bits.");
  doc % name.str() % vartype;

  class_<D, boost::shared_ptr<D> >(name.str().c_str(), doc.str().c_str(), no_init)
    .def(init<optional<T, T> >((arg("self"), arg("min")=0, arg("max")=1), "Constructs a new object of this type, 'min' and 'max' are parameters of the distribution. 'min' has to be <= 'max'."))
    .add_property("min", &D::min)
    .add_property("max", &D::max)
    .def("reset", &D::reset, "This is a noop for this distribution, here only for consistency")
    .def("__call__", __call__<D,Engine>, (arg("self"), arg("rng")))
    ;
}

