/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 18 Oct 19:21:57 2013
 *
 * @brief Bindings to re-inject C++ messages into the Python logging module
 */

#define BOB_CORE_LOGGING_MODULE
#include <bob.core/api.h>

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/documentation.h>

#include <boost/shared_array.hpp>
#include <boost/make_shared.hpp>

#define PYTHON_LOGGING_DEBUG 0

#if PYTHON_LOGGING_DEBUG != 0
#include <boost/algorithm/string.hpp>
static boost::iostreams::stream<bob::core::AutoOutputDevice> static_log("stdout");
#endif

/**
 * Free standing functions for the module's C-API
 */
int PyBobCoreLogging_APIVersion = BOB_CORE_API_VERSION;

boost::iostreams::stream<bob::core::AutoOutputDevice>& PyBobCoreLogging_Debug() {
  return bob::core::debug;
}

boost::iostreams::stream<bob::core::AutoOutputDevice>& PyBobCoreLogging_Info() {
  return bob::core::info;
}

boost::iostreams::stream<bob::core::AutoOutputDevice>& PyBobCoreLogging_Warn() {
  return bob::core::warn;
}

boost::iostreams::stream<bob::core::AutoOutputDevice>& PyBobCoreLogging_Error() {
  return bob::core::error;
}

/**
 * Objects of this class are able to redirect the data injected into a
 * boost::iostreams::stream<bob::core::AutoOutputDevice> to be re-injected in a
 * given python callable object, that is given upon construction. The key idea
 * is that you feed in something like logging.debug to the constructor, for the
 * debug stream, logging.info for the info stream and so on.
 */
struct PythonLoggingOutputDevice: public bob::core::OutputDevice {

  PyObject* m_logger; ///< to stream the data out
  PyObject* m_name; ///< the name of the method to call on the object

  /**
   * Builds a new OutputDevice from a given callable
   *
   * @param logger Is a logging.logger-style object.
   * @param name Is the name of the method to call for logging.
   *
   * This method is called from Python, so the GIL is on
   */
  PythonLoggingOutputDevice(PyObject* logger, const char* name):
    m_logger(0), m_name(0)
  {

      if (logger && logger != Py_None) {
        m_logger = Py_BuildValue("O", logger);
        m_name = Py_BuildValue("s", name);
      }

#if   PYTHON_LOGGING_DEBUG != 0
      pthread_t thread_id = pthread_self();
      static_log << "(" << std::hex << thread_id << std::dec
                 << ") Constructing new PythonLoggingOutputDevice from logger `logging.logger('"
                 << PyString_AsString(PyObject_GetAttrString(m_logger, "name")) << "')."
                 << name << "' (@" << std::hex << m_logger << std::dec
                 << ")" << std::endl;
#endif

    }

  /**
   * D'tor
   */
  virtual ~PythonLoggingOutputDevice() {
#if PYTHON_LOGGING_DEBUG != 0
    pthread_t thread_id = pthread_self();
    const char* _name = "null";
    if (m_logger) {
      _name = PyString_AsString(PyObject_GetAttrString(m_logger, "name"));
    }
    static_log << "(" << std::hex << thread_id << std::dec
               << ") Destroying PythonLoggingOutputDevice with logger `" << _name
               << "' (" << std::hex << m_logger << std::dec << ")" << std::endl;
#endif
    if (m_logger) close();
  }

  /**
   * Closes this stream for good
   */
  virtual void close() {
#if PYTHON_LOGGING_DEBUG != 0
    pthread_t thread_id = pthread_self();
    const char* _name = "null";
    if (m_logger) {
      _name = PyString_AsString(PyObject_GetAttrString(m_logger, "name"));
    }
    static_log << "(" << std::hex << thread_id << std::dec
               << ") Closing PythonLoggingOutputDevice with logger `" << _name
               << "' (" << std::hex << m_logger << std::dec << ")" << std::endl;
#endif
    Py_XDECREF(m_logger);
    m_logger = 0;
    Py_XDECREF(m_name);
    m_name = 0;
  }

  /**
   * Writes a message to the callable.
   *
   * Because this is called from C++ and, potentially, from other threads of
   * control, it ensures acquisition of the GIL.
   */
  virtual inline std::streamsize write(const char* s, std::streamsize n) {

    auto gil = PyGILState_Ensure();

    if (!m_logger || m_logger == Py_None) {
      PyGILState_Release(gil);
      return 0;
    }

#if   PYTHON_LOGGING_DEBUG != 0
    pthread_t thread_id = pthread_self();
    std::string message(s, n);
    static_log << "(" << std::hex << thread_id << std::dec
               << ") Processing message `" << boost::algorithm::trim_right_copy(message)
               << "' (size = " << n << ") with method `logging.logger('"
               << PyString_AsString(PyObject_GetAttrString(m_logger, "name")) << "')."
               << PyString_AsString(m_name) << "'" << std::endl;
#endif

    int len = n;
    while (std::isspace(s[len-1])) len -= 1;

    PyObject* value = Py_BuildValue("s#", s, len);
    auto value_ = make_safe(value);
    PyObject* result = PyObject_CallMethodObjArgs(m_logger, m_name, value, 0);
    auto result_ = make_xsafe(result);

    if (!result) len = 0;

    PyGILState_Release(gil);

    return n;
  }

};


static auto reset_doc = bob::extension::FunctionDoc(
  "reset",
  "Resets the standard C++ logging devices, or sets it to the given callable",
  "This function allows you to manipulate the sinks for messages emitted in C++, using Python callables. "
  "The first variant (without parameters) will reset all logging output to :py:data:`sys.stderr`. "
  "The second variant will reset the given logger to the given callable. "
  "If ``stream`` is not specified, it resets all loggers.\n\n"
  "This function raises a :py:exc:`ValueError` in case of problems setting or resetting any of the streams."
)
.add_prototype("")
.add_prototype("callable, [stream]")
.add_parameter("callable", "callable", "A python callable that receives an ``str`` and dumps messages to the desired output channel")
.add_parameter("stream", "one of ('debug', 'info', warn', 'error')", "[optional] If specified, only the given logger is send to the given callable. Otherwise all loggers are reset to that callable.")
;
static int set_stream(boost::iostreams::stream<bob::core::AutoOutputDevice>& s, PyObject* o, const char* n) {
  // if no argument or None, write everything else to stderr
  if (!o || o == Py_None) {
#if   PYTHON_LOGGING_DEBUG != 0
    pthread_t thread_id = pthread_self();
    static_log << "(" << std::hex << thread_id << std::dec
               << ") Resetting stream `" << n << "' to stderr" << std::endl;
#endif
    s.close();
    s.open("stderr");
    return 1;
  }

  if (PyObject_HasAttrString(o, n)) {
    PyObject* callable = PyObject_GetAttrString(o, n);
    auto callable_ = make_safe(callable);
    if (callable && PyCallable_Check(callable)) {
#if   PYTHON_LOGGING_DEBUG != 0
      pthread_t thread_id = pthread_self();
      static_log << "(" << std::hex << thread_id << std::dec
                 << ") Setting stream `" << n << "' to logger at " << std::hex
                 << o << std::dec << std::endl;
#endif

      s.close();
      s.open(boost::make_shared<PythonLoggingOutputDevice>(o, n));
      return 1;
    }
  }

  // if you get to this point, set an error
  PyErr_Format(PyExc_ValueError, "argument to set stream `%s' needs to be either None or an object with a callable named `%s'", n, n);
  return 0;
}

static PyObject* reset(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = reset_doc.kwlist(1);

  PyObject* callable = 0;
  const char* stream = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Os", kwlist, &callable, &stream)) return 0;

  if (!stream){
    // reset all streams to either stderr or the given callable
    if (!set_stream(bob::core::debug, callable, "debug")) return 0;
    if (!set_stream(bob::core::info, callable, "info")) return 0;
    if (!set_stream(bob::core::warn, callable, "warn")) return 0;
    if (!set_stream(bob::core::error, callable, "error")) return 0;
  } else {
    if (strcmp(stream, "debug") && strcmp(stream, "info") && strcmp(stream, "warn") && strcmp(stream, "error")){
      PyErr_Format(PyExc_ValueError, "If given, the parameter 'stream' needs to be one of ('debug', 'info', warn', 'error), not %s", stream);
      return 0;
    }
    if (!set_stream(bob::core::error, callable, stream)) return 0;
  }

  Py_RETURN_NONE;
BOB_CATCH_FUNCTION("reset", 0)
}

/**************************
 * Testing Infrastructure *
 **************************/

struct message_info_t {
  boost::iostreams::stream<bob::core::AutoOutputDevice>* s;
  std::string message;
  bool exit;
  unsigned int ntimes;
  unsigned int thread_id;
};

static void* log_message_inner(void* cookie) {

  message_info_t* mi = (message_info_t*)cookie;

# if PYTHON_LOGGING_DEBUG != 0
  if (PyEval_ThreadsInitialized()) {
    static_log << "(thread " << mi->thread_id << ") Python threads initialized correctly for this thread" << std::endl;
  }
  else {
    static_log << "(thread " << mi->thread_id << ") Python threads NOT INITIALIZED correctly for this thread" << std::endl;
  }
# endif

  for (unsigned int i=0; i<(mi->ntimes); ++i) {

#   if PYTHON_LOGGING_DEBUG != 0
    static_log << "(thread " << mi->thread_id << ") Injecting message `" << mi->message << " (thread " << mi->thread_id << "; iteration " << i << ")'" << std::endl;
#   endif

    *(mi->s) << mi->message
#     if PYTHON_LOGGING_DEBUG != 0
            << " (thread " << mi->thread_id << "; iteration " << i << ")"
#     endif
            << std::endl;

    mi->s->flush();
  }
  if (mi->exit) {
#   if PYTHON_LOGGING_DEBUG != 0
    static_log << "(thread " << mi->thread_id << ") Exiting this thread" << std::endl;
#   endif
    pthread_exit(0);
  }

# if PYTHON_LOGGING_DEBUG != 0
  if (mi->exit) {
    static_log << "(thread " << mi->thread_id << ") Returning 0" << std::endl;
  }
# endif
  return 0;
}

static auto _logmsg_doc = bob::extension::FunctionDoc(
  "_log_message",
  "Logs a message into Bob's logging system from C++",
  "This function is bound for testing purposes only and is not part of the Python API for bob.core"
)
.add_prototype("ntimes, stream, message")
.add_parameter("ntimes", "int", "The number of times to print the given message")
.add_parameter("stream", "str", "The stream to use for logging the message. Choose from ``('debug', 'info', 'warn', 'error')")
.add_parameter("message", "str", "The message to be logged")
;
static PyObject* log_message(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = _logmsg_doc.kwlist();

  unsigned int ntimes;
  const char* stream;
  const char* message;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Iss",
        kwlist, &ntimes, &stream, &message)) return 0;

  // implements: if stream not in ('debug', 'info', 'warn', 'error')
  boost::iostreams::stream<bob::core::AutoOutputDevice>* s = 0;
  if (strncmp(stream, "debug", 5) == 0) s = &bob::core::debug;
  else if (strncmp(stream, "info", 4) == 0) s = &bob::core::info;
  else if (strncmp(stream, "warn", 4) == 0) s = &bob::core::warn;
  else if (strncmp(stream, "error", 5) == 0) s = &bob::core::error;
  else if (strncmp(stream, "fatal", 5) == 0) s = &bob::core::error;
  else {
    PyErr_Format(PyExc_ValueError, "parameter `stream' must be one of 'debug', 'info', 'warn', 'error' or 'fatal' (synomym for 'error'), not '%s'", stream);
    return 0;
  }

  // do the work for this function
  auto no_gil = PyEval_SaveThread();

  message_info_t mi = {s, message, false, ntimes, 0};
  log_message_inner((void*)&mi);
# if PYTHON_LOGGING_DEBUG != 0
  static_log << "(thread 0) Returning to caller" << std::endl;
# endif

  PyEval_RestoreThread(no_gil);

  Py_RETURN_NONE;
BOB_CATCH_FUNCTION("_log_message", 0)
}


static auto _logmsg_mt_doc = bob::extension::FunctionDoc(
  "_log_message_mt",
  "Logs a message into Bob's logging system from C++, in a separate thread",
  "This function is bound for testing purposes only and is not part of the Python API for bob.core"
)
.add_prototype("nthreads, ntimes, stream, message")
.add_parameter("nthreads", "int", "The total number of threads from which to write messages to the logging system using the C++->Python API")
.add_parameter("ntimes", "int", "The number of times to print the given message")
.add_parameter("stream", "str", "The stream to use for logging the message. Choose from ``('debug', 'info', 'warn', 'error')")
.add_parameter("message", "str", "The message to be logged")
;
static PyObject* log_message_mt(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = _logmsg_mt_doc.kwlist();

  unsigned int nthreads;
  unsigned int ntimes;
  const char* stream;
  const char* message;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "IIss",
        kwlist, &nthreads, &ntimes, &stream, &message)) return 0;

  // implements: if stream not in ('debug', 'info', 'warn', 'error')
  boost::iostreams::stream<bob::core::AutoOutputDevice>* s = 0;
  if (strncmp(stream, "debug", 5) == 0) s = &bob::core::debug;
  else if (strncmp(stream, "info", 4) == 0) s = &bob::core::info;
  else if (strncmp(stream, "warn", 4) == 0) s = &bob::core::warn;
  else if (strncmp(stream, "error", 5) == 0) s = &bob::core::error;
  else {
    PyErr_Format(PyExc_ValueError, "parameter `stream' must be one of 'debug', 'info', 'warn' or 'error', not '%s'", stream);
    return 0;
  }

  // do the work for this function
  auto no_gil = PyEval_SaveThread();

  boost::shared_array<pthread_t> threads(new pthread_t[nthreads]);
  boost::shared_array<message_info_t> infos(new message_info_t[nthreads]);
  for (unsigned int i=0; i<nthreads; ++i) {
    message_info_t mi = {s, message, true, ntimes, i+1};
    infos[i] = mi;
  }

# if PYTHON_LOGGING_DEBUG != 0
  static_log << "(thread 0) Launching " << nthreads << " thread(s)" << std::endl;
# endif

  for (unsigned int i=0; i<nthreads; ++i) {

#   if PYTHON_LOGGING_DEBUG != 0
    static_log << "(thread 0) Launch thread " << (i+1) << ": `" << message << "'" << std::endl;
#   endif

    pthread_create(&threads[i], NULL, &log_message_inner, (void*)&infos[i]);

#   if PYTHON_LOGGING_DEBUG != 0
    static_log << "(thread 0) thread " << (i+1)
               << " == " << std::hex << threads[i] << std::dec
               << " launched" << std::endl;
#   endif

  }

  void* status;
# if PYTHON_LOGGING_DEBUG != 0
  static_log << "(thread 0) Waiting " << nthreads << " thread(s)" << std::endl;
# endif
  for (unsigned int i=0; i<nthreads; ++i) {
    pthread_join(threads[i], &status);
#   if PYTHON_LOGGING_DEBUG != 0
    static_log << "(thread 0) Waiting on thread " << (i+1) << std::endl;
#   endif
  }
# if PYTHON_LOGGING_DEBUG != 0
  static_log << "(thread 0) Returning to caller" << std::endl;
# endif

  PyEval_RestoreThread(no_gil);

  Py_RETURN_NONE;
BOB_CATCH_FUNCTION("_log_message_mt", 0)
}


static PyMethodDef module_methods[] = {
    {
      reset_doc.name(),
      (PyCFunction)reset,
      METH_VARARGS|METH_KEYWORDS,
      reset_doc.doc()
    },
    {
      _logmsg_doc.name(),
      (PyCFunction)log_message,
      METH_VARARGS|METH_KEYWORDS,
      _logmsg_doc.doc()
    },
    {
      _logmsg_mt_doc.name(),
      (PyCFunction)log_message_mt,
      METH_VARARGS|METH_KEYWORDS,
      _logmsg_mt_doc.doc()
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "C++ logging handling");

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
  auto m_ = make_xsafe(m);
  const char* ret = "O";
# else
  PyObject* m = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
  const char* ret = "N";
# endif
  if (!m) return 0;

  static void* PyBobCoreLogging_API[PyBobCoreLogging_API_pointers];

  /* exhaustive list of C APIs */
  PyBobCoreLogging_API[PyBobCoreLogging_APIVersion_NUM] = (void *)&PyBobCoreLogging_APIVersion;

  /*********************************
   * Bindings for bob.core.logging *
   *********************************/

  PyBobCoreLogging_API[PyBobCoreLogging_Debug_NUM] = (void *)PyBobCoreLogging_Debug;

  PyBobCoreLogging_API[PyBobCoreLogging_Info_NUM] = (void *)PyBobCoreLogging_Info;

  PyBobCoreLogging_API[PyBobCoreLogging_Warn_NUM] = (void *)PyBobCoreLogging_Warn;

  PyBobCoreLogging_API[PyBobCoreLogging_Error_NUM] = (void *)PyBobCoreLogging_Error;

#if PY_VERSION_HEX >= 0x02070000

  /* defines the PyCapsule */

  PyObject* c_api_object = PyCapsule_New((void *)PyBobCoreLogging_API,
      BOB_EXT_MODULE_PREFIX "." BOB_EXT_MODULE_NAME "._C_API", 0);

#else

  PyObject* c_api_object = PyCObject_FromVoidPtr((void *)PyBobCoreLogging_API, 0);

#endif

  if (c_api_object) PyModule_AddObject(m, "_C_API", c_api_object);

  /* imports dependencies */
  if (import_bob_blitz() < 0) return 0;

  return Py_BuildValue(ret, m);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
