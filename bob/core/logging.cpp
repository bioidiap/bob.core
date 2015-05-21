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
    PyObject* result = PyObject_CallMethodObjArgs(m_logger, m_name, value, 0);
    Py_DECREF(value);

    if (!result) len = 0;
    else Py_DECREF(result);

    PyGILState_Release(gil);

    return n;
  }

};

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
    if (callable && PyCallable_Check(callable)) {

#if   PYTHON_LOGGING_DEBUG != 0
    pthread_t thread_id = pthread_self();
    static_log << "(" << std::hex << thread_id << std::dec
      << ") Setting stream `" << n << "' to logger at " << std::hex
      << o << std::dec << std::endl;
#endif

      s.close();
      s.open(boost::make_shared<PythonLoggingOutputDevice>(o, n));
      Py_DECREF(callable);
      return 1;
    }
  }

  // if you get to this point, set an error
  PyErr_Format(PyExc_ValueError, "argument to set stream `%s' needs to be either None or an object with a callable named `%s'", n, n);
  return 0;

}

static PyObject* reset(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "logger",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* logger = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O",
        kwlist, &logger)) return 0;

  if (!set_stream(bob::core::debug, logger, "debug")) return 0;
  if (!set_stream(bob::core::info, logger, "info")) return 0;
  if (!set_stream(bob::core::warn, logger, "warn")) return 0;
  if (!set_stream(bob::core::error, logger, "error")) return 0;

  Py_RETURN_NONE;
}

PyDoc_STRVAR(s_reset_str, "reset");
PyDoc_STRVAR(s_reset__doc__,
"reset([debug, [info, [warn, [error]]]]) -> None\n\
\n\
Resets the standard C++ logging devices.\n\
\n\
This function allows you to manipulate the sinks for messages emitted\n\
in C++, using Python callables.\n\
\n\
Keyword Parameters:\n\
\n\
  debug\n\
    [optional] (callable) A callable that receives a string and dumps\n\
    messages to the desired output channel.\n\
  \n\
  info\n\
    [optional] (callable) A callable that receives a string and dumps\n\
    messages to the desired output channel.\n\
  \n\
  warn\n\
    [optional] (callable) A callable that receives a string and dumps\n\
    messages to the desired output channel.\n\
  \n\
  error\n\
    [optional] (callable) A callable that receives a string and dumps\n\
    messages to the desired output channel.\n\
  \n\
Raises a :py:class:`ValueError` in case of problems setting or resetting\n\
any of the streams.\n\
"
);

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

/**
 * A test function for your python bindings
 */
static PyObject* log_message(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "ntimes",
    "stream",
    "message",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  unsigned int ntimes = 0;
  const char* stream = 0;
  const char* message = 0;

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
    PyErr_SetString(PyExc_ValueError, "parameter `stream' must be one of 'debug', 'info', 'warn', 'error' or 'fatal' (synomym for 'error')");
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
}

PyDoc_STRVAR(s_logmsg_str, "__log_message__");
PyDoc_STRVAR(s_logmsg__doc__,
"__log_message__(ntimes, stream, message) -> None\n\
\n\
Logs a message into Bob's logging system from C++.\n\
\n\
This method is included for testing purposes only and should not be\n\
considered part of the Python API for Bob.\n\
\n\
Keyword parameters:\n\
\n\
  ntimes\n\
    (integer) The number of times to print the given message\n\
  \n\
  stream\n\
    (string) The stream to use for logging the message. Choose from:\n\
    ``'debug'``, ``'info'``, ``'warn'`` or ``'error'``\n\
  \n\
  message\n\
    (string) The message to be logged.\n\
\n"
);

/**
 * Logs a number of messages from a separate thread
 */
static PyObject* log_message_mt(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "nthreads",
    "ntimes",
    "stream",
    "message",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  unsigned int nthreads = 0;
  unsigned int ntimes = 0;
  const char* stream = 0;
  const char* message = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "IIss",
        kwlist, &nthreads, &ntimes, &stream, &message)) return 0;

  // implements: if stream not in ('debug', 'info', 'warn', 'error')
  boost::iostreams::stream<bob::core::AutoOutputDevice>* s = 0;
  if (strncmp(stream, "debug", 5)) s = &bob::core::debug;
  else if (strncmp(stream, "info", 4)) s = &bob::core::info;
  else if (strncmp(stream, "warn", 4)) s = &bob::core::warn;
  else if (strncmp(stream, "error", 5)) s = &bob::core::error;
  else {
    PyErr_SetString(PyExc_ValueError, "parameter `stream' must be one of 'debug', 'info', 'warn' or 'error'");
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
}

PyDoc_STRVAR(s_logmsg_mt_str, "__log_message_mt__");
PyDoc_STRVAR(s_logmsg_mt__doc__,
"__log_message_mt__(nthreads, ntimes, stream, message) -> None\n\
\n\
Logs a message into Bob's logging system from C++, in a separate thread.\n\
\n\
This method is included for testing purposes only and should not be\n\
considered part of the Python API for Bob.\n\
\n\
Keyword parameters:\n\
\n\
  nthreads\n\
    (integer) The total number of threads from which to write messages\n\
    to the logging system using the C++->Python API.\n\
  ntimes\n\
    (integer) The number of times to print the given message\n\
  \n\
  stream\n\
    (string) The stream to use for logging the message. Choose from:\n\
    ``'debug'``, ``'info'``, ``'warn'`` or ``'error'``\n\
  \n\
  message\n\
    (string) The message to be logged.\n\
\n"
);

static PyMethodDef module_methods[] = {
    {
      s_reset_str,
      (PyCFunction)reset,
      METH_VARARGS|METH_KEYWORDS,
      s_reset__doc__
    },
    {
      s_logmsg_str,
      (PyCFunction)log_message,
      METH_VARARGS|METH_KEYWORDS,
      s_logmsg__doc__
    },
    {
      s_logmsg_mt_str,
      (PyCFunction)log_message_mt,
      METH_VARARGS|METH_KEYWORDS,
      s_logmsg_mt__doc__
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
# else
  PyObject* m = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
# endif
  if (!m) return 0;
  auto m_ = make_safe(m);

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

  return Py_BuildValue("O", m);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
