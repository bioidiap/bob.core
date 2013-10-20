/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 18 Oct 19:21:57 2013
 *
 * @brief Bindings to re-inject C++ messages into the Python logging module
 */

#include <Python.h>
#include <boost/shared_array.hpp>
#include <boost/make_shared.hpp>
#include <bob/core/logging.h>

#define PYTHON_LOGGING_DEBUG 0

#if PYTHON_LOGGING_DEBUG != 0
static bob::core::OutputStream static_log("stdout");
#endif

static void pyobject_destructor(PyObject* o) {
  Py_XDECREF(o);
}

/**
 * Objects of this class are able to redirect the data injected into a
 * bob::core::OutputStream to be re-injected in a given python callable object,
 * that is given upon construction. The key idea is that you feed in something
 * like logging.debug to the constructor, for the debug stream, logging.info
 * for the info stream and so on.
 */
struct PythonLoggingOutputDevice: public bob::core::OutputDevice {

  std::shared_ptr<PyObject> m_callable; ///< to stream the data out

  /**
   * Builds a new OutputDevice from a given callable
   *
   * @param callable A python callable object. Can be a function or an object
   * that implements the __call__() slot.
   *
   * This method is called from Python, so the GIL is on
   */
  PythonLoggingOutputDevice(PyObject* callable) {

      if (callable && callable != Py_None) {
        m_callable.reset(callable, &pyobject_destructor);
        Py_INCREF(callable);
      }

#if   PYTHON_LOGGING_DEBUG != 0
      pthread_t thread_id = pthread_self();
      static_log << "(0x" << std::hex << thread_id << std::dec
        << ") Constructing new PythonLoggingOutputDevice from callable"
        << std::endl;
#endif

    }

  /**
   * Copy constructor
   *
   * This method does not require the GIL since it does not deal with any
   * interpreter information. The value stored in ``m_callable`` is managed by
   * our std::shared_ptr object.
   */
  PythonLoggingOutputDevice(const PythonLoggingOutputDevice& other):
    m_callable(other.m_callable) {

#if   PYTHON_LOGGING_DEBUG != 0
      pthread_t thread_id = pthread_self();
      static_log << "(0x" << std::hex << thread_id << std::dec
        << ") Copy-constructing PythonLoggingOutputDevice"
        << std::endl;
#endif

    }

  /**
   * D'tor
   */
  virtual ~PythonLoggingOutputDevice() {
    close();
  }

  /**
   * Closes this stream for good
   */
  virtual void close() {
    m_callable.reset();
  }

  /**
   * Writes a message to the callable.
   *
   * Because this is called from C++ and, potentially, from other threads of
   * control, it ensures acquisition of the GIL.
   */
  virtual inline std::streamsize write(const char* s, std::streamsize n) {

    auto gil = PyGILState_Ensure();

    if (!m_callable || m_callable.get() == Py_None) {
      PyGILState_Release(gil);
      return 0;
    }

#if   PYTHON_LOGGING_DEBUG != 0
    pthread_t thread_id = pthread_self();
    static_log << "(0x" << std::hex << thread_id << std::dec
      << ") Processing message `" << value << "' (size = " << n << ")" << std::endl;
#endif

    int len = strlen(s);
    while (std::isspace(s[len-1])) len -= 1;
    PyObject* value = Py_BuildValue("s#", s, len);

    PyObject* result = PyObject_CallFunctionObjArgs(m_callable.get(), value, 0);
    Py_DECREF(value);
    if (!result) len = 0;
    else Py_DECREF(result);

#if   PYTHON_LOGGING_DEBUG != 0
    result = PyObject_CallMethod(m_callable.get(), "flush", 0);
    Py_XDECREF(result);
#endif

    PyGILState_Release(gil);

    return len;
  }

  virtual std::shared_ptr<bob::core::OutputDevice> clone() const {
    return std::make_shared<PythonLoggingOutputDevice>(*this);
  }

};

static int set_stream(boost::iostreams::stream<bob::core::AutoOutputDevice>& s, 
    PyObject* o, const char* n) {

  // if no argument or None, write everything else to stderr
  if (!o || o == Py_None) {
    s.close();
    //s.open("stderr");
    return 1;
  }

  if (PyCallable_Check(o)) {
    s.close();
    s.open(boost::make_shared<PythonLoggingOutputDevice>(o));
    return 1;
  }

  // if you get to this point, set an error
  PyErr_Format(PyExc_ValueError, "argument to set stream `%s' is optional, but if set, it needs to be either None or a callable", n);
  return 0;

}

static PyObject* reset(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "debug", 
    "info", 
    "warn", 
    "error",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* debug = 0;
  PyObject* info = 0;
  PyObject* warn = 0;
  PyObject* error = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOO",
        kwlist, &debug, &info, &warn, &error)) return 0;

  if (!set_stream(bob::core::debug, debug, "debug")) return 0;
  if (!set_stream(bob::core::info, info, "info")) return 0;
  if (!set_stream(bob::core::warn, warn, "warn")) return 0;
  if (!set_stream(bob::core::error, error, "error")) return 0;

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

    *(mi->s) << mi->message << " (thread " << mi->thread_id << "; iteration " << i << ")" << std::endl;
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
"Logs a message into Bob's logging system from C++.\n\
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
      << " == 0x" << std::hex << threads[i] << std::dec
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
"Logs a message into Bob's logging system from C++, in a separate thread.\n\
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

static PyMethodDef logging_methods[] = {
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

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC init_logging(void)
{
  PyObject* m;
  
  m = Py_InitModule3("_logging", logging_methods, "C++ logging handling");
}
