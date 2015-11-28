.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Tue 15 Oct 14:59:05 2013

=========
 C++ API
=========

The C++ API of ``bob.core`` allows users to leverage from automatic converters
for classes in :py:class:`bob.core.random`.  To use the C API, clients should
first, include the header file ``<bob.core/random_api.h>`` on their compilation
units and then, make sure to call once ``import_bob_core_random()`` at their
module instantiation, as explained at the `Python manual
<http://docs.python.org/2/extending/extending.html#using-capsules>`_.

Here is a dummy C example showing how to include the header and where to call
the import function:

.. code-block:: c++

   #include <bob.blitz/capi.h>
   #include <bob.core/random_api.h>

   PyMODINIT_FUNC initclient(void) {

     PyObject* m Py_InitModule("client", ClientMethods);

     if (!m) return;

     // imports dependencies
     if (import_bob_blitz() < 0) {
       PyErr_Print();
       PyErr_SetString(PyExc_ImportError, "cannot import module");
       return 0;
     }

     // imports dependencies
     if (import_bob_core_random() < 0) {
       PyErr_Print();
       PyErr_SetString(PyExc_ImportError, "cannot import module");
       return 0;
     }

     return m;

   }


Mersenne Twister Random Number Generator (mt19937)
--------------------------------------------------

This package contains bindings to ``boost::mt19937``, which is a
powerful random number generator available within the Boost_ C++ library.

.. cpp:type:: PyBoostMt19937Object

   The pythonic object representation for a ``boost::mt19937`` object.

   .. code-block:: c

      typedef struct {
        PyObject_HEAD
        boost::shared_ptr<boost::mt19937> rng;
      } PyBoostMt19937Object;

   .. c:member:: boost::shared_ptr<boost::mt19937> rng

      A direct pointer to the boost random number generator. You can use this
      pointer in your C/C++ code if required.


.. cpp:function:: int PyBoostMt19937_Check(PyObject* o)

   Checks if the input object ``o`` is a ``PyBoostMt19937Object``. Returns
   ``1`` if it is, and ``0`` otherwise.


.. cpp:function:: int PyBoostMt19937_Converter(PyObject* o, PyBoostMt19937Object** a)

   This function is meant to be used with :c:func:`PyArg_ParseTupleAndKeywords`
   family of functions in the Python C-API. It checks the input
   object to be of type ``PyBoostMt19937Object`` and sets a **new
   reference** to it (in ``*a``) if it is the case. Returns ``0`` in case of
   failure, ``1`` in case of success.

.. cpp:function:: PyObject* PyBoostMt19937_SimpleNew()

   Creates a new instance of :cpp:type:`PyBoostMt19937Object`, with the default
   seed. Returns a **new reference**.

.. cpp:function:: PyObject* PyBoostMt19937_NewWithSeed(Py_ssize_t seed)

   Creates a new instance of :cpp:type:`PyBoostMt19937Object`, with a user
   given seed. Returns a **new reference**.

Distribution API
----------------

Together with the boost random number generator ``mt19937``, this package
provides bindings to these ``boost::random`` distributions:

  * Uniform
  * Normal (or Gaussian)
  * Log-normal
  * Gamma
  * Binomial

Distributions wrap the random number generator, skewing the distribution of
numbers according to their parametrization. Distributions are *templated*
according to the scalar data types they produce. Different distributions
support a different set of scalar types:

  ============== =================================================
   Distribution   Scalars supported
  ============== =================================================
    Uniform       bool, int8/16/32/64, uint8/16/32/64, float32/64
    Normal        float32/64
    Log-normal    float32/64
    Gamma         float32/64
    Binomial      float32/64 (internally using int64)
  ============== =================================================

.. cpp:type:: PyBoostUniformObject

   The pythonic object representation for a ``boost::random::uniform_*``
   object.

   .. code-block:: c

      typedef struct {
        PyObject_HEAD
        int type_num;
        boost::shared_ptr<void> distro;
      } PyUniformObject;

   .. c:member:: int type_num;

      The NumPy type number of scalars produced by this distribution. Accepted
      values match the scalar type produced:

       ============= ========================================
        Scalar type   NumPy scalar type number (enumeration)
       ============= ========================================
          bool        ``NPY_BOOL``
          int8        ``NPY_INT8``
          int16       ``NPY_INT16``
          int32       ``NPY_INT32``
          int64       ``NPY_INT64``
          int8        ``NPY_INT8``
          int16       ``NPY_INT16``
          int32       ``NPY_INT32``
          int64       ``NPY_INT64``
          float32     ``NPY_FLOAT32``
          float64     ``NPY_FLOAT64``
       ============= ========================================

   .. c:member:: boost::shared_ptr<void> distro

      A direct pointer to the boost distribution. The underlying allocated type
      changes with the scalar that is produced by the distribution:

       ============= ==============================================
        Scalar type   C++ data type
       ============= ==============================================
          bool        ``boost::random::uniform_smallint<uint8_t>``
          int8        ``boost::random::uniform_int<int8_t>``
          int16       ``boost::random::uniform_int<int16_t>``
          int32       ``boost::random::uniform_int<int32_t>``
          int64       ``boost::random::uniform_int<int64_t>``
          uint8       ``boost::random::uniform_int<uint8_t>``
          uint16      ``boost::random::uniform_int<uint16_t>``
          uint32      ``boost::random::uniform_int<uint32_t>``
          uint64      ``boost::random::uniform_int<uint64_t>``
          float32     ``boost::random::uniform_real<float>``
          float64     ``boost::random::uniform_real<double>``
       ============= ==============================================

   In order to use the distribution in your C/C++ code, you must first cast the
   shared pointer using ``boost::static_pointer_cast<D>``, with ``D`` matching
   one of the distributions listed above, depending on the value of

.. cpp:type:: PyBoostNormalObject

   The pythonic object representation for a
   ``boost::random::normal_distribution`` object.

   .. code-block:: c

      typedef struct {
        PyObject_HEAD
        int type_num;
        boost::shared_ptr<void> distro;
      } PyUniformObject;

   .. c:member:: int type_num;

      The NumPy type number of scalars produced by this distribution. Accepted
      values match the scalar type produced:

       ============= ========================================
        Scalar type   NumPy scalar type number (enumeration)
       ============= ========================================
          float32     ``NPY_FLOAT32``
          float64     ``NPY_FLOAT64``
       ============= ========================================

   .. c:member:: boost::shared_ptr<void> distro

      A direct pointer to the boost distribution. The underlying allocated type
      changes with the scalar that is produced by the distribution:

       ============= ================================================
        Scalar type   C++ data type
       ============= ================================================
          float32     ``boost::random::normal_distribution<float>``
          float64     ``boost::random::normal_distribution<double>``
       ============= ================================================

.. cpp:type:: PyBoostLogNormalObject

   The pythonic object representation for a
   ``boost::random::lognormal_distribution`` object.

   .. code-block:: c

      typedef struct {
        PyObject_HEAD
        int type_num;
        boost::shared_ptr<void> distro;
      } PyUniformObject;

   .. c:member:: int type_num;

      The NumPy type number of scalars produced by this distribution. Accepted
      values match the scalar type produced:

       ============= ========================================
        Scalar type   NumPy scalar type number (enumeration)
       ============= ========================================
          float32     ``NPY_FLOAT32``
          float64     ``NPY_FLOAT64``
       ============= ========================================

   .. c:member:: boost::shared_ptr<void> distro

      A direct pointer to the boost distribution. The underlying allocated type
      changes with the scalar that is produced by the distribution:

       ============= ===================================================
        Scalar type   C++ data type
       ============= ===================================================
          float32     ``boost::random::lognormal_distribution<float>``
          float64     ``boost::random::lognormal_distribution<double>``
       ============= ===================================================

.. cpp:type:: PyBoostGammaObject

   The pythonic object representation for a
   ``boost::random::gamma_distribution`` object.

   .. code-block:: c

      typedef struct {
        PyObject_HEAD
        int type_num;
        boost::shared_ptr<void> distro;
      } PyUniformObject;

   .. c:member:: int type_num;

      The NumPy type number of scalars produced by this distribution. Accepted
      values match the scalar type produced:

       ============= ========================================
        Scalar type   NumPy scalar type number (enumeration)
       ============= ========================================
          float32     ``NPY_FLOAT32``
          float64     ``NPY_FLOAT64``
       ============= ========================================

   .. c:member:: boost::shared_ptr<void> distro

      A direct pointer to the boost distribution. The underlying allocated type
      changes with the scalar that is produced by the distribution:

       ============= ===============================================
        Scalar type   C++ data type
       ============= ===============================================
          float32     ``boost::random::gamma_distribution<float>``
          float64     ``boost::random::gamma_distribution<double>``
       ============= ===============================================

.. cpp:type:: PyBoostBinomialObject

   The pythonic object representation for a
   ``boost::random::binomial_distribution`` object.

   .. code-block:: c

      typedef struct {
        PyObject_HEAD
        int type_num;
        boost::shared_ptr<void> distro;
      } PyBinomialObject;

   .. c:member:: int type_num;

      The NumPy type number of scalars produced by this distribution. Accepted
      values match the scalar type produced:

       ============= ========================================
        Scalar type   NumPy scalar type number (enumeration)
       ============= ========================================
          float32     ``NPY_FLOAT32``
          float64     ``NPY_FLOAT64``
       ============= ========================================

   .. c:member:: boost::shared_ptr<void> distro

      A direct pointer to the boost distribution. The underlying allocated type
      changes with the scalar that is produced by the distribution:

       ============= ==========================================================
        Scalar type   C++ data type
       ============= ==========================================================
          float32     ``boost::random::binomial_distribution<int64_t,float>``
          float64     ``boost::random::binomial_distribution<int64_t,double>``
       ============= ==========================================================

.. cpp:type:: PyBoostDiscreteObject

   The pythonic object representation for a
   ``boost::random::discrete_distribution`` object.

   .. code-block:: c

      typedef struct {
        PyObject_HEAD
        int type_num;
        boost::shared_ptr<void> distro;
      } PyDiscreteObject;

   .. c:member:: int type_num;

      The NumPy type number of scalars produced by this distribution. Accepted
      values match the scalar type produced:

       ============= ========================================
        Scalar type   NumPy scalar type number (enumeration)
       ============= ========================================
          int8        ``NPY_INT8``
          int16       ``NPY_INT16``
          int32       ``NPY_INT32``
          int64       ``NPY_INT64``
          int8        ``NPY_INT8``
          int16       ``NPY_INT16``
          int32       ``NPY_INT32``
          int64       ``NPY_INT64``
       ============= ========================================

   .. c:member:: boost::shared_ptr<void> distro

      A direct pointer to the boost distribution. The underlying allocated type
      changes with the scalar that is produced by the distribution:

       ============= ==========================================================
        Scalar type   C++ data type
       ============= ==========================================================
          int8        ``boost::random::uniform_int<int8_t>``
          int16       ``boost::random::uniform_int<int16_t>``
          int32       ``boost::random::uniform_int<int32_t>``
          int64       ``boost::random::uniform_int<int64_t>``
          uint8       ``boost::random::uniform_int<uint8_t>``
          uint16      ``boost::random::uniform_int<uint16_t>``
          uint32      ``boost::random::uniform_int<uint32_t>``
          uint64      ``boost::random::uniform_int<uint64_t>``
       ============= ==========================================================

.. cpp:function:: int PyBoostUniform_Check(PyObject* o)

.. cpp:function:: int PyBoostNormal_Check(PyObject* o)

.. cpp:function:: int PyBoostLogNormal_Check(PyObject* o)

.. cpp:function:: int PyBoostGamma_Check(PyObject* o)

.. cpp:function:: int PyBoostBinomial_Check(PyObject* o)

.. cpp:function:: int PyBoostDiscrete_Check(PyObject* o)

   Checks if the input object ``o`` is a ``PyBoost<Distribution>Object``.
   Returns ``1`` if it is, and ``0`` otherwise.

.. cpp:function:: int PyBoostUniform_Converter(PyObject* o, PyBoostUniformObject** a)

.. cpp:function:: int PyBoostNormal_Converter(PyObject* o, PyBoostNormalObject** a)

.. cpp:function:: int PyBoostLogNormal_Converter(PyObject* o, PyBoostLogNormalObject** a)

.. cpp:function:: int PyBoostGamma_Converter(PyObject* o, PyBoostGammaObject** a)

.. cpp:function:: int PyBoostBinomial_Converter(PyObject* o, PyBoostBinomialObject** a)

.. cpp:function:: int PyBoostDiscrete_Converter(PyObject* o, PyBoostDiscreteObject** a)

   This function is meant to be used with :c:func:`PyArg_ParseTupleAndKeywords`
   family of functions in the Python C-API. It checks the input object to be of
   type ``PyBoost<Distribution>Object`` and returns a **new reference** to it
   (in ``*a``) if it is the case. Returns ``0`` in case of failure, ``1`` in
   case of success.

.. cpp:function:: PyObject* PyBoostUniform_SimpleNew(int type_num, PyObject* min, PyObject* max)

   Creates a new instance of :cpp:type:`PyBoostUniformObject`, with the input
   scalar establishing the minimum and the maximum of the distribution. Note
   that ``bool`` distributions will raise an exception if one tries to set the
   minimum and the maximum, since that is non-sensical.

   The parameter ``type_num`` may be set to one of the supported ``NPY_``
   enumeration values (e.g. ``NPY_UINT16``).

   .. warning::

     For integral uniform distributions the range of numbers produced is
     defined as :math:`[min, max]`. For real-valued distributions, the range of
     numbers produced lies on the interval :math:`[min, max[`.

.. cpp:function:: PyObject* PyBoostNormal_SimpleNew(int type_num, PyObject* mean, PyObject* sigma)

.. cpp:function:: PyObject* PyBoostLogNormal_SimpleNew(int type_num, PyObject* mean, PyObject* sigma)

.. cpp:function:: PyObject* PyBoostGamma_SimpleNew(int type_num, PyObject* alpha)

.. cpp:function:: PyObject* PyBoostBinomial_SimpleNew(int type_num, PyObject* t, PyObject* p)

.. cpp:function:: PyObject* PyBoostDiscrete_SimpleNew(int type_num, PyObject* probs)

   Depending on the distribution, which may be one of ``Normal``,
   ``LogNormal``, ``Gamma``, ``Binomial`` or ``Discrete``, each of the parameters assume a
   different function:

     ============== ============= =============
      Distribution   Parameter 1   Parameter 2
     ============== ============= =============
      Normal         mean          sigma
      LogNormal      mean          sigma
      Gamma          alpha
      Binomial       t             p
      Discrete       probs
     ============== ============= =============

   The parameter ``type_num`` may be set to one of the supported ``NPY_``
   enumeration values (e.g. ``NPY_FLOAT64``).

.. include:: links.rst
