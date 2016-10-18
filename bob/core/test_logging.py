#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 09 Jul 2013 13:24:49 CEST

"""Tests for the logging subsystem
"""

import bob.core

def test_from_python():
  logger = bob.core.log.setup("bob.core")
  bob.core.log.set_verbosity_level(logger, 2)

  # send a log message
  logger.debug("This is a test debug message")
  logger.info("This is a test info message")
  bob.core.log.set_verbosity_level(logger, 0)
  logger.warn("This is a test warn message")
  logger.error("This is a test error message")


def test_from_python_output():
  import logging, sys

  if sys.version_info[0] < 3:
    from StringIO import StringIO
  else:
    from io import StringIO
  # create an artificial logger using the logging module
  logger = logging.getLogger("XXX.YYY")

  # add handlers
  out, err = StringIO(), StringIO()
  _warn_err = logging.StreamHandler(err)
  _warn_err.setLevel(logging.WARNING)
  logger.addHandler(_warn_err)

  _debug_info = logging.StreamHandler(out)
  _debug_info.setLevel(logging.DEBUG)
  _debug_info.addFilter(bob.core.log._InfoFilter())
  logger.addHandler(_debug_info)

  # now, set up the logger
  logger = bob.core.log.setup("XXX.YYY")

  # send log messages
  bob.core.log.set_verbosity_level(logger, 2)
  logger.debug("This is a test debug message")
  logger.info("This is a test info message")

  bob.core.log.set_verbosity_level(logger, 0)
  logger.warn("This is a test warn message")
  logger.error("This is a test error message")

  out = out.getvalue().rstrip()
  err = err.getvalue().rstrip()

  assert out.startswith("XXX.YYY")
  assert "INFO" in out
  assert out.endswith("This is a test info message")

  assert err.startswith("XXX.YYY")
  assert "ERROR" in err
  assert err.endswith("This is a test error message")


def test_from_cxx():
  from bob.core._test import _test_log_message
  _test_log_message(1, 'error', 'this is a test message')

def test_from_cxx_multithreaded():
  from bob.core._test import _test_log_message_mt
  _test_log_message_mt(2, 1, 'error', 'this is a test message')



def test_from_cxx_disable():
  from bob.core._test import _test_output_disable
  import sys, os
  import threading

  class OutputGrabber(object):
      """
      Class used to grab standard output or another stream.
      This is a slight modification of what was proposed here:
      http://stackoverflow.com/a/29834357/3301902
      """
      escape_char = "\b"

      def __init__(self, stream=None, threaded=False):
          self.origstream = stream
          self.threaded = threaded
          if self.origstream is None:
              self.origstream = sys.stdout
          self.origstreamfd = self.origstream.fileno()
          self.capturedtext = ""
          # Create a pipe so the stream can be captured:
          self.pipe_out, self.pipe_in = os.pipe()

      def start(self):
          """
          Start capturing the stream data.
          """
          self.capturedtext = ""
          # Save a copy of the stream:
          self.streamfd = os.dup(self.origstreamfd)
          # Replace the Original stream with our write pipe
          os.dup2(self.pipe_in, self.origstreamfd)
          if self.threaded:
              # Start thread that will read the stream:
              self.workerThread = threading.Thread(target=self.readOutput)
              self.workerThread.start()
              # Make sure that the thread is running and os.read is executed:
              time.sleep(0.01)

      def stop(self):
          """
          Stop capturing the stream data and save the text in `capturedtext`.
          """
          # Flush the stream to make sure all our data goes in before
          # the escape character.
          self.origstream.flush()
          # Print the escape character to make the readOutput method stop:
          self.origstream.write(self.escape_char)
          self.origstream.flush()
          if self.threaded:
              # wait until the thread finishes so we are sure that
              # we have until the last character:
              self.workerThread.join()
          else:
              self.readOutput()
          # Close the pipe:
          os.close(self.pipe_out)
          # Restore the original stream:
          os.dup2(self.streamfd, self.origstreamfd)

      def readOutput(self):
          """
          Read the stream data (one byte at a time)
          and save the text in `capturedtext`.
          """
          while True:
              data = os.read(self.pipe_out, 1)  # Read One Byte Only
              if self.escape_char in data:
                  break
              if not data:
                  break
              self.capturedtext += data


  try:
    # redirect output and error streams
    out = OutputGrabber(sys.stdout)
    err = OutputGrabber(sys.stderr)
    out.start()
    err.start()

    # run our code
    _test_output_disable()

  finally:
    # Clean up the pipe and restore the original stdout
    out.stop()
    err.stop()

  # output should contain two lines
  outs = out.capturedtext.rstrip().split("\n")
  assert len(outs) == 2
  assert outs[0] == "This is a debug message"
  assert outs[1] == "This is an info message"

  # error should contain three lines; error message is printed twice
  errs = err.capturedtext.rstrip().split("\n")
  assert len(errs) == 3
  assert errs[0] == "This is a warning message"
  assert errs[1] == "This is an error message"
  assert errs[2] == "This is an error message"
