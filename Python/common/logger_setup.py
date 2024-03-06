# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" Logger modul for other programs/apps.

This module consist of one function, the 'config_logger', which configures
the root logger by the 'logging_config.json' file.

**IMPORTANT: THE DEFAULT TWO HANDLER MUST BE UNCHANGED!**
The 'file' handler filepath is programatically modified to match the indented location independent of the
install location. Also, the 'file' handler executes a rollover at every start. When modifying the config file,
this two handler and their order must be unchanged because it is hardcoded in this script.

**EVERY MODUL WITHIN THIS PROJECT SCOPE
MUST USE THE FOLLOWING SYNTAX AT THE HEADER TO ACCESS THIS LOGGER:**

.. code-block::

   import logging
   logger = logging.getLogger(__name__)

Help
----
* https://www.youtube.com/watch?v=9L77QExPmI0&t=3s
* https://github.com/mCodingLLC/VideosSampleCode/blob/master/videos/135_modern_\
logging/logging_configs/2-stderr-json-file.json
* https://www.toptal.com/python/in-depth-python-logging
* https://stackoverflow.com/questions/15727420/using-logging-in-multiple-modules
* https://stackoverflow.com/questions/404744/determining-application-path-in-a-python-\
exe-generated-by-pyinstaller

Contents
--------
"""


import logging.config
import os
import sys
import json


# NOTE: Wihtout this, the .exe file not works!
if getattr(sys, 'frozen', False):
    INITDIR = os.path.dirname(os.path.dirname(sys.executable))
else:
    INITDIR = os.path.dirname(os.path.dirname(__file__))

# Set defaults
LOG_FILENAME = "log.txt"
LOG_DIRNAME = "log"
LOG_PATH = os.path.join(INITDIR, LOG_DIRNAME, LOG_FILENAME)


class FilterFontManager(logging.Filter):
    """ Disable matplotlib.font_manager logs. """
    def filter(self, record) -> bool:
        return record.name != "matplotlib.font_manager"


class FilterPngImagePlugin(logging.Filter):
    """ Disable PIL.PngImagePlugin logs. """
    def filter(self, record) -> bool:
        return record.name != "PIL.PngImagePlugin"


def config_logger():
    """ Configures root logger using the predefined config file.
    Every child logger will inherit the root logger configs, so no need to do that every time.
    """
    # If logging folder is missing, create it
    if not os.path.isdir(os.path.join(INITDIR, LOG_DIRNAME)):
        os.mkdir(os.path.join(INITDIR, LOG_DIRNAME))

    # Configure root logger
    config_file = os.path.join(INITDIR, "database/logging_config.json")
    with open(config_file, "r", encoding="utf-8") as file:
        config = json.load(file)
    config["handlers"]["file"]["filename"] = LOG_PATH  # Override log's path by absolute path
    logging.config.dictConfig(config)  # Set configs to root logger
    logging.getLogger("root").handlers[1].doRollover()  # Rollover file logger aka create new file at every restart

    # Initializes logger for this module
    logger = logging.getLogger(__name__)
    logger.info("Main logger created!")
    logger.info("Log file path: %s", LOG_PATH)


# Modul-teszt
if __name__ == "__main__":
    config_logger()
