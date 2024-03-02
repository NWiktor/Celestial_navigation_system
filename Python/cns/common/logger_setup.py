# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" Logger modul for other programs/apps.

Initializes and returs a logger named *'Main'* for all modules. This modul
contains all settings for the logger. The logger level is DEBUG, as it is
intended to catch every log message. **EVERY MODUL WITHIN THIS PROJECT SCOPE
MUST USE THE FOLLOWING SYNTAX AT THE HEADER TO ACCESS THIS LOGGER:**

.. code-block::

   from logger import MAIN_LOGGER as l

The logger consists of two log handler: a console handler, and filehandler,
the latter uses filerotating and stores the logs of the last five (5)
running.

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
import pathlib
from logging.handlers import RotatingFileHandler


# NOTE: Wihtout this, the .exe file not works!
if getattr(sys, 'frozen', False):
    initdir = os.path.dirname(sys.executable)
else:
    initdir = os.path.dirname(__file__)

# Set defaults
LOG_DIRPATH = "log/"
LOG_FILENAME = "main.txt"
MAIN_LOGGER = None
LOG_FILE = os.path.join(initdir, LOG_DIRPATH, LOG_FILENAME)
# FORMATTER = logging.Formatter(
#     '%(asctime)s %(module)s [%(levelname)s] : %(message)s',
#     datefmt='%Y/%m/%d %H:%M:%S')

# If logging folder is missing, create it
if not os.path.isdir(os.path.join(initdir, LOG_DIRPATH)):
    os.mkdir(os.path.join(initdir, LOG_DIRPATH))

# Initializes logger for all the modules
MAIN_LOGGER = logging.getLogger("cns_main")  # Eddig OK

config_file = pathlib.Path("common\logging_config.json")
with open(config_file, "r") as f:
    config = json.load(f)
logging.config.dictConfig(config)


####
# MAIN_LOGGER.setLevel(logging.DEBUG)
# # Console handler
# console_handler = logging.StreamHandler(sys.stdout)
# console_handler.setLevel(logging.DEBUG)
# console_handler.setFormatter(FORMATTER)
# MAIN_LOGGER.addHandler(console_handler)
#
# # File handler, no upper limit, max. 5 run is logged
# file_handler = RotatingFileHandler(LOG_FILE, maxBytes=0, backupCount=5)
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(FORMATTER)
# file_handler.doRollover()  # Rolls log at each start
# MAIN_LOGGER.addHandler(file_handler)
MAIN_LOGGER.info("Main logger created!")


# Modul-teszt
if __name__ == "__main__":
    pass
