# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" Logger modul for other programs/apps.

This module consist of one function, the 'config_logger', which configures
the root logger by the 'logging_config.json' file.

**IMPORTANT: THE DEFAULT TWO HANDLER MUST BE UNCHANGED!**
The 'file' handler filepath is programatically modified to match the indented
location independent of the installation location. Also, the 'file' handler
executes a rollover at every start. When modifying the config file, this two
handler and their order must be unchanged because it is hardcoded in this
script.

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
* https://stackoverflow.com/questions/404744/determining-application-path-in-a-\
python-exe-generated-by-pyinstaller

Contents
--------
"""


import logging.config
import os
import sys
import json
from pathlib import Path


INITDIR = Path(__file__).parents[0]
# INITDIR = os.path.dirname(os.path.dirname(__file__))

# Set defaults
LOG_FILENAME = Path("log.txt")
LOG_DIRNAME = Path("log")
LOG_PATH = Path.joinpath(INITDIR, LOG_DIRNAME, LOG_FILENAME)


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
    Every child logger will inherit the root logger configs, so no need to do
    this again.
    """
    # If logging folder is missing, create it
    folderpath = Path.joinpath(INITDIR, LOG_DIRNAME)
    if not folderpath.is_dir():
        folderpath.mkdir()

    # Configure root logger
    config_file = Path.joinpath(INITDIR, Path("database/logging_config.json"))
    with open(config_file, "r", encoding="utf-8") as file:
        config = json.load(file)

    # Override log's path by absolute path
    config["handlers"]["file"]["filename"] = LOG_PATH
    logging.config.dictConfig(config)  # Set configs to root logger
    # Rollover file logger aka create new file at every restart
    logging.getLogger("root").handlers[1].doRollover()

    # Initializes logger for this module
    logger = logging.getLogger(__name__)
    logger.info("Main logger created!")
    logger.info("Log file path: %s", LOG_PATH)


# Modul-teszt
if __name__ == "__main__":
    config_logger()
