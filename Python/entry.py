import json
import logging
import common.logger_setup
import common.launch_OCI

common.logger_setup.config_logger()

with open("database/config_settings.json", "r", encoding="utf-8") as f:
    config = json.load(f)

__author__: str = f"{ config['author'] }"
__version__: str = (f"{config['program_build']}."
                    f"{config['program_version_major']}."
                    f"{config['program_version_minor']}."
                    f"{config['program_version_patch']}")

logging.info("Program version: %s, %s", __version__, __author__)

# Start main
common.launch_OCI.main()
