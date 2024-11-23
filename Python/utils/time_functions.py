# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" Summary of this code file goes here. The purpose of this module can be
expanded in multiple sentences. Below a short free-text summary of the included
classes and functions to give an overview. More detailed summary of the
functions can be provided inside the function's body.

Libs
----
* some_module - This is used for imported, non-standard modules, to help track
    dependencies. Summary is not needed.

Help
----
* https://en.wikipedia.org/wiki/Truncated_icosahedron

Contents
--------
"""

# Standard library imports
import datetime
import logging
# Third party imports
# Local application imports

# Class initializations and global variables
logger = logging.getLogger(__name__)


def julian_date(date: datetime.datetime) -> float:
    """ Calculates full Julian date of a given date according to:
    JD = JDN + hour/24 + minute/1440 + second/86400

    https://stackoverflow.com/questions/13943062/extract-day-of-year-and-julian-day-from-a-string-date
    https://en.wikipedia.org/wiki/Julian_day
    """
    jdn = date.toordinal() + 1721424.5
    jd_ = jdn + date.hour / 24 + date.minute / 1440 + date.second / 86400
    logging.debug(f"Julian date number of {date.year}-{date.month:02d}-{date.day:02d} "
                  f"{date.hour:02d}:{date.minute:02d}:{date.second:02d} is: {jd_}")
    return jd_


def j2000_date(date: datetime.datetime) -> float:
    """ Calculates the elapsed time since the J2000 epoch.

    https://en.wikipedia.org/wiki/Epoch_(astronomy)
    """
    julian_d = julian_date(date)
    j2000_years = 2000 + (julian_d - 2451545.0) / 365.25
    j2000_days = 2000 + (julian_d - 2451545.0)
    logging.debug(f"J2000 date of {date.year}-{date.month:02d}-{date.day:02d} "
                  f"{date.hour:02d}:{date.minute:02d}:{date.second:02d} is: {j2000_years}")
    return j2000_years


def secs_to_mins(total_seconds: int) -> str:
    """ Formats seconds to HH:MM:SS format. """
    total_minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(total_minutes, 60)
    if hours == 0:
        return f"{minutes:02d}:{seconds:02d}"
    return f"{hours}:{minutes:02d}:{seconds:02d}"


# Include guard
if __name__ == '__main__':
    pass
