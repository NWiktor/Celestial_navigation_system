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
* https://currentmillis.com/?now

Contents
--------
"""

# Standard library imports
import datetime
import logging
import math

# Third party imports
# Local application imports

# Class initializations and global variables
logger = logging.getLogger(__name__)


def julian_date(date: datetime.datetime) -> float:
    """ Calculates full Julian date of a given date according to (in days):
    JD = JDN + hour/24 + minute/1440 + second/86400

    https://stackoverflow.com/questions/13943062/extract-day-of-year-and-julian-day-from-a-string-date
    https://en.wikipedia.org/wiki/Julian_day
    """
    jdn = date.toordinal() + 1721424.5
    jd_days = jdn + date.hour / 24 + date.minute / 1440 + date.second / 86400
    logging.debug(f"Julian date number of {date.year}-{date.month:02d}-{date.day:02d} "
                  f"{date.hour:02d}:{date.minute:02d}:{date.second:02d} is: {jd_days}")
    return jd_days


def j2000_date(date: datetime.datetime) -> float:
    """ Calculates the elapsed time since the J2000 epoch in years.

    https://en.wikipedia.org/wiki/Epoch_(astronomy)
    """
    julian_d = julian_date(date)  # Julian date in days
    # j2000_days = (julian_d - 2451545.0)
    j2000_years = (julian_d - 2451545.0) / 365.25  # elapsed days since epoch (years)
    logging.info(f"J2000 date of {date.year}-{date.month:02d}-"
                 f"{date.day:02d} is: J2000+{j2000_years}")
    return j2000_years


def gregorian_date(j2000_date_years: float) -> datetime.datetime:
    """ Calculates the Gregorian date from J2000 date (years). """

    _julian_date = (j2000_date_years * 365.25) + 2451545.0
    jd_days = _julian_date - 1721424.5  # Calculate julian days
    date = datetime.datetime.fromordinal(math.floor(jd_days))  # Create date

    hours = (jd_days % 1) * 24
    minutes = (hours % 1) * 60
    seconds = (minutes % 1) * 60
    # microseconds = (seconds % 1) * 1000

    # Create time params separately, bc. date doesnt support float
    time = datetime.time(hour=int(hours), minute=int(minutes),
                         second=int(seconds),
                         # microsecond=int(microseconds)
                         )

    return datetime.datetime.combine(date, time)


def secs_to_mins(total_seconds: int) -> str:
    """ Formats seconds to HH:MM:SS format. """
    total_minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(total_minutes, 60)
    if hours == 0:
        return f"{minutes:02d}:{seconds:02d}"
    return f"{hours}:{minutes:02d}:{seconds:02d}"


# Include guard
if __name__ == '__main__':
    print(datetime.datetime.now())
    a = j2000_date(datetime.datetime.now())
    b = julian_date(datetime.datetime.now())
    print(a)
    print(b)

    c = gregorian_date(a)
    print(c)
