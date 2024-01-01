# -*- coding: utf-8 -*-
#!/usr/bin/python3

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
# First import should be the logging module if any!
import datetime

# Third party imports

# Local application imports
from logger import MAIN_LOGGER as l

# Class initializations and global variables

# Class and function definitions

def julian_date(year, month, day, hour, minute, second=0):
    # https://stackoverflow.com/questions/13943062/extract-day-of-year-and-julian-day-from-a-string-date
    # https://en.wikipedia.org/wiki/Julian_day
    """ Calculates full Julian date of a given date according to:
    JD = JDN + hour/24 + minute/1440 + second/86400
    """
    jdn = datetime.date(year,month,day).toordinal() + 1721424.5
    jd_ = jdn + hour/24 + minute/1440 + second/86400
    l.debug(f"Julian date number of {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d} is: {jd_}")
    return jd_


def j2000_date(year, month, day, hour, minute, second=0):
    # https://en.wikipedia.org/wiki/Epoch_(astronomy)
    """ Calculates J200 date of a given date from Julian date. """
    julian_d = julian_date(year, month, day, hour, minute, second)
    j2000 = 2000 + (julian_d - 2451545.0) / 365.25
    l.debug(f"J2000 date of {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d} is: {j2000}")
    return j2000


# Main function for module testing
def main():
    """  """
    julian_date(2000,1,1,12,00)
    j2000_date(2000,1,1,12,00)


# Include guard
if __name__ == '__main__':
    main()
