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
import sys
import time
import math as m

# Third party imports
import numpy as np

# Local application imports

# Class initializations and global variables
golden_ratio = (1 + 5 ** 0.5) / 2


# Class and function definitions

def calculate_coordinates(outside_radius):

    print(f"{outside_radius=}")

    # Define ratio properly (or write approximation algorhitm)
    a = outside_radius * 4 / ( m.sqrt( (58+18*m.sqrt(5)) ) )
    print(f"{a=}")


    # # Using the given radius, parameter 'a' can be calculated
    # vector1 = np.array([golden_ratio,aaa,0])
    # vector2 = np.array([0,golden_ratio,aaa])
    # print(f"{vector1=}")
    # print(f"{vector2=}")

    # # Calculate vector to truncated icosahedron's edge node
    # vector12 = (vector2 - vector1) * (1/3)
    # print(f"{vector12=}")

    # # Calculate radius of truncated icosahedron
    # vector3 = vector1 + vector12
    # radius = np.linalg.norm(vector3)
    # print(f"{vector3=}")
    # print(f"{radius=}")

    # print(radius/outside_radius)
    


    



    coordinates = []


    if outside_radius > 0:
        pass

    return coordinates


# Main function for module testing
def main():

    print(calculate_coordinates(100))


# Include guard
if __name__ == '__main__':
    main()
