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
import logging
from enum import StrEnum

# Local application imports

# Class initializations and global variables
logger = logging.getLogger(__name__)
gravitational_constant: float = 6.67430 * pow(10, -11)  # m^3 kg-1 s-2


# Class and function definitions
class AsteriodType(StrEnum):
    """ Describes the asteroid by composition. """
    CARBON = "Carbon"
    METAL = "Metal"
    SILICONE = "Silicone"


class Component:

    def __init__(self, material: str, percentage: float):
        self.material = material
        self.percentage = percentage


class Composition:

    def __init__(self, composition: list[Component]):
        self.composition = composition

    def validate_composition(self):
        total = 0.0
        for component in self.composition:
            total += component.percentage

        if total < 100.0:
            self.composition.append(Component("Other", 100.0-total))
        elif total > 100.0:
            print("Wrong data")
        else:
            print("Data is valid")




# Include guard
if __name__ == '__main__':
    pass
