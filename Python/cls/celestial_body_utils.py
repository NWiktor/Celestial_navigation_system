# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" This is a collection of small classes and functions for the CelestialBody
class.

Help
----
* Asteroid (wikipedia)

Contents
--------
"""

# Standard library imports
import logging
from enum import StrEnum

# Local application imports

# Class initializations and global variables
logger = logging.getLogger(__name__)


# Class and function definitions
class AsteriodType(StrEnum):
    """ Describes the asteroid types by composition. """
    CARBON = "Carbon"
    METAL = "Metal"
    SILICONE = "Silicone"


class Component:
    """ Dataclass for material and percentage value.

    Raises ValueError if the percentage is not between 0 and 100.
    """
    def __init__(self, material: str, percentage: float,
                 material_short: str = None):
        self.material = material
        self.material_short = material_short

        if 0.0 <= percentage <= 100.0:
            self.percentage = percentage
        else:
            raise ValueError


class Composition:
    """ Class for representing an object's composition. A list of Component
    objects (material and percentage value pairs).
    """

    def __init__(self, composition: list[Component],
                 *, source: str | list[str] = None):
        self.composition = composition
        self.source = source
        self.check_total_percentage()

    def check_total_percentage(self) -> None:
        """ Check the sum of the components percentages.

        If it is less than 100%, adds an 'other' component. If it more than
        100%, raises ValueError.
        """
        total = 0.0
        for component in self.composition:
            total += component.percentage

        if total < 100.0:
            self.composition.append(Component("Other", 100.0-total))
        elif total > 100.0:
            raise ValueError(f"Total percentage is above 100%: {total}%")
        else:
            print("Data is valid")

    def get_composition(self) -> list[str]:
        """ Returns the list of components."""
        comp_list = ["Chemical composition:"]

        for component in self.composition:
            if component.material_short is not None:
                short_name = f" ({component.material_short})"
            else:
                short_name = ""

            comp_list.append(f"{component.material}{short_name}"
                             f" - {component.percentage:.3f}%")
        return comp_list


# Include guard
if __name__ == '__main__':
    pass
