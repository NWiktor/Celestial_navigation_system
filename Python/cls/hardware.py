# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" This module describes the rocket hardware classes.

Libs
----

Help
----

Contents
--------
"""

# Standard library imports
import logging
from typing import Union
from enum import Enum
import numpy as np

# Local application imports

# Class initializations and global variables
logger = logging.getLogger(__name__)

# Add here rocket hardware specs
# Falcon
# Vulcan
# Starship


class Stage:
    """ Rocket stage class, defined by empty mass, propellant mass, number of
    engines, engine thrust and specific impulse.
    """

    def __init__(self, empty_mass: float, propellant_mass: float,
                 number_of_engines: int, thrust_per_engine: float,
                 specific_impulse: Union[float, list[float]]):
        self._empty_mass = empty_mass  # kg
        self._propellant_mass0 = propellant_mass  # kg
        self._propellant_mass = propellant_mass  # kg
        self.stage_thrust = thrust_per_engine * number_of_engines
        self.specific_impulse = specific_impulse  # s
        self.onboard = True

    def get_thrust(self) -> float:
        """ Returns thrust, if there is any fuel left in the stage to generate
        it.
        """
        if self._propellant_mass > 0:
            return self.stage_thrust  # N aka kg/m/s
        return 0.0

    def get_mass(self) -> float:
        """ Returns the actual total mass of the stage. """
        return self._empty_mass + self._propellant_mass

    def get_propellant_percentage(self) -> float:
        """ Returns the percentage of fuel left. """
        return self._propellant_mass / self._propellant_mass0 * 100

    def get_specific_impulse(self, pressure_ratio: float = 0.0) -> float:
        """ Returns specific impulse value.

        If the class initiated with a list for specific impulse, this function
        can compensate atmospheric pressure change by the pressure ratio: (0.0
        is sea-level, 1.0 is vacuum pressure). If instead a float is given, this
        is omitted.
        """
        # If only one value is given, it is handled as a constant
        if isinstance(self.specific_impulse, int):
            return self.specific_impulse

        # If a list is given, linear interpolation is used
        if isinstance(self.specific_impulse, list):
            return float(
                    np.interp(
                        pressure_ratio,
                        [0, 1], self.specific_impulse
                    )
            )

        logger.warning("Specific impulse is not in the expected format (float"
                       "or list of floats)!")
        return 0.0

    def burn_mass(self, mass: float, time) -> None:
        """ Burn the given amount of fuel.
        Returns the percentage of fuel left in the tanks.
        """
        self._propellant_mass = max(0.0, self._propellant_mass - abs(mass))
        logger.debug("Fuel left %.2f %% at (%s s)",
                     self.get_propellant_percentage(), time)


class RocketEngineStatus(Enum):
    """ Describes the status of the rocket engine during liftoff. """
    STAGE_0 = 0
    STAGE_1_BURN = 1
    STAGE_1_COAST = 10
    STAGE_2_BURN = 2
    STAGE_2_COAST = 20
    STAGE_3_BURN = 3
    STAGE_3_COAST = 30
    STAGE_4_BURN = 4
    STAGE_4_COAST = 40


class RocketAttitudeStatus(Enum):
    """ Describes the status of the rocket attitude control programs during
    liftoff.
    """
    VERTICAL_FLIGHT = 0
    ROLL_PROGRAM = 1
    PITCH_PROGRAM = 2
    GRAVITY_ASSIST = 3


# Include guard
if __name__ == '__main__':
    pass
