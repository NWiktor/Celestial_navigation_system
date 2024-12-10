# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" This module describes the rocket hardware.

Defines a rocket Stage class and a Rocket class, as well as utility classes for
inner functions.

Libs
----

Help
----

Contents
--------
"""

# Standard library imports
import logging
from typing import Union, Any
from enum import Enum
import numpy as np

# Local application imports

# Class initializations and global variables
logger = logging.getLogger(__name__)


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


# TODO: test specific impulse values
class Stage:
    """ Rocket stage class, defined by empty mass, propellant mass, number of
    engines, engine thrust and specific impulse.

    :param float empty_mass_kg: Empty (dry) mass of the stage.
    :param float propellant_mass_kg: Propellant mass of the stage.
    :param int number_of_engines: Number of engines in the stage.
    :param float thrust_per_engine_N: Thrust_per_engine in Newtons.
    :param specific_impulse: Specific impulse of the stage.
    """
    def __init__(self, empty_mass_kg: float, propellant_mass_kg: float,
                 number_of_engines: int, thrust_per_engine_N: float,
                 specific_impulse: Union[float, list[float]]):
        self._empty_mass_kg = empty_mass_kg
        self._propellant_mass0_kg = propellant_mass_kg
        self._propellant_mass_kg = propellant_mass_kg
        self.stage_thrust = thrust_per_engine_N * number_of_engines
        self.specific_impulse = specific_impulse  # s
        self.onboard = True

    def get_thrust(self) -> float:
        """ Returns thrust, if there is any fuel left in the stage to generate
        it.
        """
        if self._propellant_mass_kg > 0:
            return self.stage_thrust  # N aka kg/m/s
        return 0.0

    def get_mass(self) -> float:
        """ Returns the actual total mass of the stage. """
        return self._empty_mass_kg + self._propellant_mass_kg

    def _get_propellant_percentage(self) -> float:
        """ Returns the percentage of fuel left. """
        return self._propellant_mass_kg / self._propellant_mass0_kg * 100

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

    def burn_mass(self, mass_kg: float, time: Any) -> None:
        """ Burn the given amount of fuel. Logs the percentage of fuel left in
        the tanks.

        :param float mass_kg: Mass of the burnt fuel in kg.
        :param time: Time for logging purposes only.
        """
        self._propellant_mass_kg = max(
            0.0, self._propellant_mass_kg - abs(mass_kg)
        )
        logger.debug("Fuel left %.3f%% at (%s s)",
                     self._get_propellant_percentage(), time)


class Rocket:

    def __init__(self):
        pass

# TODO: Add here rocket hardware specs
# Falcon
# Vulcan
# Starship




""" Falcon 9 hardware specs:

Note: 2nd stage empty mass minus payload fairing

Sorurces:
* https://aerospaceweb.org/question/aerodynamics/q0231.shtml
* https://spaceflight101.com/spacerockets/falcon-9-ft/
* https://en.wikipedia.org/wiki/Falcon_9#Design
"""
FALCON9_1ST = Stage(25600, 395700, 9,
                    934e3, [312, 283])
FALCON9_2ND = Stage(2000, 92670, 1,
                    934e3, 348)

# Merlin 1D values: thrust: 845 kN, vac.: 981 kN, 311, and 282 sec spec. impulse

# Include guard
if __name__ == '__main__':
    pass
