# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" This module describes the rocket hardware.

Defines a rocket Stage class and a Rocket class, as well as utility classes for
inner functions.

Libs
----

Help
----
* https://en.wikipedia.org/wiki/SpaceX_Merlin
* https://www.spacex.com/vehicles/falcon-heavy/
* https://aerospaceweb.org/question/aerodynamics/q0231.shtml
* https://spaceflight101.com/spacerockets/falcon-9-ft/
* https://en.wikipedia.org/wiki/Falcon_9#Design

Contents
--------
"""

# Standard library imports
import logging
from typing import Union, Any
from enum import Enum
import numpy as np
import math as m

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
        self.stage_thrust_N = thrust_per_engine_N * number_of_engines
        self.specific_impulse = specific_impulse  # s
        self.onboard = True

    def get_thrust(self) -> float:
        """ Returns thrust force (Newton), if there is any fuel left in the
        stage to generate it.
        """
        if self._propellant_mass_kg > 0:
            return self.stage_thrust_N  # N aka kg/m/s
        logger.warning("Fuel tanks empty!")
        return 0.0

    def get_mass(self) -> float:
        """ Returns the actual total mass (kg) of the stage. """
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
                        pressure_ratio, [0, 1], self.specific_impulse
                    )
            )

        logger.warning("Specific impulse is not in the expected format (float"
                       "or list of floats)!")
        return 0.0

    def burn_mass(self, mass_kg: float, time: Any) -> None:
        """ Burn the given amount of fuel. Logs the percentage of fuel left in
        the tanks.

        :param float mass_kg: Mass of the burnt fuel in kg.
        :param Any time: Time for logging purposes only.
        """
        self._propellant_mass_kg = max(
            0.0, self._propellant_mass_kg - abs(mass_kg)
        )
        logger.debug("Fuel left %.3f%% at (%s s)",
                     self._get_propellant_percentage(), time)


class Rocket:
    """ Rocket class, which describes a multi-stage rocket. """

    def __init__(self, name: str, stages: list[Stage],
                 fairing_mass_kg: float,
                 coefficient_of_drag: float, diameter_m: float):

        self.name = name
        self.stages = stages
        self._stage_status: RocketEngineStatus = RocketEngineStatus.STAGE_0

        # Mass properties
        self._payload_mass_kg = 0.0
        self.fairing_mass_kg = fairing_mass_kg
        self.total_mass = self.get_total_mass_kg()  # Mass without payload!

        # Drag coefficient (-) times cross-sectional area of rocket (m2)
        self.drag_constant = coefficient_of_drag * (
                m.pi * pow(diameter_m, 2) / 4)

    def staging_event_1(self) -> float:
        """ Sets first stage 'onboard' attribute to False, and returns the
        recalculated total mass of the rocket.
        """
        self.stages[0].onboard = False
        return self.get_total_mass_kg()

    def fairing_jettison_event(self) -> float:
        """ Sets 'fairing_mass_kg' attribute to 0.0, and returns the
        recalculated total mass of the rocket.
        """
        self.fairing_mass_kg = 0.0
        return self.get_total_mass_kg()

    def set_stage_status(self, status: RocketEngineStatus):
        self._stage_status = status

    def get_stage_status(self) -> RocketEngineStatus:
        return self._stage_status

    def set_payload_mass_kg(self, mass_kg: float):
        """ Sets payload mass (kg). """
        self._payload_mass_kg = mass_kg
        self.total_mass = self.get_total_mass_kg()

    def get_payload_mass(self) -> float:
        """ Gets payload mass (kg). """
        return self._payload_mass_kg

    def get_total_mass_kg(self) -> float:
        """ Calculates the actual total mass of the rocket, when called. """
        return (self._get_stage_mass_kg() + self.fairing_mass_kg
                + self._payload_mass_kg)

    def _get_stage_mass_kg(self) -> float:
        """ Returns the sum of the masses (kg) of each rocket stage,
        depending on actual staging.
        """
        mass = 0
        for stage in self.stages:
            if stage.onboard:
                mass += stage.get_mass()
        return mass

    def get_thrust(self) -> float:
        """ Calculates actual thrust (force (N)) of the rocket, depending on
        actual staging.
        """
        # TODO: simplify this, by using one statement -> remove if-else
        if self._stage_status == RocketEngineStatus.STAGE_1_BURN:
            return self.stages[0].get_thrust()

        if self._stage_status == RocketEngineStatus.STAGE_2_BURN:
            return self.stages[1].get_thrust()

        return 0

    def get_isp(self, pressure_ratio: float) -> float:
        """ Calculates actual specific impulse of the rocket, depending on
        actual staging and pressure ratio.

        :param float pressure_ratio: Ratio of sea-level and current pressure.
        """
        # TODO: simplify this, by using one statement -> remove if-else
        if self._stage_status == RocketEngineStatus.STAGE_1_BURN:
            return self.stages[0].get_specific_impulse(pressure_ratio)

        if self._stage_status == RocketEngineStatus.STAGE_2_BURN:
            return self.stages[1].get_specific_impulse(pressure_ratio)

        # NOTE: when engine is not generating thrust, isp is not valid, but 1
        #  is returned to avoid ZeroDivisionError and NaN values
        return 1


# TODO: Add here rocket hardware specs
# Vulcan
# Starship

FALCON9_1ST = Stage(25600, 395700, 9,
                    934e3, [312, 283])
FALCON9_2ND = Stage(2000, 92670, 1,
                    934e3, 348)

FALCON9 = Rocket(
    "Falcon 9 Block 5", [FALCON9_1ST, FALCON9_2ND],
    1900, 0.25, 5.2,
)

# Merlin 1D values: thrust: 845 kN, vac.: 981 kN, 311, and 282 sec spec. impulse

# Include guard
if __name__ == '__main__':
    FALCON9_1ST = Stage(25600, 395700, 9,
                        934e3, [312, 283])

    for i in range(0, 101):
        print(FALCON9_1ST.get_specific_impulse(i/100))
