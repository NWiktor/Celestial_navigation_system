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
# First import should be the logging module if any!
from dataclasses import dataclass
import math as m
from typing import Union
from enum import Enum

# Third party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from logger import MAIN_LOGGER as L


@dataclass
class Engine:
    """ Rocket engine class, defined by name, thrust and specific_impulse. """

    def __init__(self, name: str, thrust: float, specific_impulse: Union[int, list[int]]):
        self.name = name
        self.thrust = thrust  # N aka kg/m/s
        self._specific_impulse = specific_impulse  # s

    def specific_impulse(self, ratio: float = 0.0) -> float:
        """ Returns the specific impulse.

        If only one (float) value is given, returns it.

        If a list is given, linearly interpolates between them, using the ratio.
        Ratio 0.0 returns first value, ratio 1.1 returns second value.
        This feature allows value-corrections depending on the external pressure.
        https: // en.wikipedia.org / wiki / Atmospheric_pressure
        """
        if isinstance(self._specific_impulse, int):
            return self._specific_impulse

        if isinstance(self._specific_impulse, list):
            return self._specific_impulse[0] + (self._specific_impulse[1] - self._specific_impulse[0]) * ratio

        return 0


@dataclass
class Stage:
    """ Rocket stage class, defined by engine, empty mass, propellant mass, number of engines and burn duration. """

    def __init__(self, engine: Engine, empty_mass: float, propellant_mass: float,
                 number_of_engines: int, burn_duration: int):
        self.engine = engine
        self._empty_mass = empty_mass  # kg
        self._propellant_mass = propellant_mass  # kg
        self.number_of_engines = number_of_engines
        self.burn_duration = burn_duration  # s
        self.stage_thrust = self.engine.thrust * self.number_of_engines

    def get_thrust(self) -> float:
        """ Returns thrust, if there is any fuel left in the stage to generate it. """
        if self._propellant_mass > 0:
            return self.stage_thrust  # N aka kg/m/s
        return 0.0

    def get_mass(self):
        """ Returns the actual total mass of the stage. """
        return self._empty_mass + self._propellant_mass

    def burn_mass(self, standard_gravity):
        """ Reduces propellant mass, by calculating the proper delta-m after burning for 1 s. """
        delta_m = self.stage_thrust / (self.engine.specific_impulse() * standard_gravity)

        # Updates itself with new mass, negative values are omitted
        self._propellant_mass = max(0.0, self._propellant_mass - delta_m)


class RocketStatus(Enum):
    """ Describes the status of the rocket during liftoff. """
    STAGE_0 = 0
    STAGE_1_BURN = 1
    STAGE_1_COAST = 11
    STAGE_2_BURN = 2
    STAGE_2_COAST = 22


@dataclass
class PlanetLocation:
    """ Launch site class, given by longitude, latitude, surface radius (where the site is located),
    atmospheric density via the get_density function, gravity and gravitational parameter.
    """

    def __init__(self, name, latitude: float, longitude: float, surface_radius: float,
                 std_gravity, std_gravitational_parameter):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.surface_radius = surface_radius  # m
        self.std_gravity = std_gravity  # m/s^2
        self.std_gravitational_parameter = std_gravitational_parameter  # m^3/s^2

    # pylint: disable = unused-argument
    def get_density(self, altitude) -> float:
        """ Placeholder function for override. """
        return 0.0


@dataclass
class EarthLocation(PlanetLocation):
    """ Launch site class for locations on Earth's surface. """

    def __init__(self, name, latitude: float, longitude: float):
        super().__init__(f"{name}, Earth", latitude, longitude, 6371000, 9.80665,
                         3.986004418e14)

    def get_density(self, altitude: float) -> float:
        """ Approximates air density in function of height on Earth, measured from sea level.
        https://en.wikipedia.org/wiki/Density_of_air
        """
        if 0 <= altitude <= 120000:
            return 1.204 * m.exp(-altitude / 10400)
        return 0


class SpaceCraft:
    """ Spacecraft class, defined by name, payload mass, drag coefficient and diameter; and stages. """

    def __init__(self, name, payload_mass: float, coefficient_of_drag: float, diameter: float, stages: list):
        self.name = name
        self.stage_status = RocketStatus.STAGE_0
        self.stages = stages

        # Physical properties
        # Drag coefficient (-) times cross-sectional area of rocket (m2)
        self.drag_constant = coefficient_of_drag * (m.pi * pow(diameter, 2) / 4)

        # State variables / dynamic vectors and mass
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])

        # Mass properties
        self.payload_mass = payload_mass
        self.total_mass = self.payload_mass + self.get_stage_mass()

    def get_stage_mass(self):
        """ Sums the mass of each rocket stage, depending on actual staging. """

        if self.stage_status in (RocketStatus.STAGE_0, RocketStatus.STAGE_1_BURN, RocketStatus.STAGE_1_COAST):
            return self.stages[0].get_mass() + self.stages[1].get_mass()

        if self.stage_status in (RocketStatus.STAGE_2_BURN, RocketStatus.STAGE_2_COAST):
            return self.stages[1].get_mass()

        return 0

    def thrust(self):
        """ Calculates actual thrust (force) of the rocket, depending on actual staging. """

        if self.stage_status == RocketStatus.STAGE_1_BURN:
            return self.stages[0].get_thrust()

        if self.stage_status == RocketStatus.STAGE_2_BURN:
            return self.stages[1].get_thrust()

        return 0

    def update_mass(self, standard_gravity):
        """ Updates total rocket mass after burning, depending on gravity and actual staging. """

        # Calculate delta m in stage 1
        if self.stage_status == RocketStatus.STAGE_1_BURN:
            self.stages[0].burn_mass(standard_gravity)

        # Calculate delta m in stage 2
        elif self.stage_status == RocketStatus.STAGE_2_BURN:
            self.stages[1].burn_mass(standard_gravity)

        # Calculate new total mass
        self.total_mass = self.payload_mass + self.get_stage_mass()

    def launch(self, launch_site: PlanetLocation, meco, seco):
        """ Yield rocket's status variables during launch, every second. """

        # MECO can't be later than stage 1 burn duration
        meco_time = min(meco, self.stages[0].burn_duration)
        stage_separation = meco_time + 8
        second_stage_ignition = meco_time + 14
        L.debug("MAIN ENGINE CUT OFF at T+%s", seconds_to_minutes(meco_time))

        # SECO can't be later than stage 1 and 2 total burn duration, but it can't be earlier than second stage ignition
        seco_time = max(min(seco, self.stages[0].burn_duration + self.stages[1].burn_duration), second_stage_ignition)
        L.debug("SECOND ENGINE CUT OFF at T+%s", seconds_to_minutes(seco_time))

        # Start calculation
        time = int(seco_time * 3)  # Total time for loop
        yield self.position, self.velocity, self.acceleration, self.total_mass, 0, 0, 9.81  # Yield initial values

        for i in range(0, time):
            #  Calculate stage status according to time
            if i <= meco_time:
                self.stage_status = RocketStatus.STAGE_1_BURN
            elif meco_time < i <= stage_separation:
                self.stage_status = RocketStatus.STAGE_1_COAST
            elif second_stage_ignition < i <= seco_time:
                self.stage_status = RocketStatus.STAGE_2_BURN
            else:
                self.stage_status = RocketStatus.STAGE_2_COAST

            # Calculate flight characteristics
            air_density = launch_site.get_density(self.position[2])
            distance_from_barycenter = self.position[2] + launch_site.surface_radius
            thrust = self.thrust()
            drag = self.drag_constant * air_density * pow(self.velocity[2], 2) / 2
            gravity = launch_site.std_gravitational_parameter / pow(distance_from_barycenter, 2)

            L.debug("Rocket altitude is %s m", self.position[2])
            L.debug("Thrust is %s N", thrust)
            L.debug("Rocket total mass is %s kg", self.total_mass)
            L.debug("Air density is %s kg/m3", air_density)
            L.debug("Drag force is %s N", drag)

            # Calculate position, velocity and acceleration
            self.acceleration[2] = (thrust - drag) / self.total_mass - gravity
            self.velocity[2] += self.acceleration[2]
            self.position[2] += self.velocity[2]

            # Calculate new spacecraft mass
            self.update_mass(launch_site.std_gravity)

            yield self.position, self.velocity, self.acceleration, self.total_mass, thrust, drag, gravity


def seconds_to_minutes(total_seconds) -> str:
    """ Formats seconds to HH:MM:SS format. """
    total_minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(total_minutes, 60)
    if hours == 0:
        return f"{minutes:02d}:{seconds:02d}"
    return f"{hours}:{minutes:02d}:{seconds:02d}"


# Main function for module testing
# pylint: disable=too-many-statements, too-many-locals
def main():
    """ Defines a Spacecraft and LaunchSite classes, then calculates and plots flight parameters during liftoff. """
    # Launch-site
    cape = EarthLocation("Cape Canaveral", 28.3127, 80.3903)

    # Falcon9 hardware specs:
    # https://aerospaceweb.org/question/aerodynamics/q0231.shtml
    # https://en.wikipedia.org/wiki/Falcon_9#Design
    merlin1d_p = Engine("Merlin 1D+", 934e3, [283, 312])
    merlin1d_vac = Engine("Merlin 1D vac", 934e3, 348)
    first_stage = Stage(merlin1d_p, 25600, 395700, 9, 162)
    second_stage = Stage(merlin1d_vac, 3900, 92670, 1, 397)
    falcon9 = SpaceCraft("Falcon 9", 22800, 0.25, 5.2, [first_stage, second_stage])

    # Launch
    i = 0
    time_limit = 1400
    time_data = []
    alt_data = []
    vel_data = []
    acc_data = []
    mass_data = []
    thrust_data = []
    drag_data = []
    gravity_data = []

    for p, v, a, mass, t, d, g in falcon9.launch(cape, 130, 465):
        time_data.append(i)
        alt_data.append(p[2]/1000)
        vel_data.append(v[2]/1000)
        acc_data.append(a[2]/9.81)
        mass_data.append(mass/1000)
        thrust_data.append(t/1000)
        drag_data.append(d/1000)
        gravity_data.append(g)
        i += 1

    # Plotting
    plt.style.use('_mpl-gallery')

    fig = plt.figure(layout='constrained', figsize=(19, 9.5))
    fig.suptitle("Falcon9 launch from Cape Canaveral")
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.set_title("Altitude (z)")
    ax1.set_ylabel("Altitude (km)")
    ax1.set_xlim(0, time_limit)
    ax1.set_ylim(0, 3000)
    ax1.scatter(time_data, alt_data, s=0.5)

    ax2 = fig.add_subplot(4, 2, 2)
    ax2.set_title("Velocity (z)")
    ax2.set_ylabel("Velocity (km/s)")
    ax2.set_xlim(0, time_limit)
    ax2.set_ylim(0, 8)
    ax2.scatter(time_data, vel_data, s=0.5)

    ax3 = fig.add_subplot(4, 2, 3)
    ax3.set_title("Acceleration (z)")
    ax3.set_ylabel("Acceleration (g)")
    ax3.set_xlim(0, time_limit)
    ax3.set_ylim(-2.5, 7.5)
    ax3.scatter(time_data, acc_data, s=0.5)

    ax4 = fig.add_subplot(4, 2, 4)
    ax4.set_title("Spacecraft mass")
    ax4.set_ylabel("Mass (ton)")
    ax4.set_xlim(0, time_limit)
    ax4.set_ylim(0, 600)
    ax4.scatter(time_data, mass_data, s=0.5)

    ax5 = fig.add_subplot(4, 2, 5)
    ax5.set_title("Thrust")
    ax5.set_ylabel("Force (kN)")
    ax5.set_xlim(0, time_limit)
    # ax5.set_ylim(0, 80)
    ax5.scatter(time_data, thrust_data, s=0.5)

    ax6 = fig.add_subplot(4, 2, 6)
    ax6.set_title("Drag")
    ax6.set_ylabel("Force (kN)")
    ax6.set_xlim(0, time_limit)
    # ax6.set_ylim(0, 3)
    ax6.scatter(time_data, drag_data, s=0.5)

    ax7 = fig.add_subplot(4, 2, 7)
    ax7.set_title("Gravity")
    ax7.set_ylabel("Force (m/s2)")
    ax7.set_xlim(0, time_limit)
    ax7.set_ylim(4, 10)
    ax7.scatter(time_data, gravity_data, s=0.5)

    plt.show()


# Include guard
if __name__ == '__main__':
    main()
