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
from modules import ode_solvers as mch

# Local application imports
from logger import MAIN_LOGGER as L


@dataclass
class Stage:
    """ Rocket stage class, defined by engine thrust, specific impulse, empty mass, propellant mass,
    number of engines and burn duration.
    """

    def __init__(self, empty_mass: float, propellant_mass: float, number_of_engines: int, thrust_per_engine: float,
                 specific_impulse: Union[int, list[int]]):
        self._empty_mass = empty_mass  # kg
        self._propellant_mass = propellant_mass  # kg
        self.stage_thrust = thrust_per_engine * number_of_engines
        self.specific_impulse = specific_impulse  # s

    def get_thrust(self) -> float:
        """ Returns thrust, if there is any fuel left in the stage to generate it. """
        if self._propellant_mass > 0:
            return self.stage_thrust  # N aka kg/m/s
        return 0.0

    def get_mass(self):
        """ Returns the actual total mass of the stage. """
        return self._empty_mass + self._propellant_mass

    def burn_mass(self, standard_gravity, pressure_ratio: float = 0.0, duration: float = 1.0):
        """ Reduces propellant mass, by calculating the proper delta-m after burning for a given duration (s).
        The function takes into account athmospheric pressure change by the pressure ratio:
        (0.0 is sea-level, 1.0 is vaccum pressure).
        """

        # If only one value is given, it is handled as a constant
        if isinstance(self.specific_impulse, int):
            isp = self.specific_impulse

        # If a list is given, linearly interpolating between them by the pressure-ratio
        elif isinstance(self.specific_impulse, list):
            isp_sl = min(self.specific_impulse)
            isp_vac = max(self.specific_impulse)
            isp = isp_sl + (isp_vac - isp_sl) * (1 - pressure_ratio)
        else:
            isp = 0

        # Updates itself with new mass, negative values are omitted
        delta_m = - self.stage_thrust / (isp * standard_gravity) * duration
        self._propellant_mass = max(0.0, self._propellant_mass + delta_m)


class SpaceCraftStatus(Enum):
    """ Describes the status of the rocket during liftoff. """
    STAGE_0 = 0
    STAGE_1_BURN = 1
    STAGE_1_BURN_THROTTLED = 11
    STAGE_1_COAST = 10
    STAGE_2_BURN = 2
    STAGE_2_COAST = 20
    STAGE_3_BURN = 3
    STAGE_3_COAST = 30


@dataclass
class PlanetLocation:
    """ Launch site class, given by longitude, latitude, surface radius (where the site is located),
    atmospheric density via the get_density function, gravity and gravitational parameter.
    """

    def __init__(self, name, latitude: float, longitude: float, surface_radius: float, angular_velocity: float,
                 std_gravity, std_gravitational_parameter):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.surface_radius = surface_radius  # m
        self.angular_velocity = angular_velocity  # rad/s
        self.std_gravity = std_gravity  # m/s^2
        self.std_gravitational_parameter = std_gravitational_parameter  # m^3/s^2

    # pylint: disable = unused-argument
    def get_density(self, altitude) -> float:
        """ Placeholder function for override. """
        return 0.0

    # pylint: disable = unused-argument
    def get_pressure(self, altitude) -> float:
        """ Placeholder function for override. """
        return 0.0


@dataclass
class EarthLocation(PlanetLocation):
    """ Launch site class for locations on Earth's surface. """

    def __init__(self, name, latitude: float, longitude: float):
        super().__init__(f"{name}, Earth", latitude, longitude, 6371000, 7.292115e-5,
                         9.80665, 3.986004418e14)

    def get_density(self, altitude: float) -> float:
        """ Approximates air density in function of height on Earth, measured from sea level.
        https://en.wikipedia.org/wiki/Density_of_air
        """
        if 0 <= altitude <= 120000:
            return 1.204 * m.exp(-altitude / 10400)
        return 0.0

    def get_pressure(self, altitude: float) -> float:
        altitude += 1
        return 0.0


# TODO: refactor payload as stage3
class SpaceCraft:
    """ Spacecraft class, defined by name, payload mass, drag coefficient and diameter; and stages. """

    def __init__(self, name, payload_mass: float, coefficient_of_drag: float, diameter: float, stages: list):
        self.name = name
        self.stage_status = SpaceCraftStatus.STAGE_0
        self.stages = stages

        # Physical properties
        # Drag coefficient (-) times cross-sectional area of rocket (m2)
        self.drag_constant = coefficient_of_drag * (m.pi * pow(diameter, 2) / 4)

        # State variables / dynamic vectors and mass
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # m and m/s
        self.acceleration = np.array([0.0, 0.0, 0.0])

        # Mass properties
        self.payload_mass = payload_mass
        self.total_mass = self.payload_mass + self.get_stage_mass()

    def get_stage_mass(self):
        """ Sums the mass of each rocket stage, depending on actual staging. """

        if self.stage_status in (SpaceCraftStatus.STAGE_0,
                                 SpaceCraftStatus.STAGE_1_BURN,
                                 SpaceCraftStatus.STAGE_1_COAST):
            return self.stages[0].get_mass() + self.stages[1].get_mass()

        if self.stage_status in (SpaceCraftStatus.STAGE_2_BURN, SpaceCraftStatus.STAGE_2_COAST):
            return self.stages[1].get_mass()

        return 0

    def thrust(self):
        """ Calculates actual thrust (force) of the rocket, depending on actual staging. """

        if self.stage_status == SpaceCraftStatus.STAGE_1_BURN:
            return self.stages[0].get_thrust()

        if self.stage_status == SpaceCraftStatus.STAGE_2_BURN:
            return self.stages[1].get_thrust()

        return 0

    # CAVEAT: This function should not be merged to thrust function, because it may be called multiple times during
    #  RK-4 integration, but mass should be reduced only once.
    def update_mass(self, standard_gravity, duration=1):
        """ Updates total rocket mass after burning, depending on gravity and actual staging. """

        # Calculate delta m in stage 1
        if self.stage_status == SpaceCraftStatus.STAGE_1_BURN:
            self.stages[0].burn_mass(standard_gravity, duration)

        # Calculate delta m in stage 2
        elif self.stage_status == SpaceCraftStatus.STAGE_2_BURN:
            self.stages[1].burn_mass(standard_gravity, duration)

        # Calculate new total mass
        self.total_mass = self.payload_mass + self.get_stage_mass()

    # @staticmethod
    def launch_ode(self, t, state, mu, drag_const, angular_v_earth):
        """ 2nd order ODE of the state-vectors, during launch.

        The function returns the second derivative of position vector at a given time (acceleration vector),
        using the position (r) and velocity (v) vectors. All values represent the same time.

        Trick: While passing through the velocity vector unchanged, we can numerically integrate both functions in
        the RK4-solver in one step (this is outside of this functions scope).

        State-vector: rx, ry, rz, vx, vy, vz
        State-vector_dot: vx, vy, vz, ax, ay, az
        """
        r = state[:3]  # Position vector
        v = state[3:6]  # Velocity vector
        mass = self.total_mass  # Mass

        # Unit vector of velocity
        unit_v = v / np.linalg.norm(v)

        # 2nd order ODE function (acceleration)
        if t <= 16:  # Vertical flight until tower is cleared
            a_thrust = self.thrust() / mass * (r / np.linalg.norm(r))
        elif 16 < t <= 30:  # Initial pitch-over maneuver ended
            a_thrust = self.thrust() / mass * mch.pitch_over_matrix(-0.01) * (r / np.linalg.norm(r))
        else:  # Gravity assist
            a_thrust = self.thrust() / mass * unit_v

        a_gravity = -r * mu / np.linalg.norm(r) ** 3
        a_drag = - unit_v * drag_const * np.linalg.norm(v) ** 2 / mass
        a = (a_gravity + a_thrust + a_drag + 2 * np.cross(angular_v_earth, v)
             + np.cross(angular_v_earth, np.cross(angular_v_earth, r)))

        return np.concatenate((v, a))  # vx, vy, vz, ax, ay, az

    # pylint: disable = too-many-locals
    def launch(self, launch_site: PlanetLocation, meco, ses_1, seco_1, ses_2, seco_2):
        """ Yield rocket's status variables during launch, every second. """

        # FLIGHT PROFILE
        # EXAMPLE: https://spaceflight101.com/falcon-9-ses-10/flight-profile/#google_vignette
        stage_separation = meco + 3
        L.debug("MAIN ENGINE CUT OFF at T+%s (%s)", seconds_to_minutes(meco), meco)
        L.debug("STAGE SEPARATION at T+%s (%s)", seconds_to_minutes(stage_separation), stage_separation)
        L.debug("SECOND ENGINE START 1 at T+%s (%s)", seconds_to_minutes(ses_1), ses_1)
        L.debug("SECOND ENGINE CUT OFF 1 at T+%s (%s)", seconds_to_minutes(seco_1), seco_1)
        L.debug("SECOND ENGINE START 2 at T+%s (%s)", seconds_to_minutes(ses_2), ses_2)
        L.debug("SECOND ENGINE CUT OFF 2 at T+%s (%s)", seconds_to_minutes(seco_2), seco_2)

        # Start calculation - Update state vector with initial conditions
        # https://en.wikipedia.org/wiki/Earth%27s_rotation
        x, y, z = mch.convert_spherical_to_cartesian_coords(launch_site.surface_radius,
                                                            launch_site.latitude * m.pi/180,
                                                            launch_site.longitude * m.pi/180)
        r_rocket = np.array([x, y, z])  # m
        angular_v_earth = np.array([0, 0, launch_site.angular_velocity])  # rad/s
        v_rocket = np.cross(angular_v_earth, r_rocket)
        self.state = np.concatenate((r_rocket, v_rocket))
        self.acceleration = np.array([0.0, 0.0, 0.0])
        yield self.state, self.acceleration, self.total_mass  # Yield initial values

        for i in range(0, 3500):  # Calculate stage status according to time
            if i <= meco:
                self.stage_status = SpaceCraftStatus.STAGE_1_BURN
            elif meco < i <= stage_separation:
                self.stage_status = SpaceCraftStatus.STAGE_1_COAST
            elif ses_1 < i <= seco_1 or ses_2 < i <= seco_2:
                self.stage_status = SpaceCraftStatus.STAGE_2_BURN
            else:
                self.stage_status = SpaceCraftStatus.STAGE_2_COAST

            # Calculate flight characteristics
            distance_from_surface = np.linalg.norm(self.state[0:3]) - launch_site.surface_radius
            air_density = launch_site.get_density(distance_from_surface)
            drag_const = self.drag_constant * air_density / 2

            # Calculate state-vector and acceleration
            # The ODE is solved for the acceleration vector, which is used as an initial condition for the
            # RK4 numerical integrator function.
            # Passing not only the acceleration vector, but the velocity vector to the RK4, we can numerically
            # integrate twice with one function-call, thus we get back the state-vector as well.
            self.state, self.acceleration = mch.rk4(self.launch_ode, 0, self.state, 1,
                                                    launch_site.std_gravitational_parameter,
                                                    drag_const, angular_v_earth)

            # Calculate new spacecraft mass
            # TODO: implement timestep based mass calculation
            self.update_mass(launch_site.std_gravity)

            # Log new data
            altitude_above_surface = np.linalg.norm(self.state[0:3]) - launch_site.surface_radius
            if altitude_above_surface <= 0:
                L.warning("WARNING! LITHOBRAKING!")
                break
            L.debug("Rocket state is %s", self.state)
            L.debug("Rocket altitude is %s m", altitude_above_surface)
            L.debug("Rocket total mass is %s kg", self.total_mass)
            L.debug("Air density is %s kg/m3", air_density)
            L.debug("Acceleration is %s m/s2", np.linalg.norm(self.acceleration))

            # Yield values
            yield self.state, self.acceleration, self.total_mass


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
    cape = EarthLocation("Cape Canaveral", 28.3127, -80.3903)

    # Falcon9 hardware specs:
    # https://aerospaceweb.org/question/aerodynamics/q0231.shtml
    # https://spaceflight101.com/spacerockets/falcon-9-ft/
    # https://en.wikipedia.org/wiki/Falcon_9#Design
    first_stage = Stage(25600, 395700, 9, 934e3, [283, 312])
    second_stage = Stage(3900, 107500, 1, 934e3, 348)
    falcon9 = SpaceCraft("Falcon 9", 22000, 0.25, 5.2, [first_stage, second_stage])

    # Launch
    i = 0
    time_limit = 3500
    time_data = []
    alt_data = []
    rx = []
    ry = []
    rz = []
    vel_data = []
    acc_data = []
    mass_data = []

    # MECO: 145 s
    # TODO: Modelling throttle to 80%
    # SES_1: MECO + 11s
    for state, a, mass in falcon9.launch(cape, 130, 141, 514, 3090, 3390):
        i += 1
        time_data.append(i)
        rx.append(state[0])
        ry.append(state[1])
        rz.append(state[2])
        alt_data.append((np.linalg.norm(state[0:3]) - 6371000) / 1000)  # Altitude in km-s
        vel_data.append(np.linalg.norm(state[3:6]) / 1000)  # Velocity in km/s
        acc_data.append(np.linalg.norm(a))  # Accceleration in m/s2
        mass_data.append(mass / 1000)  # Mass in 1000 kg-s

        if i == time_limit:
            break

    # Plotting
    # TODO: implement colormap for each stage of the flight
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html#sphx-glr-gallery-lines-bars-and-markers-multicolored-line-py
    plt.style.use('_mpl-gallery')

    fig = plt.figure(layout='constrained', figsize=(19, 9.5))
    fig.suptitle("Falcon9 launch from Cape Canaveral")
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Flight path")
    ax1.set_xlim(0, time_limit)
    # ax1.set_ylim(0, 8)
    ax1.scatter(time_data, alt_data, s=0.5)

    ax1.scatter(time_data, vel_data, s=0.5)

    # ax1.scatter(time_data, acc_data, s=0.5)

    # ax1.scatter(time_data, mass_data, s=0.5)

    ax5 = fig.add_subplot(1, 2, 2, projection='3d')
    ax5.plot(rx, ry, rz, label="Trajectory")

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 6371000 * np.outer(np.cos(u), np.sin(v))
    y = 6371000 * np.outer(np.sin(u), np.sin(v))
    z = 6371000 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax5.plot_surface(x, y, z)
    ax5.set_aspect('equal')
    plt.show()


# Include guard
if __name__ == '__main__':
    main()
