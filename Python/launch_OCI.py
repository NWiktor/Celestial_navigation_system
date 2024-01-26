# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" This module contains all relevant class and function for orbit propagation around a celestial body. The module
calculates the trajectory of a two-stage rocket launched from surface in Object-Centered Inertial reference frame (OCI).

Libs
----
* Numpy
* Matplotlib

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
        """ Placeholder function for override by child. """
        L.error("Missing function!")
        return 0.0

    # pylint: disable = unused-argument
    def get_pressure(self, altitude) -> float:
        """ Placeholder function for override by child. """
        L.error("Missing function!")
        return 0.0

    def get_relative_velocity(self, state: np.array) -> float:
        """ Returns the speed of the rocket relative to the atmosphere.

        The atmosphere of the planet is modelled as static (no winds). The function calculates the atmospheric
        velocity (in inertial ref. frame), and substracts it from the rocket's speed in inertial frame, then takes
        the norm of the resulting vector.
        """
        return np.linalg.norm(state[3:6] - np.cross(np.array([0, 0, self.angular_velocity]), state[0:3]))

    def get_surface_velocity(self):
        """ Placeholder func. """


@dataclass
class EarthLocation(PlanetLocation):
    """ Launch site class for locations on Earth's surface.

    https://en.wikipedia.org/wiki/Earth%27s_rotation
    https://en.wikipedia.org/wiki/Density_of_air
    """

    def __init__(self, name, latitude: float, longitude: float):
        super().__init__(f"{name}, Earth", latitude, longitude, 6371000, 7.292115e-5,
                         9.80665, 3.986004418e14)

    def get_density(self, altitude: float) -> float:
        """ Approximates air density in function of height on Earth, measured from sea level. """
        if 0 <= altitude <= 120000:
            return 1.204 * m.exp(-altitude / 10400)
        return 0.0

    # TODO: implement functionality
    def get_pressure(self, altitude: float) -> float:
        altitude += 1
        return 0.0


class Stage:
    """ Rocket stage class, defined by empty mass, propellant mass, number of engines,
    engine thrust and specific impulse.
    """

    def __init__(self, empty_mass: float, propellant_mass: float, number_of_engines: int, thrust_per_engine: float,
                 specific_impulse: Union[int, list[int]]):
        self._empty_mass = empty_mass  # kg
        self._propellant_mass0 = propellant_mass  # kg
        self._propellant_mass = propellant_mass  # kg
        self.stage_thrust = thrust_per_engine * number_of_engines
        self.specific_impulse = specific_impulse  # s
        self.onboard = True

    def get_thrust(self) -> float:
        """ Returns thrust, if there is any fuel left in the stage to generate it. """
        if self._propellant_mass > 0:
            return self.stage_thrust  # N aka kg/m/s
        L.warning("Fuel tank is empty, no thrust!")
        return 0.0

    def get_mass(self) -> float:
        """ Returns the actual total mass of the stage. """
        return self._empty_mass + self._propellant_mass

    def get_propellant_percentage(self) -> float:
        """ Returns the percentage of fuel left. """
        return self._propellant_mass / self._propellant_mass0 * 100

    def get_specific_impulse(self, pressure_ratio: float = 0.0) -> float:
        """ Returns specific impulse value.

        If the class initiated with a list, for specific impulse, this function can compensate atmospheric pressure
        change by the pressure ratio: (0.0 is sea-level, 1.0 is vacuum pressure). If instead a float is given, this is
        omitted.
        """
        if isinstance(self.specific_impulse, int):  # If only one value is given, it is handled as a constant
            return self.specific_impulse

        if isinstance(self.specific_impulse, list):
            # If a list is given, linearly interpolating between them by the pressure-ratio
            return np.interp(pressure_ratio, [0, 1], self.specific_impulse)

        L.warning("Specific impulse is not in the expected format (float or list of floats)!")
        return 0.0

    def burn_mass(self, mass: float) -> None:
        """ Burn the given amount of fuel. """
        self._propellant_mass = max(0.0, self._propellant_mass - abs(mass))
        L.debug("Fuel left: %s", self.get_propellant_percentage())


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
    """ Describes the status of the rocket attitude control programs during liftoff. """
    VERTICAL_FLIGHT = 0
    ROLL_PROGRAM = 1
    PITCH_PROGRAM = 2
    GRAVITY_ASSIST = 3


class RocketFlightProgram:
    """ Describes the rocket launch program (staging, engine throttling, roll and pitch maneuvers).

    EXAMPLE VALUES (LEO)
    * MECO: 145 s
    * STAGE_SEPARATION: MECO + 3 s
    * SES_1: MECO + 11s
    * fairing jettison = 195 s (LEO) - 222 s (GTO)
    * https://spaceflight101.com/falcon-9-ses-10/flight-profile/#google_vignette
    """

    # pylint: disable = too-many-arguments
    def __init__(self, meco: float, ses_1: float, seco_1: float, ses_2: float, seco_2: float,
                 throttle_map, fairing_jettison: float,
                 pitch_maneuver_start: float, pitch_maneuver_end: float, ss_1: float = None, ss_2: float = None):
        """
        throttle_map - tuple(t, y): t is the list of time-points since launch, and y is the list of throttling
        factor at the corresponding t values. Outside the given timerange, 100% is the default value. Burn duration,
        staging is not evaluated with t.
        Example: 80% throttling between 34 and 45 seconds after burn. Before and after no throttling (100%).
            throttle_map = ([34, 45], [0.8, 0.8])
        """
        # Staging parameters
        self.meco = meco  # s
        self.ses_1 = ses_1  # s
        self.seco_1 = seco_1  # s
        self.ses_2 = ses_2  # s
        self.seco_2 = seco_2  # s
        self.throttle_map = throttle_map  # second - % mapping
        self.fairing_jettison = fairing_jettison  # s

        # Stage separation
        if ss_1 is None:
            self.ss_1 = meco + 3  # s
        else:
            self.ss_1 = ss_1  # s
        if ss_2 is None:
            self.ss_2 = seco_2 + 3  # s
        else:
            self.ss_2 = ss_2  # s

        # Attitude control
        self.pitch_maneuver_start = pitch_maneuver_start
        self.pitch_maneuver_end = pitch_maneuver_end

        # Log data
        self.print_program()

    def get_engine_status(self, t: float) -> RocketEngineStatus:
        """ Return RocketEngineStatus at a given t time since launch. """
        if t < self.meco:
            return RocketEngineStatus.STAGE_1_BURN
        if self.meco <= t < self.ses_1:
            return RocketEngineStatus.STAGE_1_COAST
        if self.ses_1 <= t < self.seco_1:
            return RocketEngineStatus.STAGE_2_BURN
        if self.seco_1 <= t < self.ses_2:
            return RocketEngineStatus.STAGE_2_COAST
        if self.ses_2 <= t < self.seco_2:
            return RocketEngineStatus.STAGE_2_BURN

        return RocketEngineStatus.STAGE_2_COAST

    def get_throttle(self, t: float) -> float:
        """ Return engine throttling factor (0.0 - 1.0) at a given t time since launch. """
        return np.interp(t, self.throttle_map[0], self.throttle_map[1], left=1, right=1)

    def get_attitude_status(self, t: float) -> RocketAttitudeStatus:
        """ Return RocketAttitudeStatus at a given t time since launch. """
        if t < self.pitch_maneuver_start:
            return RocketAttitudeStatus.VERTICAL_FLIGHT
        if self.pitch_maneuver_start <= t < self.pitch_maneuver_end:
            return RocketAttitudeStatus.PITCH_PROGRAM

        return RocketAttitudeStatus.GRAVITY_ASSIST

    def print_program(self):
        """ Print flight program. """
        L.info("--- FLIGHT PROFILE DATA ---")
        L.info("MAIN ENGINE CUT OFF at T+%s (%s s)", secs_to_mins(self.meco), self.meco)
        L.info("STAGE SEPARATION 1 at T+%s (%s s)", secs_to_mins(self.ss_1), self.ss_1)
        L.info("SECOND ENGINE START 1 at T+%s (%s s)", secs_to_mins(self.ses_1), self.ses_1)
        L.info("PAYLOAD FAIRING JETTISON at T+%s (%s s)", secs_to_mins(self.fairing_jettison), self.fairing_jettison)
        L.info("SECOND ENGINE CUT OFF 1 at T+%s (%s s)", secs_to_mins(self.seco_1), self.seco_1)
        L.info("SECOND ENGINE START 2 at T+%s (%s s)", secs_to_mins(self.ses_2), self.ses_2)
        L.info("SECOND ENGINE CUT OFF 2 at T+%s (%s s)", secs_to_mins(self.seco_2), self.seco_2)
        L.info("STAGE SEPARATION 2 at T+%s (%s s)", secs_to_mins(self.ss_2), self.ss_2)


class RocketLaunch:
    """ RocketLaunch class, defined by name, payload mass, drag coefficient and diameter; and stages. """

    def __init__(self, name: str, payload_mass: float, fairing_mass: float, coefficient_of_drag: float, diameter: float,
                 stages: list[Stage], flight_program: RocketFlightProgram, central_body: PlanetLocation):
        self.name = name
        self.stage_status = RocketEngineStatus.STAGE_0
        self.stages = stages
        self.flight_program = flight_program
        self.central_body = central_body

        # Physical properties
        # Drag coefficient (-) times cross-sectional area of rocket (m2)
        self.drag_constant = coefficient_of_drag * (m.pi * pow(diameter, 2) / 4)

        # State variables / dynamic vectors and mass
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # m, m/s and kg

        # Mass properties
        self.payload_mass = payload_mass  # kg
        self.fairing_mass = fairing_mass  # kg
        self.total_mass = self.get_total_mass()  # kg

    def get_total_mass(self) -> float:
        """ Calculates the total mass of the rocket, at any given time. """
        return self.get_stage_mass() + self.fairing_mass + self.payload_mass

    def get_stage_mass(self) -> float:
        """ Sums the mass of each rocket stage, depending on actual staging. """
        mass = 0
        for stage in self.stages:
            if stage.onboard:
                mass += stage.get_mass()
        return mass

    def get_thrust(self) -> float:
        """ Calculates actual thrust (force) of the rocket, depending on actual staging. """

        if self.stage_status == RocketEngineStatus.STAGE_1_BURN:
            return self.stages[0].get_thrust()

        if self.stage_status == RocketEngineStatus.STAGE_2_BURN:
            return self.stages[1].get_thrust()

        return 0

    def get_isp(self, pressure_ratio: float) -> float:
        """ Calculates actual specific impulse of the rocket, depending on actual staging. """

        if self.stage_status == RocketEngineStatus.STAGE_1_BURN:
            return self.stages[0].get_specific_impulse(pressure_ratio)

        if self.stage_status == RocketEngineStatus.STAGE_2_BURN:
            return self.stages[1].get_specific_impulse(pressure_ratio)

        return 0

    def launch_ode(self, time, state, dt):
        """ 2nd order ODE of the state-vectors, during launch.

        The function returns the second derivative of position vector at a given time (acceleration vector),
        using the position (r) and velocity (v) vectors. All values represent the same time.

        Trick: While passing through the velocity vector unchanged, we can numerically integrate both functions in
        the RK4-solver in one step (this is outside of this functions scope).

        State-vector: rx, ry, rz, vx, vy, vz, m
        State-vector_dot: vx, vy, vz, ax, ay, az, m_dot
        """
        r = state[:3]  # Position vector
        v = state[3:6]  # Velocity vector
        mass = state[6]  # Mass

        # Calculate flight characteristics at the actual step
        v_relative = self.central_body.get_relative_velocity(state)
        air_density = self.central_body.get_density(np.linalg.norm(r) - self.central_body.surface_radius)
        pressure_ratio = air_density / 1.2  # TODO: hardcoded!
        drag_const = self.drag_constant * air_density / 2

        # Calculate aceleration
        # Calculate drag and gravity
        a_drag = -(v / np.linalg.norm(v)) * drag_const * v_relative ** 2 / mass
        a_gravity = -r * self.central_body.std_gravitational_parameter / np.linalg.norm(r) ** 3

        # Calculate thrust
        thrust = self.get_thrust() * self.flight_program.get_throttle(time) / mass

        # Vertical flight until tower is cleared
        if time < self.flight_program.pitch_maneuver_start:
            a_thrust = thrust * (r / np.linalg.norm(r))

        # Initial pitch-over maneuver -> Slight offset of Thrust and Velocity vectors
        elif self.flight_program.pitch_maneuver_start <= time < self.flight_program.pitch_maneuver_end:
            # FIXME: cleanup, and investigate how this works exactly
            unit_r = r / np.linalg.norm(r)
            k_vector = np.cross(unit_r, np.array([1, 0, 0]))
            unit_k = k_vector / np.linalg.norm(k_vector)
            a_thrust = mch.rodrigues_rotation(thrust * unit_r, unit_k, 0.01 * m.pi / 180)

        else:  # Gravity assist -> Thrust is parallel with velocity
            a_thrust = thrust * (v / np.linalg.norm(v))

        # Calculate acceleration (v_dot) and m_dot
        a = a_gravity + a_thrust + a_drag  # 2nd order ODE function (acceleration)
        m_dot = - thrust / (self.get_isp(pressure_ratio) * self.central_body.std_gravity) * dt
        return np.concatenate((v, a, [m_dot]))  # vx, vy, vz, ax, ay, az, m_dot

    def launch(self, inclination: float, timestep=1):
        """ Yield rocket's status variables during launch, every second. """

        # CALCULATION START
        # Calculations of target orbit
        # TODO: pre-flight checks for inclination limits
        # TODO: implement calculations for desired orbit, and provide defaults for minimal energy orbit
        # launch_azimuth = m.asinh(m.cos(inclination * m.pi/180) / m.cos(self.central_body.latitude * m.pi / 180))
        # target_velocity =

        # Update state vector with initial conditions
        r_rocket = mch.convert_spherical_to_cartesian_coords(self.central_body.surface_radius,
                                                             self.central_body.latitude * m.pi / 180,
                                                             self.central_body.longitude * m.pi / 180)

        omega_planet = np.array([0, 0, self.central_body.angular_velocity])  # rad/s
        self.state = np.concatenate((r_rocket, np.cross(omega_planet, r_rocket), self.total_mass))
        # Yield initial values
        yield 0, self.state, np.array([0.0, 0.0, 0.0]), 0  # time, state, acc., flight_angle

        time = 0  # Current step
        while time <= 8000:
            # Calculate stage status according to time
            self.stage_status = self.flight_program.get_engine_status(time)

            # Calculate state-vector, acceleration and delta_m
            # The ODE is solved for the acceleration vector and m_dot, which is used as an initial condition for the
            # RK4 numerical integrator function, which then solves for the velocity function.
            # Passing not only the acceleration vector, but the velocity vector to the RK4, we can numerically
            # integrate twice with one function-call, thus we get back the full state-vector.
            self.state, state_dot = mch.runge_kutta_4(self.launch_ode, time, self.state, timestep, timestep)
            acceleration = state_dot[3:6]
            delta_m = state_dot[6]
            # Set mass for rocket: burn mass, and evaluate staging events
            # Burn mass from stage
            if self.stage_status == RocketEngineStatus.STAGE_1_BURN:
                self.stages[0].burn_mass(delta_m)
            if self.stage_status == RocketEngineStatus.STAGE_2_BURN:
                self.stages[1].burn_mass(delta_m)

            # Evaluate staging events:
            if time == self.flight_program.fairing_jettison:
                self.total_mass -= self.fairing_mass
            if time == self.flight_program.ss_1:
                self.stages[0].onboard = False
            if time == self.flight_program.ss_2:
                self.stages[1].onboard = False

            self.get_total_mass()

            # Log new data and end-conditions
            # TODO: implement checks for mass, target velocity, etc.
            altitude_above_surface = np.linalg.norm(self.state[0:3]) - self.central_body.surface_radius
            if altitude_above_surface <= 0:
                L.warning("WARNING! LITHOBRAKING!")
                break

            # TODO: Implement angle calculation between position and velocity vector - 90 deg ??
            # Flight angle
            angle = m.acos(np.dot(acceleration, self.state[3:6]) /
                           (np.linalg.norm(acceleration) * np.linalg.norm(self.state[3:6])))

            # Yield values
            time += timestep
            yield time, self.state, acceleration, angle


def secs_to_mins(total_seconds) -> str:
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
    second_stage = Stage(3900, 92670, 1, 934e3, 348)

    # TODO: Modelling throttle to 80% properly, and test it
    throttle_map = [[50, 90], [0.8, 0.8]]
    flight_program = RocketFlightProgram(130, 141, 514, 3090, 3390, throttle_map,
                                         195, 16, 60, None)
    falcon9 = RocketLaunch("Falcon 9", 15000, 1900, 0.25, 5.2,
                           [first_stage, second_stage], flight_program, cape)

    # Launch
    time_data = []
    alt_data = []
    rx = []
    ry = []
    rz = []
    vx = []
    vy = []
    vz = []
    vel_data = []
    acc_data = []
    mass_data = []
    angle = []

    for time, state, acc, fpa in falcon9.launch(28.5, 1):
        time_data.append(time)
        rx.append(state[0])
        ry.append(state[1])
        rz.append(state[2])
        vx.append(state[3])
        vy.append(state[4])
        vz.append(state[5])
        alt_data.append((np.linalg.norm(state[0:3]) - 6371000) / 1000)  # Altitude in km-s
        vel_data.append(np.linalg.norm(state[3:6]) / 1000)  # Velocity in km/s
        acc_data.append(np.linalg.norm(acc) / 9.82)  # Acceleration in g-s
        mass_data.append(state[6] / 1000)  # Mass in 1000 kg-s
        angle.append(fpa * 180 / m.pi)

    # Plotting
    # TODO: implement colormap for each stage of the flight
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html#sphx-glr-gallery-lines-bars-and-markers-multicolored-line-py
    plt.style.use('_mpl-gallery')

    fig = plt.figure(layout='tight', figsize=(19, 9.5))
    fig.suptitle("Falcon9 launch from Cape Canaveral")
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Flight profile")
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('flight altitude (km)', color="m")
    ax1.set_xlim(0, len(time_data))
    # ax1.set_ylim(0, 8)
    ax1.plot(time_data, alt_data, color="m")

    ax2 = ax1.twinx()
    ax2.set_ylabel('flight path angle (deg)', color="r")
    # TODO: set limits 90deg to -90deg
    ax2.scatter(time_data, angle, s=0.5, color="r")

    # Flight velocity, acceleration
    ax3 = fig.add_subplot(2, 2, 2)
    ax3.set_title("Flight velocity and acceleration")
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('acceleration (g)', color="b")
    ax3.set_xlim(0, len(time_data))
    ax3.set_ylim(0, 8)
    ax3.scatter(time_data, acc_data, s=0.5, color="b")
    ax3.tick_params(axis='y', labelcolor="b")

    ax4 = ax3.twinx()
    ax4.set_ylabel('velocity (km/s)', color="g")
    ax4.set_ylim(0, 8)
    ax4.plot(time_data, vel_data, color="g")
    ax4.tick_params(axis='y', labelcolor="g")

    # Mass
    ax6 = fig.add_subplot(2, 2, 3)
    ax6.set_title("Mass")
    ax6.set_xlabel('time (s)')
    ax6.set_ylabel('mass (kg)')
    ax6.set_xlim(0, len(time_data))
    ax6.set_ylim(0, 1000)
    ax6.scatter(time_data, mass_data, s=0.5, color="b")

    # Plot trajectory in 3D
    ax5 = fig.add_subplot(2, 2, 4, projection='3d')
    ax5.set_title("Flight trajectory")
    ax5.plot(rx, ry, rz, label="Trajectory", color="m")

    # Plot surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 6371000 * np.outer(np.cos(u), np.sin(v))
    y = 6371000 * np.outer(np.sin(u), np.sin(v))
    z = 6371000 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax5.plot_surface(x, y, z)
    ax5.set_aspect('equal')

    # Reference vectors
    ax5.plot([0, 6371000 * 1.1], [0, 0], [0, 0], label="x axis", color="r")
    ax5.plot([0, 0], [0, 6371000 * 1.1], [0, 0], label="y axis", color="g")
    ax5.plot([0, 0], [0, 0], [0, 6371000 * 1.1], label="z axis", color="b")
    ax5.plot([0, rx[0]], [0, ry[0]], [0, rz[0]], label="launch", color="w")  # Launch site

    # Velocity vector at given pos
    pos = 0
    ax5.plot([rx[pos], rx[pos]+vx[pos]*10000], [ry[pos], ry[pos]+vy[pos]*10000],
             [rz[pos], rz[pos]+vz[pos]*10000], label="start_v", color="c")

    plt.show()


# Include guard
if __name__ == '__main__':
    main()
