# -*- coding: utf-8 -*-
# !/usr/bin/python3
""" This module contains all relevant class and function for orbit propagation
around a celestial body. The module calculates the trajectory of a two-stage
rocket launched from surface in Object-Centered Inertial reference frame (OCI).

Libs
----
* Numpy
* Matplotlib

Help
----
* https://en.wikipedia.org/wiki/Orbital_speed

Contents
--------
"""

# Standard library imports
# First import should be the logging module if any!
import logging
import math as m
from datetime import datetime

# Third party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from utils import (convert_spherical_to_cartesian_coords,
                   runge_kutta_4, unit_vector, rodrigues_rotation,
                   angle_of_vectors, cross)
from cls import (LaunchSite, CircularOrbit, Rocket,
                 EngineStatus, FALCON9,
                 FlightProgram, AttitudeStatus)
from database import CAPE_TEST, CAPE_CANEVERAL

logger = logging.getLogger(__name__)


class RocketLaunch:
    """ RocketLaunch class.

    :param str mission_name: Name of the launch (misson).
    :param Rocket rocket: Launch-vehicle from Rocket class.
    :param float payload_mass: Mass of payload (kg).
    :param flightprogram: Flightprogram.
    :param target_orbit: Target orbit of launch.
    :param launchsite: Location data of launch.
    :param float earliest_launch_date: Earlist date of launch.
    """

    def __init__(self, mission_name: str, rocket: Rocket, payload_mass: float,
                 flightprogram: FlightProgram,
                 target_orbit: CircularOrbit, launchsite: LaunchSite,
                 earliest_launch_date: float = None):
        self.mission_name = mission_name
        self.rocket = rocket
        self.rocket.set_payload_mass_kg(payload_mass)  # Call to set payload!
        self.flightprogram = flightprogram
        self.target_orbit = target_orbit
        self.launchsite = launchsite
        self.launch_azimuth: list[float | None] = [None, None]
        self.earliest_launch_date = earliest_launch_date

        if earliest_launch_date is None:
            pass
            # FIXME: self.earliest_launch_date = time.now()
        else:
            self.earliest_launch_date = earliest_launch_date

        # Check if orbit is reachable
        self._check_radius()
        self.target_velocity = self._get_target_velocity(
            self.target_orbit.radius_km * 1000)  # m/s
        self._check_inclination()
        self._get_launch_azimuth()  # Calculate laumch azimuth
        self._get_launch_date()  # Time of launch to get desired LoAN

        # Launchsite vectors
        self._density_at_surface = self.launchsite.get_density(0.0)
        self.r_launch = None
        self.launch_plane_unit = None

        # State variables / dynamic vectors and mass (m, m/s and kg)
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def _check_radius(self):
        """ Check if specified target orbit radius is valid: greater than the
        planet surface.
        """
        if (self.target_orbit.radius_km * 1000 <=
                self.launchsite.planet.surface_radius_m):
            logger.error("ERROR: orbit radius (%.3f km) is smaller "
                         "than surface radius!",
                         self.target_orbit.radius_km)
            raise ValueError

        logger.info("Orbit radius: %.3f km", self.target_orbit.radius_km)

    def _check_inclination(self):
        """ Check if specified target orbit inclination is valid: greater than
        the launch-site latitude.
        """
        if self.target_orbit.inclination_deg < self.launchsite.latitude_deg:
            logger.error("ERROR: Cannot launch directly into orbit with"
                         "inclination (%.3f°) smaller than launchsite"
                         "latitude (%.3f)!",
                         self.target_orbit.inclination_deg,
                         self.launchsite.latitude_deg)
            raise ValueError

        logger.info("Inclination: %.3f°", self.target_orbit.inclination_deg)

    # TODO: test baikonour and korou azimuth limit (349 to 90)
    def _get_launch_azimuth(self):
        """ Check if target orbit is reacheable by direct orbit insertion.

        A handy formula to remember is: cos(i) = cos(φ) * sin(β), where i is
        the inclination, β is the launch azimuth, and φ is the launch
        latitude.
        https: // www.orbiterwiki.org / wiki / Launch_Azimuth
        """
        launch_azimuth = m.asin(
                m.cos(self.target_orbit.inclination_deg * m.pi / 180)
                / m.cos(self.launchsite.latitude_deg * m.pi / 180)
                )  # rad
        v_eqrot = (self.launchsite.planet.surface_radius_m
                   * self.launchsite.planet.angular_velocity_rad_per_s)  # m/s
        launch_azimuth_corr = m.atan2(
                self.target_velocity * m.sin(launch_azimuth)
                - v_eqrot * m.cos(self.launchsite.latitude_deg * m.pi / 180),
                self.target_velocity * m.cos(launch_azimuth)
        ) / m.pi * 180  # deg

        # TODO: Test out range of launch_azimuth1 and launch_azimuth2
        launch_azimuth1 = launch_azimuth_corr  # range is -90° - 90° ??
        launch_azimuth2 = 180 - launch_azimuth_corr  # range is 90° - 270°
        logger.info("Launch azimuth for AN: %.3f°", launch_azimuth1)
        logger.info("Launch azimuth for DN: %.3f°", launch_azimuth2)

        # launch_azimuth1 = (launch_azimuth1 + 360) % 360
        # launch_azimuth2 = (launch_azimuth2 + 360) % 360

        if self.launchsite.launch_azimuth_range is not None:

            # Check values
            start_lim = self.launchsite.launch_azimuth_range[0]  # 0-360°
            end_lim = self.launchsite.launch_azimuth_range[1]  # 0-360°

            # if end_lim < start_lim:
            #     end_lim += 360

            if not start_lim <= launch_azimuth1 <= end_lim:
                logger.warning("WARNING: Launch azimuth for ascending "
                               "node (%.3f°) is out of permitted range!",
                               launch_azimuth1)
            self.launch_azimuth[0] = launch_azimuth1

            if not start_lim <= launch_azimuth2 <= end_lim:
                logger.warning("WARNING: Launch azimuth for descending "
                               "node (%.3f°) is out of permitted range!",
                               launch_azimuth2)
            self.launch_azimuth[1] = launch_azimuth2

            if (self.launch_azimuth[0] is None and
               self.launch_azimuth[1] is None):
                logger.error("ERROR: Launch is not possible from this location"
                             "because of launch azimuth limitations!")
                raise ValueError

        else:
            logger.warning("No checks for launch azimuth!")
            self.launch_azimuth[0] = launch_azimuth1
            self.launch_azimuth[1] = launch_azimuth2

    def _get_target_velocity(self, radius_m):
        """ Calculates orbital velocity (m/s) for the given radius (m). """
        target_velocity = m.sqrt(
                self.launchsite.std_gravitational_parameter / radius_m
        )
        logger.debug("Target velocity for orbit: %.3f m/s", target_velocity)
        return target_velocity

    def _get_launch_date(self):
        """
        cos(i) = cos(φ) * sin(β), where i is
        the inclination, β is the launch azimuth, and φ is the launch
        latitude. """
        # TODO: implement

        # alfa - inclination auxiliary variable
        # delta - launch-window location angle
        # gamma - launch-direction auxiliary variable, γ,
        # cos delta = cos(gamma) / sin(alfa)

        # North hemisphere:
        # LWST_AN = Ω + δ
        # LWST_DN = Ω + (180º – δ)
        # South hemisphere:
        # LWSTAN  = Ω – δ
        # LWSTDN  = Ω + (180º + δ)

    def _check_end_condition_crash(self) -> bool:
        """ Calculates altitude from planet surface, if it's negative, crash
        has been occured. """
        altitude = np.linalg.norm(self.state[0:3]) - self.launchsite.radius_m
        if altitude <= 0:
            logger.warning("WARNING! LITHOBRAKING!")
            return True

        return False

    def _check_end_condition_orbit(self) -> bool:
        """ Calculates if the vehicle speed, altitude and flight angle match
        with the target orbit.
        """
        r_current = np.linalg.norm(self.state[0:3])
        v_current = np.linalg.norm(self.state[3:6])
        altitude_m = r_current - self.launchsite.radius_m
        # flight_angle_deg = angle_of_vectors(
        #     unit_vector(self.state[0:3]),
        #     unit_vector(self.state[3:6])
        # )

        delta_r = self.target_orbit.radius_km * 1000 - r_current
        delta_v = self.target_velocity - v_current
        limit_r = self.target_orbit.radius_km * 0.1
        limit_v = self.target_velocity * 0.001
        # limit_beta = 90 * 0.1

        # If radius and velocity is close to target orbit:
        if abs(delta_r) <= limit_r and abs(delta_v) <= limit_v:
            logger.info("Targeted orbit reached at %.3f m, %.3f m/s",
                        altitude_m, v_current)
            return True

        # If radius is bigger than target orbit, but close to circular r-v pair
        delta_v2 = self._get_target_velocity(r_current) - v_current
        if delta_r <= 0 and abs(delta_v2) <= limit_v:
            logger.info("Stable orbit reached at %.3f m, %.3f m/s",
                        altitude_m, v_current)
            return True

        logger.info("Deviation from target orbit: %.3f m, %.3f m/s",
                    delta_r, delta_v)
        return False

    def _set_inital_conditions(self):
        """ Set initial conditions before launch calculation. """

        # TODO: check this in alfonso's program, how to account for j2000 frame
        angle_offset_deg = 0.0

        # Calculate initial conditions
        self.r_launch = convert_spherical_to_cartesian_coords(
            self.launchsite.radius_m,
            self.launchsite.latitude_deg * m.pi / 180,
            (self.launchsite.longitude_deg + angle_offset_deg) * m.pi / 180
        )
        omega_cb = np.array([0, 0, self.launchsite.angular_velocity_rad_per_s])
        v_cb_rotation = cross(omega_cb, self.r_launch)

        # Update state vector with initial conditions
        self.state = np.concatenate(
            (self.r_launch, v_cb_rotation, [self.rocket.total_mass])
        )

    def _set_launch_plane_normal(self, r_vector: np.array):
        """ Calculate launch plane's normal vector from the given 'r'
        position vector.

        Launch plane normalvector is static, it represents the initial plane,
        in which the rocket is launched.
        """
        east_launch = cross(np.array((0, 0, 1)), r_vector)
        local_north = unit_vector(cross(r_vector, east_launch))
        local_zenith = unit_vector(r_vector)
        launch_plane_normal = rodrigues_rotation(
            local_north,
            local_zenith,
            (90 - self.launch_azimuth[0]) * m.pi / 180
        )
        self.launch_plane_unit = unit_vector(launch_plane_normal)

    # pylint: disable = too-many-locals
    def _launch_ode(self, time: float, state: np.array, dt: float):
        """ 2nd order ODE of the state-vectors, during launch.

        The function returns the second derivative of position vector at a given
        time (acceleration vector), using the position (r) and velocity (v)
        vectors. All values represent the same time.

        Trick: While passing through the velocity vector unchanged, we can
        numerically integrate both functions in the RK4-solver in one step (this
        is outside of this functions scope).

        State-vector: rx, ry, rz, vx, vy, vz, m
        State-vector_dot: vx, vy, vz, ax, ay, az, m_dot
        """
        r = state[:3]  # Position vector
        v = state[3:6]  # Velocity vector
        mass = state[6]  # Mass

        # Get attutude status
        att_status = self.flightprogram.get_attitude_status(time)

        # Calculate flight characteristics at the actual step
        v_rel_skalar = np.linalg.norm(
            self.launchsite.get_relative_velocity_vector(state)
        )
        altitude_m = np.linalg.norm(r) - self.launchsite.radius_m
        air_density = self.launchsite.get_density(altitude_m)
        pressure_ratio = air_density / self._density_at_surface
        drag_const = self.rocket.drag_constant * air_density / 2

        # Calculate acceleration: gravity
        a_gravity = (-r * self.launchsite.std_gravitational_parameter
                     / np.linalg.norm(r) ** 3)
        # Calculate acceleration: drag (numeric value only, direction later)
        drag = drag_const * v_rel_skalar ** 2 / mass

        # Calculate acceleration: thrust (numeric value only, direction later)
        thrust_force = (self.rocket.get_thrust()
                        * self.flightprogram.get_throttle(time))
        thrust = thrust_force / mass

        # Orbital parameters
        # NOTE: flight angle represents the angle between the local zenith, and
        #  the rocket relative velocity vector. It should start from 0°
        #  (vertical flight) to 90° as the rocket reaches orbit.
        # NOTE: flight angle calculation must be corrected for planet rotation,
        #  so dynamic relative velocity vector must be used.
        v_rel = self.launchsite.get_relative_velocity_vector(state)
        flight_angle = angle_of_vectors(unit_vector(r), unit_vector(v_rel))

        # Vertical flight until tower is cleared in non-inertial frame
        if att_status == AttitudeStatus.VERTICAL_FLIGHT:
            # NOTE: should use dynamic 'r_tower' vector, because launch site is
            #  rotating in the inertial frame with the central body; however
            #  this is very hard to compensate later with hand-made
            #  thrust-vectoring, so just use 'r_launch' to eliminate offset
            #  from the launch-plane
            # NOTE: Use time and not dt!
            r_tower = rodrigues_rotation(
                self.r_launch,
                np.array([0, 0, 1]),
                self.launchsite.angular_velocity_rad_per_s * m.pi / 180 * time)
            a_thrust = thrust * unit_vector(r_tower)  # self.r_launch)
            a_drag = - drag * unit_vector(r_tower)

            # Calculate launch plane, which is needed only at the last step
            # before the pitch program
            self._set_launch_plane_normal(r)

        # Initial pitch-over maneuver -> Slight offset of thrust from velocity
        elif att_status == AttitudeStatus.PITCH_PROGRAM:
            # NOTE: incrementally rotating the acceleration vector around the
            #  launch plane normal vector, to imitate pitch manuever
            # TODO: find universally applicable parameters, or implement checks
            #  to set it automatically
            v_pitch = rodrigues_rotation(
                    unit_vector(r),
                    self.launch_plane_unit,
                    (flight_angle
                     + self.flightprogram.pitch_angle) * m.pi / 180)
            a_thrust = thrust * unit_vector(v_pitch)
            a_drag = - drag * unit_vector(v_pitch)

        else:  # Gravity assist -> Thrust is parallel with velocity
            # NOTE: keep acceleration vector in launch plane
            v_pitch = rodrigues_rotation(
                unit_vector(r),
                self.launch_plane_unit,
                flight_angle * m.pi / 180)
            a_thrust = thrust * unit_vector(v_pitch)
            a_drag = - drag * unit_vector(v_pitch)

        # Calculate acceleration (v_dot) and m_dot
        a = a_gravity + a_thrust + a_drag  # 2nd order ODE function (acc.)
        m_dot = (- thrust_force  # Thrust force is already throttled
                 / (self.rocket.get_isp(pressure_ratio)
                    * self.launchsite.planet.surface_gravity_m_per_s2 * dt
                    )
                 )
        return np.concatenate((v, a, [m_dot]))  # vx, vy, vz, ax, ay, az, m_dot

    def launch(self, total_steps: int = 16000, increment: int = 1):
        """ Yield rocket's status variables during launch, every second. """

        # Update state vector with initial conditions
        self._set_inital_conditions()

        # Yield initial values - time, state, acc., alt., flight angle
        yield 0, self.state, np.array([0.0, 0.0, 0.0]), 0, 0

        logger.info("--- FLIGHT CALCULATION START ---")
        time_step = 0  # Current step
        while time_step <= total_steps:
            # Calculate stage status according to time
            self.rocket.set_stage_status(
                self.flightprogram.get_engine_status(time_step))

            # Calculate state-vector, acceleration and delta_m
            # NOTE: The ODE is solved for the acceleration vector and m_dot,
            #  which is used as an initial condition for the RK4 numerical
            #  integrator function, which then solves for the velocity function.
            #  Passing not only the acceleration vector, but the velocity vector
            #  to the RK4, we can numerically integrate twice with one
            #  function-call, thus we get back the full state-vector.
            self.state, state_dot = runge_kutta_4(
                self._launch_ode, time_step, self.state, increment, increment
            )
            acceleration = state_dot[3:6]

            # Post-processing
            # NOTE: relative velocity vector used for flight angle calculation,
            #  this way planet rotation is corrected.
            v_rel = self.launchsite.get_relative_velocity_vector(self.state)
            # NOTE: flight angle represents the angle between the local zenith,
            #  and the rocket relative velocity vector. It should start from 0°
            #  (vertical flight) to 90° as the rocket reaches orbit.
            flight_angle = angle_of_vectors(
                unit_vector(self.state[0:3]), unit_vector(v_rel))
            # NOTE: the current orbital plane is determined by the current pos.
            #  and velocity vectors. It is changing bc. of the planet rotation.
            orbital_plane_current = cross(
                unit_vector(self.state[0:3]), unit_vector(self.state[3:6])
            )
            # NOTE: current inclination is changing as the current orbital plane
            #  changes. It should reach the target value at the same time as
            #  orbital radius and velocity.
            inclination_current = angle_of_vectors(
                unit_vector(orbital_plane_current),
                np.array([0, 0, 1])
            )
            acc_deviation = angle_of_vectors(
                state_dot[3:6], self.launch_plane_unit
            )

            # Logging variables
            logger.info("--- TIMESTEP: T+%s s ---", time_step)
            logger.info("Relative velocity vector: %s m/s", v_rel)
            logger.info("Current inclination: %.3f°", inclination_current)
            logger.info("Flight angle: %.3f°", flight_angle)
            logger.info("Angle between accelaration and launch "
                        "plane: %.3f°", acc_deviation)

            # Burn mass from stage
            # TODO: remove ifs
            stage_status = self.rocket.get_stage_status()
            if stage_status == EngineStatus.STAGE_1_BURN:
                self.rocket.stages[0].burn_mass(state_dot[6], time_step)
            if stage_status == EngineStatus.STAGE_2_BURN:
                self.rocket.stages[1].burn_mass(state_dot[6], time_step)

            # Evaluate staging events, and update statevector with new mass
            if time_step == self.flightprogram.fairing_jettison:
                self.state[6] = self.rocket.fairing_jettison_event()
            if time_step == self.flightprogram.ss_1:
                self.state[6] = self.rocket.staging_event_1()

            # Log new data and end-conditions
            r_current = np.linalg.norm(self.state[0:3])
            altitude_above_surface = r_current - self.launchsite.radius_m

            if self._check_end_condition_crash():
                break

            if self._check_end_condition_orbit():
                logger.info("--- ORBITAL INSERTION SUCCESFUL ---")
                break

            # Yield values
            time_step += increment
            yield (time_step, self.state, acceleration,
                   altitude_above_surface, flight_angle)


# TODO: implement this
class RocketLanding:
    """ xxxx """

    def __init__(self):
        # use the same with launch but in reverse order
        pass


# TODO: implement this
class LaunchTrajectory3D:
    """ xxx """

    def __init__(self, name: str, rocketlaunch: RocketLaunch,
                 no_earlier_date: datetime):
        self.name = name
        self.rocketlaunch = rocketlaunch
        self.no_earlier_date = no_earlier_date

    def get_position(self, time):
        """ xxxx """
        # return postion at given time, just like the orbit functions
        # if no stable orbit: return launch func
        # if stable orbit reached, create orbit inst and return values from
        # there to skip iteration
        return


# Main function for module testing
# pylint: disable = too-many-statements
# TODO: add option for continous plotting?
# TODO: use 3d visualization with ursina
def plot(rocketlaunch: RocketLaunch):
    """ Plots the given RocketLaunch parameters. """
    # Launch
    time_data, state_data, alt_data, vel_data, acc_data = [], [], [], [], []
    mass_data, beta_data = [], []

    cbsr = rocketlaunch.launchsite.radius_m
    plot_title = (f"{rocketlaunch.mission_name} launch from"
                  f" {rocketlaunch.launchsite.name}")

    for time, state, acc, _, beta in rocketlaunch.launch(550, 1):
        time_data.append(time)
        state_data.append(state)
        alt_data.append((np.linalg.norm(state[0:3]) - cbsr) / 1000)  # Alt. - km
        vel_data.append(np.linalg.norm(state[3:6]))  # Velocity - m/s
        acc_data.append(np.linalg.norm(acc) / 9.82)  # Acceleration - g
        mass_data.append(state[6] / 1000)  # Mass - 1000 kg
        beta_data.append(beta)  # Flight angle - °

    # pts_x = []
    # pts_y = []
    # pts_z = []
    # vector = np.array([rx[0], ry[0], rz[0]])
    # for i in range(0, 90):
    #     rot_v = rodrigues_rotation(vector, launch_plane_normal, i * m.pi / 180)
    #     pts_x.append(rot_v[0])
    #     pts_y.append(rot_v[1])
    #     pts_z.append(rot_v[2])

    # Plotting
    plt.style.use('_mpl-gallery')

    fig = plt.figure(layout='tight', figsize=(19, 9.5))
    fig.suptitle(plot_title)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Flight profile")
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('flight altitude (km)', color="m")
    ax1.plot(time_data, alt_data, color="m")

    # Flight velocity, acceleration
    ax3 = fig.add_subplot(2, 2, 2)
    ax3.set_title("Flight velocity and acceleration")
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('acceleration (g)', color="b")
    ax3.set_xlim(0, len(time_data))
    ax3.set_ylim(0, 10)
    ax3.scatter(time_data, acc_data, s=0.5, color="b")
    ax3.tick_params(axis='y', labelcolor="b")

    ax4 = ax3.twinx()
    ax4.set_ylabel('velocity (m/s)', color="g")
    ax4.set_ylim(0, 10000)
    ax4.plot(time_data, vel_data, color="g")
    ax4.tick_params(axis='y', labelcolor="g")

    # Mass
    ax6 = fig.add_subplot(2, 2, 3)
    ax6.set_title("Mass")
    ax6.set_xlabel('time (s)')
    ax6.set_ylabel('mass (kg)')
    ax6.set_xlim(0, len(time_data))
    ax6.set_ylim(0, 600)
    ax6.scatter(time_data, mass_data, s=0.5, color="b")

    # Flight angle
    ax6 = fig.add_subplot(2, 2, 4)
    ax6.set_title("Flight angle")
    ax6.set_xlabel('time (s)')
    ax6.set_ylabel('flight angle (°)')
    ax6.set_xlim(0, len(time_data))
    ax6.set_ylim(0, 120)
    ax6.plot(time_data, beta_data, color="y")

    # Plot trajectory in 3D
    # ax5 = fig.add_subplot(2, 2, 4, projection='3d')
    # ax5.set_title("Flight trajectory")
    # ax5.plot(rx, ry, rz, label="Trajectory", color="m")
    #
    # ax5.plot(pts_x, pts_y, pts_z, label="Orbit_plane", color="y")
    #
    # # Plot CB surface
    # u = np.linspace(0, 2 * np.pi, 100)
    # v = np.linspace(0, np.pi, 100)
    # x = cbsr * np.outer(np.cos(u), np.sin(v))
    # y = cbsr * np.outer(np.sin(u), np.sin(v))
    # z = cbsr * np.outer(np.ones(np.size(u)), np.cos(v))
    # ax5.plot_surface(x, y, z)
    # ax5.set_aspect('equal')
    #
    # # Reference vectors
    # ax5.plot([0, launch_plane_normal[0] * cbsr * 1.1],
    #          [0, launch_plane_normal[1] * cbsr * 1.1],
    #          [0, launch_plane_normal[2] * cbsr * 1.1],
    #          label="launch_plane_normal", color="y")
    # ax5.plot([0, cbsr * 1.1], [0, 0], [0, 0], label="x axis", color="r")
    # ax5.plot([0, 0], [0, cbsr * 1.1], [0, 0], label="y axis", color="g")
    # ax5.plot([0, 0], [0, 0], [0, cbsr * 1.1], label="z axis", color="b")
    # # Launch site:
    # ax5.plot([0, rx[0]], [0, ry[0]], [0, rz[0]], label="launch", color="w")

    plt.show()


# Include guard
def main():
    """ Example SpaceX Falcon 9 launch demonstrating the use of the
    LaunchSite, Stage and RocketLaunch classes.
    
    The function calculates and plots flight parameters during and after
    liftoff.
    """
    # TODO: Modelling throttle to 80% properly, and test it
    throttle_map = [[70, 80, 81, 150, 550],
                    [0.8, 0.8, 1.0, 0.88, 0.88]]
    flight_program = FlightProgram(145, 156, 514,
                                   throttle_map, 195)
    targetorbit = CircularOrbit(310 + 6_378, 51.6, -45,
                                90,
                                0)
    mission_414_falcon9 = RocketLaunch("Mission 414",
                                       FALCON9, 15000,
                                       flight_program,
                                       targetorbit,
                                       CAPE_TEST)

    # Plot launch
    plot(mission_414_falcon9)


# Include guard
if __name__ == '__main__':
    main()
