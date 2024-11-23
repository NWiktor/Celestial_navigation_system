# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" This module contains all relevant class and function for orbit propagation
around a celestial body. The module calculates the trajectory of a two-stage
rocket launched from surface in Object-Centered Inertial reference frame (OCI).

Libs
----
* Ursina

Help
----
* https://www.solarsystemscope.com/textures/

Contents
--------
"""

# Standard library imports
# First import should be the logging module if any!
import logging
import math as m

# Third party imports
from ursina import *

# Local application imports

logger = logging.getLogger(__name__)
CAMERA_AZIMUTH = 55
CAMERA_POLAR = 120
CAMERA_RADIUS = 100

TIME_SCALE_FACTOR = 200  # the passing of time is multiplied by this number
DIMENSION_SCALE_FACTOR = 1000  # all dimensions (in km) are divided by this number
GRID_SIZE_KM = 100000
SUBGRID_RATIO = 5

GLOBAL_TIME = 0
alpha = 0
START_TIME = 0

OBJECTS = []


class Planet(Entity):
    """ Creates a planet entity."""
    def __init__(self, radius_km, position=(0, 0, 0), texture_file=None):
        scale = radius_km / DIMENSION_SCALE_FACTOR * 2
        super().__init__(parent=scene, scale=scale, position=position)

        if texture_file is not None:
            self.texture_entity = PlanetTexture(self, texture_file)


class PlanetTexture(Entity):
    """ Creates a texture entity, which is rotated around the X axis, to align
    the texture and the parent body coordinate system."""
    def __init__(self, parent, texture_file):
        super().__init__(parent=parent, position=(0, 0, 0),
                         model='sphere',
                         rotation_x=-90,
                         texture=texture_file)


def update():
    global alpha, CAMERA_AZIMUTH, CAMERA_POLAR, CAMERA_RADIUS

    # Camera
    CAMERA_AZIMUTH += held_keys['d'] * 20 * time.dt
    CAMERA_AZIMUTH -= held_keys['a'] * 20 * time.dt
    CAMERA_POLAR += held_keys['w'] * 15 * time.dt
    CAMERA_POLAR -= held_keys['s'] * 15 * time.dt
    CAMERA_RADIUS += held_keys['up arrow'] * 50 * time.dt
    CAMERA_RADIUS -= held_keys['down arrow'] * 50 * time.dt

    camera.x = (CAMERA_RADIUS * m.cos(m.radians(CAMERA_AZIMUTH))
                * m.sin(m.radians(CAMERA_POLAR)))
    camera.y = (CAMERA_RADIUS * m.sin(m.radians(CAMERA_AZIMUTH))
                * m.sin(m.radians(CAMERA_POLAR)))
    camera.z = CAMERA_RADIUS * m.cos(m.radians(CAMERA_POLAR))
    camera.look_at(earth, up=earth.back)
    rotation_info.text = (
            f"Camera pos.: {camera.position}\n"
            + f"Camera rot.: {camera.rotation}\n"
            + f"Camera azimuth.: {CAMERA_AZIMUTH}\n"
            + f"Camera polar.: {CAMERA_POLAR}\n"
            + f"Camera radius.: {CAMERA_RADIUS}\n"
    )

    # Animation
    radius = 25000 / DIMENSION_SCALE_FACTOR
    alpha += 10 * time.dt
    moon.x = radius * m.cos(m.radians(alpha))
    moon.y = radius * m.sin(m.radians(alpha))


def input(key):
    global CAMERA_RADIUS
    if key == 'escape':
        quit()

    if key == 'scroll up':
        CAMERA_RADIUS += 100 * time.dt

    if key == 'scroll down':
        CAMERA_RADIUS -= 100 * time.dt


def main():
    # 384_400 - original orbit of moon
    earth = Planet(6_378, (0, 0, 0),
                   'resource/2k_earth_daymap.jpg')
    moon = Planet(1_737, (0, 0, 0),
                  'resource/lroc_color_poles_1k.jpg')

    orbit = Entity(model=Circle(120, radius=25000 / DIMENSION_SCALE_FACTOR,
                                mode='line', thickness=2),
                   rotation_x=0, color=color.hsv(60, 1, 1, .3))

    return earth, moon


def create_unit_vectors(parent=scene, scale=1, right_handed=False):
    height = 5
    radius = 0.1
    z_dir = 1
    if right_handed:
        z_dir = -1

    x_vector = Entity(model=Cylinder(6, radius=radius,
                                     direction=(1, 0, 0), height=height),
                      parent=parent, world_scale=scale, color=color.red)
    y_vector = Entity(model=Cylinder(6, radius=radius,
                                     direction=(0, 1, 0), height=height),
                      parent=parent, world_scale=scale, color=color.green)
    z_vector = Entity(model=Cylinder(6, radius=radius,
                                     direction=(0, 0, z_dir), height=height),
                      parent=parent, world_scale=scale, color=color.blue)


def create_grid():
    grid_size = GRID_SIZE_KM / DIMENSION_SCALE_FACTOR  # size of 1 grid in units
    grid_number = 10  # number of grids (n x n)
    grid = Entity(model=Grid(grid_number, grid_number),
                  scale=grid_number * grid_size,
                  rotation_x=0,
                  color=color.blue)
    subgrid = duplicate(grid)
    grid.model.thickness = 2
    subgrid.model = Grid(grid_number * SUBGRID_RATIO,
                         grid_number * SUBGRID_RATIO)
    subgrid.color = color.azure


if __name__ == '__main__':
    app = Ursina(vsync=False, fullscreen=True)
    window.color = color.black

    rotation_info = Text(position=window.top_left)

    earth, moon = main()
    create_unit_vectors(scale=3)
    create_unit_vectors(earth, scale=3, right_handed=True)
    create_unit_vectors(moon, scale=3, right_handed=True)

    create_grid()

    app.run()
