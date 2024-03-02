# pylint: disable = missing-module-docstring
import unittest
import matplotlib.pyplot as plt
from Python.utils import EarthAtmosphere


class TestEarthAtmosphere(unittest.TestCase):
    """ aaaa """

    def test_apply_limits(self):
        """ Test for put of limit values. """

        test_class = EarthAtmosphere()

        self.assertEqual(test_class.apply_limits(-2), 0, "Should be zero for negativ values!")


def test_plot_atmosphere():
    """ aaa """

    alt = []
    tmp = []
    pres = []
    rho = []
    for i in range(0, 100_000):
        data = EarthAtmosphere().atmospheric_model(i)
        alt.append(i)
        tmp.append(data[0])
        pres.append(data[1])
        rho.append(data[2])

    # Plotting
    plt.style.use('_mpl-gallery')

    fig = plt.figure(layout='tight', figsize=(19, 9.5))
    fig.suptitle("Atmospheric test")
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Temperature")
    ax1.set_xlabel('altitude (m)')
    ax1.set_ylabel('temperature (°C)', color="m")
    ax1.plot(alt, tmp, color="m")

    # Flight velocity, acceleration
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Pressure")
    ax2.set_xlabel('altitude (m)')
    ax2.set_ylabel('pressure (kPa)', color="b")
    # ax2.set_xlim(0, len(time_data))
    # ax2.set_ylim(0, 10)
    ax2.scatter(alt, pres, s=0.5, color="b")
    # ax2.tick_params(axis='y', labelcolor="b")

    # Flight velocity, acceleration
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("Air density")
    ax3.set_xlabel('altitude (m)')
    ax3.set_ylabel('air density', color="g")
    # ax3.set_xlim(0, len(time_data))
    # ax3.set_ylim(0, 10)
    ax3.scatter(alt, rho, s=0.5, color="g")
    ax3.tick_params(axis='y', labelcolor="g")


if __name__ == '__main__':
    unittest.main()
    test_plot_atmosphere()
