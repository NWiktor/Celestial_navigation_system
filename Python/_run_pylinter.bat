ECHO OFF
chcp 65001
cls


:loop
cls
pylint cls/atmosphere.py --output-format=colorized
pylint cls/celestial_body.py --output-format=colorized
pylint cls/celestial_body_utils.py --output-format=colorized
pylint cls/flightprogram.py --output-format=colorized
pylint cls/hardware.py --output-format=colorized
pylint cls/kepler_orbit.py --output-format=colorized
pylint cls/planet.py --output-format=colorized
pylint cls/star.py --output-format=colorized

rem pylint common/launch_OCI.py --output-format=colorized
pause
goto loop
