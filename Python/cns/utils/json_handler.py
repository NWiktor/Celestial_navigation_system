# -*- coding: utf-8 -*-
# !/usr/bin/python3

"""JSON database handler module for SeatUP Desktop App. This module describes
the JSON database handler classes.

Help
----
* https://www.pythoncentral.io/hashing-strings-with-python/
* https://stackoverflow.com/questions/18337407/saving-utf-8-texts-with-json-/
    dumps-as-utf8-not-as-u-escape-sequence

Contents
--------
"""

# Standard library imports
import json

# pylint: disable = no-name-in-module
# Third party imports
# from PyQt5.QtGui import QStandardItem
# from PyQt5.QtCore import Qt

# Local application imports
from logger import MAIN_LOGGER as L


class JsonDbHandler:
    """JSON database handler baseclass.

    This class contains all base function and method for handling JSON
    databases, e.g. load, dump, update etc.

    :param path: Filepath of JSON database.
    :type path: filepath

    """

    def __init__(self, path):
        self._filepath = path

    def load(self):
        """Loads a JSON object from a given filepath, then converts it to python
        dictionary.

        :return: Dictionary from JSON object.
        :rtype: dict

        """
        with open(self._filepath, encoding='utf8') as read_file:
            return json.load(read_file)  # Változóba tölti be a json fájlt

    def dump(self, database):
        """Dumps (writes) a given python dictionary to the disk as JSON object.

        :param database: Python dictionary to be written to disk.
        :type database: dict

        """
        with open(self._filepath, "w", encoding='utf8') as write_file:
            json.dump(database, write_file, ensure_ascii=False, indent=4)

    # NOTE: update2 is unnecessary, because it's implies 3x nested dictionary,
    # which is too complex to implement
    def update(self, new_database):
        """Loads, updates and dumps a JSON object with another database (dict).

        :param new_database: New database.
        :type new_database: dict

        """
        database = self.load()
        database.update(new_database)
        self.dump(database)

    # TODO: Find better name for behavior
    def insert_section(self, identifier, section):  # Set value in database (overwrites if exists)
        """Insert / overwrite given entry (section) to JSON object.

        :param str identifier: Key in database (JSON object).
        :param section: Dictionary to be inserted under *identifier* key into
            database (JSON object).
        :type section: dict

        """
        database = self.load()  # DB betöltése
        database[identifier] = section  # Adott bejegyzés beszúrása
        self.dump(database)  # DB kiírása
        L.info("Database %s succesfully updated!", self._filepath)

    # TODO: Check behavior, and add option to toggle update/overwrite
    # TODO: Find better name for behavior
    def insert_section2(self, identifier, identifier2, section):  # Overwrites sub-value (id2) with new value,
        # but main values (id1) preserved
        """Insert / overwrite given entry (section) to JSON database.

        :param str identifier: Key in database (JSON object).
        :param str identifier2: Subkey in database (JSON object).
        :param section: Dictionary to be inserted under *identifier* key and
            *identifier2* subkey into database (JSON object).
        :type section: dict

        """
        database = self.load()  # DB betöltése
        if database[identifier] is not None:
            # Adott bejegyzés beszúrása
            database[identifier].update({identifier2: section})
        else:
            database[identifier] = {identifier2: section}
        self.dump(database)  # DB kiírása
        L.info("Database %s succesfully updated!", self._filepath)

    def delete_section(self, identifier):
        """Delete given entry (section) from JSON object.

        :param str identifier: Key in database (JSON object).

        """
        database = self.load()  # DB betöltése
        del database[identifier]  # Adott bejegyzés törlése
        self.dump(database)  # DB kiírása
        L.info("Database %s succesfully updated!", self._filepath)

    # NOTE: delete_section2 is unnecessary, as existing sub-keys with None/null
    # value are preferred over missing keys

    def print_contents(self):
        """Print contents of JSON object."""

        database = self.load()
        print(f"\n---- Contents: ({self._filepath}) ----\n")
        if isinstance(database, list):
            for i in database:
                self.pretty_print(i)
        else:
            self.pretty_print(database)

    def pretty_print(self, printed_dict, indentation=0, tab="    "):
        """Pretty prints dictionary to console.

        :param printed_dict: Dictionary to be pretty printed. This can be a
            sub-dictionary, when called recursively.
        :type printed_dict: dict
        :param int indentation: Number of indentation. Incremented when called
            recursively.
        :param str tab: Indentation (default = "    ", *4 whitespaces*)

        """

        for key in printed_dict:
            value = printed_dict[key]  # Set value for understandability

            if isinstance(value, dict):  # If subkey is dict
                new_dict = printed_dict[key]  # New dictionary
                print(f"{indentation*tab}{key} : ")
                self.pretty_print(new_dict, indentation+1, tab)

            else:
                print(f"{indentation*tab}{key} : {value}")

    def __str__(self):
        """This function overloads the string conversion of the JSON object, to
        pretty prints the JSON object to console.

        """
        self.print_contents()
        return ""


# Include guard
if __name__ == "__main__":
    pass
