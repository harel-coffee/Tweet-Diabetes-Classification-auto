"""

    Util functions for general system purposes

    Author: Adrian Ahne

    Date: 25-07-2018

"""

import sys

def load_library(path):
    """
        load library of the given path
    """

    if path not in sys.path:
        sys.path.insert(0, path)
