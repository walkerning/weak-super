# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

# meta infos
NAME = "wksuper"
DESCRIPTION = "weak supervision"
VERSION = "0.1"

AUTHOR = "babaandbaobao"

# package contents
MODULES = []
PACKAGES = find_packages(exclude=["tests", "tests.*"])

# dependencies
INSTALL_REQUIRES = [
    "toml"
]
TESTS_REQUIRE = []

# entry points
ENTRY_POINTS = """
[console_scripts]
wks-train=wksuper.trainer:main
"""

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,

    py_modules=MODULES,
    packages=PACKAGES,

    entry_points=ENTRY_POINTS,
    zip_safe=True,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
