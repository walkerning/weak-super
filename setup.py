# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

# meta infos
NAME = "wksuper"
DESCRIPTION = "weak supervision"
VERSION = "0.1"

AUTHOR = "babaandbaobao"

# package contents
MODULES = []
PACKAGES = find_packages(exclude=["tests", "tests.*"])

# dependencies
# also need cv2 to work, sadly there is no such non-python dependency
# management now...
INSTALL_REQUIRES = [
    "toml==0.9.1",
    "numpy==1.11.0",
    "scipy==0.17.1",
    "scikit-image==0.12.3",
    "matplotlib==1.5.1",
    "scikit-learn==0.17.1",
    "Cython==0.24",
    "psutil==4.3.0", # poped by pip freeze
]
TESTS_REQUIRE = [
    "pytest==2.9.2"
]

# entry points
ENTRY_POINTS = """
[console_scripts]
wks-train=wksuper:main
wks-test=wksuper:test_main
wks-clean-cache=wksuper.helper:clean_cache_main
"""

# extensions
EXTENSIONS = [
    Extension(
        "wksuper.nms.cpu_nms",
        ["wksuper/nms/cpu_nms.pyx"]),
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,

    py_modules=MODULES,
    packages=PACKAGES,

    entry_points=ENTRY_POINTS,
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    
    ext_modules=cythonize(EXTENSIONS),
)
