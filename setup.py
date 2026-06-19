# Compatibility shim for older pip (<22.0) that does not support
# PEP 660 editable installs from pyproject.toml alone. All real metadata
# (dependencies, scripts, optional-dependencies) lives in pyproject.toml;
# this file just delegates to setuptools so `pip install -e .` works on
# Python 3.10's bundled pip too.
from setuptools import setup

setup()
