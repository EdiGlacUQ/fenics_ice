#!/usr/bin/env python3
import setuptools
if __name__ == "__main__":
    setuptools.setup(
    name="fenics_ice",
    description="A finite element model framework "
                "that quantifies the initialization uncertainty for "
                "time-dependent ice sheet models.",
    url="https://github.com/EdiGlacUQ/fenics_ice.git",
    license="GNU LGPL version 3",
    packages=["fenics_ice"],
    python_requires=">=3.8")