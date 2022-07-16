FEniCS_ice
----------

**FEniCS_ice is a finite element model framework, 
that quantifies the initialization uncertainty 
for time-dependent ice sheet models.**

FEniCS_ice is a Python code intended for Bayesian quantification of uncertainty 
of large-scale parameter sets within an ice-sheet flow model. 
The implementation consist of a combination of data assimilation and bayesian inference.
This involves 1) inferring unobserved model parameters from data; 2) determining
the uncertainty of the inferred model parameters; 3) running the forward model to make a
prediction about a quantity of interest (e.g. ice mass loss); 4) Propagating the uncertainty 
in the inferred parameters to the model prediction. 

The code contains a dynamic solver for ice-sheet flow which implements the 
Shallow-Shelf Approximation (`MacAyeal et al, 1989.`_)
and approximates parameter covariance using a low-rank approximation to 
the inverse of the cost-function Hessian. The code uses Algorithmic Differentiation 
to find sensitivity of time-evolving Quantities of Interest to parameter sets, 
allowing projections of parameter uncertainty forward in time.

.. _MacAyeal et al, 1989.: https://doi.org/10.1029/JB094iB04p04071


Installation, documentation
---------------------------

The documentation website currently in construction,
in the meantime there is a `User guide`_ and the basic installation info.

.. _User guide: https://github.com/EdiGlacUQ/fenics_ice/tree/main/user_guide

Conda installation:
------------------

1. Clone the repository::

    git clone https://github.com/EdiGlacUQ/fenics_ice.git
2. To install via `Conda`_ use `install.sh`_, this script will install and test FEniCS_ice.
Be sure to set **CONDA_HOME** before installing, and add the line below to your .bashrc::

    export FENICS_ICE_BASE_DIR="/path/to/fenics_ice/repo"
3. Run install.sh.

4. Run all serial tests::

    pytest -v --order-scope=module --color=yes
5. Run all parallel tests::

    mpirun -n 2 pytest -v --order-scope=module --color=yes
To install via Mamba:
---------------------

1. Clone the repository and create `Mamba`_ environment::

    cd fenics_ice
    mamba env create -f environment.yml
    pip install -e .
Make sure the environment "fenics_ice" is activated.

2. Install `tlm_adjoint`_::

    git clone https://github.com/EdiGlacUQ/tlm_adjoint.git
    cd tlm_adjoint
    pip install -e .
3. Run all serial tests::

    pytest -v --order-scope=module --color=yes
4. Run all parallel tests::

    mpirun -n 2 pytest -v --order-scope=module --color=yes

Code source
------------

- View the source code `on GitHub`_.
- Report bugs or share your ideas on the `issue tracker`_.
- Improve the model by submitting a `pull request`_.

.. _on GitHub: https://github.com/EdiGlacUQ/fenics_ice
.. _issue tracker: https://github.com/EdiGlacUQ/fenics_ice/issues
.. _pull request: https://github.com/EdiGlacUQ/fenics_ice/pulls

.. _Conda: https://docs.conda.io/en/latest/miniconda.html
.. _install.sh: https://github.com/EdiGlacUQ/fenics_ice/blob/main/install.sh
.. _Mamba: https://mamba.readthedocs.io/en/latest/installation.html#micromamba
.. _tlm_adjoint: https://github.com/EdiGlacUQ/tlm_adjoint

About
-----
:Version:
    .. image:: https://img.shields.io/badge/python-3.8%2B-blue
        :target: https://www.python.org/
        :alt: Pypi version
        
:Citation:
    .. image:: https://img.shields.io/badge/Citation-GMD%20paper-orange.svg
        :target: https://doi.org/10.5194/gmd-14-5843-2021
        :alt: GMD Paper

    .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5153231.svg
        :target: https://zenodo.org/record/5153231
        :alt: Zenodo

:Tests:       
    .. image:: https://img.shields.io/badge/test-passing-green
        :target: https://github.com/EdiGlacUQ/fenics_ice/actions/workflows/test-fice.yml
        :alt: Linux build status

    .. image:: 
        :target: 
        :alt: Documentation in construction


:License:
    .. image:: https://img.shields.io/badge/license-GNU--LGPL--v3-green
        :target: https://github.com/EdiGlacUQ/fenics_ice/blob/main/COPYING
        :alt: GNU LGPL version 3

:Authors:

    See the `link`_ for a list of all contributors.

    .. _link: https://github.com/EdiGlacUQ/fenics_ice/people