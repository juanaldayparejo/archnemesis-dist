.. raw:: html

   <p align="center">
     <img src="https://raw.githubusercontent.com/juanaldayparejo/archnemesis-dist/main/docs/images/archnemesis_logo_white_background.png" alt="archNEMESIS logo" width="400"/>
   </p>

=========================

.. image:: https://img.shields.io/badge/version-v1.0.6-red
  :target: https://doi.org/10.5281/zenodo.17948742

.. image:: https://img.shields.io/badge/readthedocs-latest-blue
   :target: https://archnemesis.readthedocs.io

.. image:: https://img.shields.io/badge/github-code-green
   :target: https://github.com/juanaldayparejo/archnemesis-dist

.. image:: https://img.shields.io/badge/archNEMESIS-reference-yellow
   :target: https://doi.org/10.5334/jors.554

.. image:: https://img.shields.io/badge/NEMESIS-reference-yellow
   :target: https://doi.org/10.1016/j.jqsrt.2007.11.006

.. image:: https://img.shields.io/badge/discord-join-pink
   :target: https://discord.gg/Te43qbrVFK
__________

ArchNEMESIS is an open source Python package developed for the analysis of remote sensing spectroscopic observations of planetary atmospheres. 
It is based on the widely used NEMESIS (Non-linear Optimal Estimator for MultivariatE Spectral analySIS) radiative transfer and retrieval tool, 
which has been extensively used for the investigation of a wide variety of planetary environments.

ArchNEMESIS is currently maintained by `Juan Alday <https://research.open.ac.uk/people/ja22256>`_ and `Joseph Penn <https://www.physics.ox.ac.uk/our-people/penn>`_.
The `NEMESIS <https://nemesiscode.github.io/index.html>`_ algorithm, code archNEMESIS is based on, was originally developed by `Patrick Irwin <https://www.physics.ox.ac.uk/our-people/irwin>`_.

In this website, we aim to provide a detailed description of the code and its functionalities. In addition, we include several jupyter notebooks
to help users get used to some of these functionalities. 

If interested users are missing key points in the documentation, would appreciate seeing jupyter notebooks for certain purposes, or want to report issues, please do so by contacting us or joining our `Discord <https://discord.gg/Te43qbrVFK>`_ channel.

Installation
--------------------

There are three main ways to install archNEMESIS, depending on your use case:

Installing from PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to install archNEMESIS is via PyPI.
We recommend doing this inside a clean Python virtual environment:

.. code-block:: bash

   python -m venv archnemesis-env
   source archnemesis-env/bin/activate
   pip install archnemesis

This will install the latest stable release of the package along with its dependencies.
It is the recommended method if you just want to use the library without editing the source code.


Installing from GitHub (developer mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the latest development version, clone the GitHub repository:

.. code-block:: bash    

   git clone https://github.com/juanaldayparejo/archnemesis-dist.git
 
Before installing archNEMESIS, we recommend users to create and load a new Python `virtual environment <https://docs.python.org/3/library/venv.html>`_ for a clean install:

.. code-block:: bash

   python -m venv name_of_virtual_environment/
   source name_of_virtual_environment/bin/activate

Then move into the package directory:

.. code-block:: bash

   cd archnemesis-dist

Finally, install the library in editable mode:

.. code-block:: bash

   pip install --editable .

This will install archNEMESIS along with all required dependencies, while keeping the source editable.


Citing archNEMESIS
--------------------

If archNEMESIS has been significant in your research, we suggest citing the following articles:

- archNEMESIS reference publication:
   - Alday, J., Penn, J., Irwin, P., Mason, J., Yang, J. and Dobinson, J. (2025) archNEMESIS: An Open-Source Python Package for Analysis of Planetary Atmospheric Spectra, *Journal of Open Research Software*, 13(1), p. 10. doi: `10.5334/jors.554 <https://doi.org/10.5334/jors.554>`_


- NEMESIS reference publication:
   - Irwin, P. G. J., Teanby, N. A., De Kok, R., Fletcher, L. N., Howett, C. J. A., Tsang, C. C. C., ... & Parrish, P. D. (2008). The NEMESIS planetary atmosphere radiative transfer and retrieval tool. *Journal of Quantitative Spectroscopy and Radiative Transfer*, 109(6), 1136-1150. doi: `10.1016/j.jqsrt.2007.11.006 <https://doi.org/10.1016/j.jqsrt.2007.11.006>`_



Revision history
-----------------------------

- `1.0.6 <https://doi.org/10.5281/zenodo.17948742>`_ (16 December, 2025)
   - Fixing bugs to reconcile results with NEMESIS.
   - Implemented LineData class with functionality to extract data from HITRAN using HAPI.
   - Implemented first version of cross section calculations.
   - Combined radiative transfer calculations with and without gradients in same function (CIRSradg is now CIRSrad(gradients=True)).
   - Added forward modelling error capabilities.
   - First implementation of nemesisdisc, forward model type for gradient calculations (no scattering) and multiple averaging points.
   - Added functionality to read information from previous retrievals (LIN parameter).
   - Implemented AOTF spectrometer modelling capabilities in nemesisSO.
   - Implemented integration of signal across filters in nemesis (for modelling spectra from radiometers).

- `1.0.5 <https://doi.org/10.5281/zenodo.15839794>`_ (8 July, 2025)
    - New release of archNEMESIS for publication at Journal of Open Research Software.
    - Fixed minor bugs throughout the code.
    - Added unit test for calculation of optical properties using Mie Theory.
    - Updated setup file to avoid dependency conflicts.

- `1.0.4 <https://doi.org/10.5281/zenodo.15789739>`_ (2 July, 2025)
    - New release of archNEMESIS for release on PyPI and Docker Hub.

- `1.0.3 <https://doi.org/10.5281/zenodo.15699119>`_ (19 June, 2025)
    - New release of archNEMESIS for first release on PyPI.

- `1.0.2 <https://doi.org/10.5281/zenodo.15698743>`_ (19 June, 2025)
    - Fixed minor bugs throughout the code.
    - Included new model parameterisations.
    - Included new automatic tests (e.g., forward models for solar occultation and limb geometry).
    - Flags now identified with ENUMS rather than magic numbers.
    - Model parameterisations now defined as classes.

- `1.0.1 <https://doi.org/10.5281/zenodo.15123560>`_ (2 April, 2025)
    - Fixed minor bugs throughout the code.
    - Implementation of Oren-Nayar surface reflectance model.
    - Implementation of different surface reflectance models in multiple scattering calculations.
    - Included new automatic tests.
    - Included new model parameterisations.

- `1.0.0 <https://doi.org/10.5281/zenodo.14746548>`_ (27 January, 2025)
    - First release for publication at Journal of Open Research Software.

Dependencies
-----------------------------

- Numerical calculations: `numpy <https://numpy.org/>`_; `scipy <https://scipy.org/>`_
- Visualisations: `matplotlib <https://matplotlib.org/>`_; `basemap <https://matplotlib.org/basemap/stable/>`_; `corner <https://corner.readthedocs.io/en/latest/>`_
- File handling: `h5py <https://www.h5py.org/>`_
- Optimisation: `numba <https://numba.pydata.org/>`_; `joblib <https://joblib.readthedocs.io/en/stable/>`_
- Nested sampling: `pymultinest <https://johannesbuchner.github.io/PyMultiNest/>`_ 
- Extraction of ERA-5 model profiles: `cdsapi <https://pypi.org/project/cdsapi/>`_; `pygrib <https://jswhit.github.io/pygrib/>`_  

.. toctree::
   :maxdepth: 2

.. toctree::
   :caption: General Structure
   :hidden:
   
   documentation/general_structure.ipynb
 
.. toctree::
   :caption: Reference classes
   :hidden:
   
   documentation/reference_classes.ipynb

.. toctree::
   :caption: Model parameterisations
   :hidden:
   
   documentation/model_parameterisations.ipynb

.. toctree::
   :caption: Forward Model
   :hidden:
   
   documentation/forward_model.ipynb
 
.. toctree::
   :caption: Retrievals
   :hidden:
   
   documentation/retrievals.ipynb
   
.. toctree::
   :caption: Examples
   :hidden:
   
   examples


