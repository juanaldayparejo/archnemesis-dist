.. raw:: html

   <p align="center">
     <img src="https://raw.githubusercontent.com/juanaldayparejo/archnemesis-dist/main/docs/images/archnemesis_logo_white_background.png" alt="archNEMESIS logo" width="400"/>
   </p>

=========================

.. image:: https://img.shields.io/badge/version-v1.0.1-red
  :target: https://doi.org/10.5281/zenodo.15123560

.. image:: https://img.shields.io/badge/readthedocs-lates-blue
   :target: https://archnemesis.readthedocs.io

.. image:: https://img.shields.io/badge/github-code-green
   :target: https://github.com/juanaldayparejo/archnemesis-dist

.. image:: https://img.shields.io/badge/archNEMESIS-reference-yellow
   :target: https://doi.org/10.48550/arXiv.2501.16452

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

The latest version of code has to be downloaded from `Github <https://github.com/juanaldayparejo/archnemesis-dist.git>`_ under a GNU General Public License v3. To do so, type in the command window:

.. code-block:: bash    

   git clone https://github.com/juanaldayparejo/archnemesis-dist.git
 

Before installing archNEMESIS, we recommend users to create and load a new python `virtual environment <https://docs.python.org/3/library/venv.html>`_ for a clean install:

.. code-block:: bash

   python -m venv name_of_virtual_environment/

   source name_of_virtual_environment/bin/activate


Then, we need to get into the archNEMESIS package folder using:

.. code-block:: bash

   cd archnemesis-dist

Finally, we need to install the library. Given that archNEMESIS is a highly dynamic package were new additions are frequently introduced, we recommend installing the package 
but keeping it editable by typing:

.. code-block:: bash
   
   pip install --editable .
 
This will install archNEMESIS, but with the ability to update any changes made to the code (e.g., when introducing new model parameterisations or methods). In addition, it will install all the required libraries archNEMESIS depends on.

Citing archNEMESIS
--------------------

If archNEMESIS has been significant in your research, we suggest citing the following articles:

- archNEMESIS reference publication:
   - Alday, J., Penn, J., Irwin, P. G. J., Mason, J. P., Yang, J. (2025). archNEMESIS: an open-source Python package for analysis of planetary atmospheric spectra. *Preprint in arXiv*. doi: `10.48550/ARXIV.2501.16452 <https://doi.org/10.48550/ARXIV.2501.16452>`_

- NEMESIS reference publication:
   - Irwin, P. G. J., Teanby, N. A., De Kok, R., Fletcher, L. N., Howett, C. J. A., Tsang, C. C. C., ... & Parrish, P. D. (2008). The NEMESIS planetary atmosphere radiative transfer and retrieval tool. *Journal of Quantitative Spectroscopy and Radiative Transfer*, 109(6), 1136-1150. doi: `10.1016/j.jqsrt.2007.11.006 <https://doi.org/10.1016/j.jqsrt.2007.11.006>`_

Revision history
-----------------------------

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
- Visualisations: `matplotlib <https://matplotlib.org/>`_; `basemap <https://matplotlib.org/basemap/stable/>`_
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


