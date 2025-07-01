Here, we include several jupyter notebooks that demonstrate the functionalities of archNEMESIS. These examples cover a range of topics, including reference classes, forward models, and retrievals. Each section provides practical examples to help users understand how to utilise archNEMESIS effectively.
The majority of the examples include all the necessary files to run the notebooks, but some of them require additional heavy data files (e.g., line-by-line and correlated-k look-up tables) that are available upon request. The examples are designed to be self-contained and easy to follow, making it simple for users to get started with archNEMESIS.

The source input files and notebooks for each specific example can be found in the `docs/examples` folder of the archNEMESIS repository. 

Reference classes
------------------

This section includes Jupyter notebooks with examples showing how to use the different functionalities of archNEMESIS for the pre-processing and post-processing of forward models and retrievals. 

.. nbgallery::
 
   examples/makephase/run_makephase.ipynb
   examples/surface_modes/surface.ipynb
   examples/atmosphere_tutorial/atmosphere_tutorial.ipynb
   examples/measurement/measurement_class.ipynb
   examples/stellar/StellarExample.ipynb
   examples/layer_example/layer_example.ipynb
   examples/telluric_example/example_telluric.ipynb
   examples/create_UV_lta/create_UV_lta.ipynb
   examples/lookup_tables/lookup_tables.ipynb
   examples/cia_archnemesis/convert_cia_nemesis.ipynb
   examples/stellar/noaa_solar_spectrum.ipynb


Forward models
------------------

This sections includes Jupyter notebookes with examples showing how archNEMESIS can be used to calculate the forward model for different planetary atmosphere and in several observing geometries.

.. nbgallery::

   examples/mars_solocc/mars_SO.ipynb
   examples/Jupiter_CIRS_nadir_thermal_emission/Jupiter_CIRS.ipynb
   examples/mars_groundbased/mars_groundbased.ipynb
   examples/Mars_DISORT/archnemesis_disort_comparisons.ipynb
   
Retrievals
------------------

This sections includes Jupyter notebookes with examples showing how archNEMESIS can be used to perform retrievals in different scenarios.

.. nbgallery::

   examples/retrieval_Jupiter_Tprofile/Jupiter_CIRS_retrieval.ipynb
   examples/Neptune_JWST_nested_sampling/Neptune_JWST.ipynb
   
