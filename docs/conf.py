import sys
import os
sys.path.insert(0,os.path.abspath('../archnemesis/Path/'))

project = 'archNEMESIS'

extensions = [
    "nbsphinx",
    "sphinx_gallery.load_style",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
#    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_console_highlighting",
    ]


#Defining paramters for autodoc documentation
#napoleon_google_docstring = False
#napoleon_numpy_docstring = True
#napoleon_include_init_with_doc = False
#napoleon_include_private_with_doc = True
#napoleon_include_special_with_doc = True
#napoleon_use_admonition_for_examples = True
#napoleon_use_admonition_for_notes = True
#napoleon_use_admonition_for_references = False
#napoleon_use_ivar = True
#napoleon_use_keyword = True
#napoleon_use_param = True
#napoleon_use_rtype = True
#napoleon_preprocess_types = False
#napoleon_type_aliases = None
#napoleon_attr_annotations = False


#Defining each image shown in the gallery
nbsphinx_thumbnails = {
    'examples/makephase/run_makephase': '_static/mars_sunset2.jpg',
    'examples/surface_modes/surface': '_static/mars_duststorm.jpg',
    'examples/atmosphere_tutorial/atmosphere_tutorial': '_static/planetary_atmospheres.png',
    'examples/stellar/StellarExample': '_static/solar_spec.jpg',
    'examples/measurement/measurement_class': '_static/observation_sketch.png',
    'examples/mars_solocc/mars_SO': '_static/exomars_SO.jpg',
    'examples/mars_aotf/mars_aotf': '_static/exomars_SO.jpg',
    'examples/disc_weights/disc_weights': '_static/exoplanet_orbit.png',
    'examples/Exoplanet_thermal_emission/exoplanet': '_static/exoplanet_orbit.png',
    'examples/Exoplanet_primary_transit/exoplanet': '_static/exoplanet_orbit.png',
    'examples/retrieval_exoplanet_transit/retrieval_exoplanet': '_static/exoplanet_orbit.png',
    #'examples/Jupiter_CIRS_nadir_thermal_emission/Jupiter_CIRS': '_static/jupiter_cassini.jpg',
    #'examples/Measurement/Measurement': '_static/observation_sketch.png',
    'examples/mars_rover/mars_rover': '_static/mars_rover.jpg',
}

#Adding logo
html_logo = "images/archnemesis_logo_white_background.png"

#Defining the actual appearance of the website
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static','images']

exclude_patterns = ['_build', '**.ipynb_checkpoints']

html_theme_options = {
    'logo_only': True,  # Don't show project name next to logo
    'display_version': True,  # Still show version below logo if you want
}
