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
    "sphinx.ext.intersphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_parser",
    ]

myst_enable_extensions = [
    "colon_fence",
]

#Creating gas_table.html file for documentation
##################################################################################

from pathlib import Path
from archnemesis.Data.gas_data import generate_isotopologue_table_html

docdir = Path(__file__).parent
outfile = Path(__file__).parent / "_static" / "gas_table.html"

html = generate_isotopologue_table_html()

# Only overwrite if the contents changed
if (not outfile.exists()) or (outfile.read_text(encoding="utf-8") != html):
    outfile.write_text(html, encoding="utf-8")


#Defining paramters for autodoc documentation
##################################################################################

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
    'examples/Jupiter_CIRS_nadir_thermal_emission/Jupiter_CIRS': '_static/jupiter_cassini.jpg',
    #'examples/Measurement/Measurement': '_static/observation_sketch.png',
    'examples/mars_rover/mars_rover': '_static/mars_rover.jpg',
    'examples/Mars_DISORT/archnemesis_disort_comparisons': '_static/mars_orbiter.jpg',
    'examples/mars_groundbased/mars_groundbased': '_static/nasa_irtf.jpg',
}

#Adding logo
html_logo = "images/archnemesis_logo_white_background.png"

#Defining the actual appearance of the website
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static','images']

html_css_files = [
    'css/model_reference_list.css',
    'css/api.css',
]

exclude_patterns = ['_build', '**.ipynb_checkpoints']

html_theme_options = {
    'logo_only': True,  # Don't show project name next to logo
    'display_version': True,  # Still show version below logo if you want
}


extensions.append("autoapi.extension")
autoapi_dirs = [f'{docdir.parent / "archnemesis"}']


autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",  # Adds a clean overview table at the top of pages
    #"imported-members",
    #"special-members",
]
autoapi_member_order = "bysource"  # Matches file structure instead of A-Z alphabetical

autodoc_typehints = "description"
autodoc_typehints_format = "short"

autoapi_python_class_content = "both"
autoapi_add_toctree_entry = True
autoapi_keep_files = True
add_module_names = False

#autoapi_own_page_level = "class"
autoapi_own_page_level = "function"
autoapi_own_page_types = ['class', 'function', 'data', 'exception']



# NOTE: Dataclasses are not documented in the API properly yet. I am trying to fix them.
"""
import sys


def print_all_decorators(x, level=0):
    for k,v in x.obj.items():
        print(k)
    #print(x.obj)
    #print(' '*level + )
    if (d:=getattr(x, 'decorators', None)) is not None:
        print(f'{d=}')
        
    if level==3:
        sys.exit("LEVEL")
    #for y in x.children:
    #    print_all_decorators(y,level+1)

def walk_ast_and_fix_dataclasses(app):
    for k,v in app.env.autoapi_objects.items():
        if k.endswith('Model444'):
            print(f'{k=}')
            print_all_decorators(v)
    
    sys.exit('objects')


def setup(app):
    # Connect directly to the stage right after AutoAPI finishes walking the AST,
    # but right before Sphinx writes the template files to disk.
    app.connect("builder-inited", walk_ast_and_fix_dataclasses)
"""