from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="archnemesis",
    version="1.0.6",
    author="Juan Alday",
    description="Python implementation of the NEMESIS radiative transfer and retrieval code",
    long_description=long_description,
    long_description_content_type="text/markdown",  # important for Markdown rendering
    url="https://github.com/juanaldayparejo/archnemesis-dist",
    project_urls={
        "Documentation": "https://archnemesis.readthedocs.io",
        "Source": "https://github.com/juanaldayparejo/archnemesis-dist",
        "Tracker": "https://github.com/juanaldayparejo/archnemesis-dist/issues",
        "DockerHub": "https://hub.docker.com/r/juanaldayparejo/archnemesis",
    },
    #packages=["archnemesis"],
    packages=find_packages(),  #automatically include all subpackages like archnemesis.cfg
    install_requires=[
      'numpy',
      'matplotlib',
      'numba>=0.57.0',
      'scipy',
      'pymultinest',
      'cdsapi',
      'joblib',
      'h5py',
      'basemap',
      'pytest',
      'corner',
      'typing_extensions'
    ],
    extras_require={
        'grib': ['pygrib'],
        'docs': ['sphinx', 'sphinx_rtd_theme'],
        'spectroscopy': ['hitran-api'], # The HITRAN api module "hapi" is called "hitran-api" on pypi
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
