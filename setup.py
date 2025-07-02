from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="archnemesis",
    version="1.0.4",
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
    packages=["archnemesis"],
    install_requires=[
      'numpy',
      'matplotlib',
      'numba>=0.57.0',
      'scipy',
      'pymultinest',
      'cdsapi',
      'pygrib',
      'joblib',
      'h5py',
      'basemap',
      'pytest'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
