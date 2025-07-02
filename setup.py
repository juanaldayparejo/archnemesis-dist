from setuptools import setup

setup(name='archnemesis',
      version='1.0.4',
      description='Python implementation of the NEMESIS radiative transfer and retrieval code',
      packages=['archnemesis'],
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
      )
