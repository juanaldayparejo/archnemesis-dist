# flake8: noqa # Do not check this file as it is for forward imports

# NOTE: This is just here to make 'flake8' play nice with the type hints
# the problem is that importing Variables_0 or ForwardModel_0 creates a circular import
# this actually means that I should possibly redesign how those work to avoid circular imports
# but that is outside the scope of what I want to accomplish here
from archnemesis.Variables_0 import Variables_0
from archnemesis.ForwardModel_0 import ForwardModel_0
#from archnemesis.Scatter_0 import Scatter_0
from archnemesis.Atmosphere_0 import Atmosphere_0

nx = 'number of elements in state vector'
m = 'an undetermined number, but probably less than "nx"'
mx = 'synonym for nx'
mparam = 'the number of parameters a model has'
nparam = 'the number of parameters a model has'
NCONV = 'number of spectral bins'
NGEOM = 'number of geometries'
NX = 'number of elements in state vector'
NDEGREE = 'number of degrees in a polynomial'
NWINDOWS = 'number of spectral windows'