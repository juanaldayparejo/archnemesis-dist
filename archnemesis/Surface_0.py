from __future__ import annotations #  for 3.9 compatability
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import os
from numba import jit, njit
from archnemesis.enums import WaveUnit, LowerBoundaryCondition

from archnemesis.helpers import h5py_helper

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

###############################################################################################

"""
Created on Tue Jul 22 17:27:12 2021

@author: juanalday

[JD] We might want to rework this class to be a bit more general e.g. "LowerBoundary" so
     that it more accurately specifies what it does. That way it makes a bit more sense to use
     it for gas giants. Really, we should make an interface 'LowerBoundaryCondition' and then
     have subclasses for each of the different types of lower boundary conditions.
"""

class Surface_0:

    def __init__(self, GASGIANT=False, ISPACE=WaveUnit.Wavenumber_cm, LOWBC=LowerBoundaryCondition.LAMBERTIAN, GALB=-1.0, NEM=2, NLOCATIONS=1):

        """
        Inputs
        ------
        @param GASGIANT: log,
            Flag indicating whether the planet has surface (GASGIANT=False) or not (GASGIANT=True)

        @param ISPACE: int
            Spectral units
                0 :: Wavenumber (cm-1)
                1 :: Wavelength (um)

        @param LOWBC: int,
            Flag indicating the lower boundary condition.
                0 :: Thermal emission only (i.e. no reflection)
                1 :: Lambertian surface
                2 :: Hapke surface
 
        @param GALB: int,
            Ground albedo
            
        @param NEM: int,
            Number of spectral points defining the emissivity of the surface   

        @param NLOCATIONS: int,
            Number of surface points (i.e. different latitudes/longitudes with different properties)
        
        Attributes
        ----------

        @attribute LATITUDE: real or 1D array (depending on number of locations)
            Latitude of each location (degree)

        @attribute LONGITUDE: real or 1D array (depending on number of locations)
            Longitude of each location (degree)

        @attribute TSURF: real or 1D array (depending on number of locations)
            Surface temperature (K)

        Attributes for Thermal emission from surface:
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        @attribute VEM: 1D array
            Wavelengths at which the emissivity and other surface parameters are defined
            Assumed to be equal for all locations

        @attribute EMISSIVITY: 1D array or 2D array (depending on number of locations)
            Surface emissitivity 


        Attributes for Lambertian surface:
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        None. The albedo of a lambertian surface is given in the .set file by GALB, 
        and if GALB<0, it is calculated as 1.0 - EMISSIVITY.


        Attributes for Hapke surface:
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        The method implemented here is derived in Hapke (2012) - Theory of Reflectance and Emittance Spectroscopy
        In particular, it is derived in chapter 12.3.1 of that book. We also assume that the scattering phase function
        of the surface is given by a Double Henyey-Greenstein function. 

        @attribute SGLALB : 1D array or 2D array (depending on number of locations)
            Single scattering albedo w

        @attribute K : 1D array or 2D array (depending on number of locations)
            Porosity coefficient 

        @attribute BS0 : 1D array or 2D array (depending on number of locations)
            Amplitude of opposition effect (0 to 1)    

        @attribute hs : 1D array or 2D array (depending on number of locations)
            Width of the opposition surge

        @attribute BC0 : 1D array or 2D array (depending on number of locations)
            Amplitude of the coherent backscatter opposition effect (0 to 1)

        @attribute hc : 1D array or 2D array (depending on number of locations)
            Width of the backscatter function

        @attribute ROUGHNESS : 1D array or 2D array (depending on number of locations)
            Roughness mean slope angle (degrees)

        @attribute G1 : 1D array or 2D array (depending on number of locations)
            Asymmetry factor of the first Henyey-Greenstein function defining the phase function

        @attribute G2 : 1D array or 2D array (depending on number of locations)
            Asymmetry factor of the second Henyey-Greenstein function defining the phase function
        
        @attribute F: 1D array or 2D array (depending on number of locations)
            Parameter defining the relative contribution of G1 and G2 of the double Henyey-Greenstein phase function


        Attributes for Oren-Nayar surface reflection:
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        The method implemented here is described in Oren & Nayar (1998). This method is a generalisation of 
        the Lambertian reflection model to account for the roughness of the surface. In the case that the
        roughness is zero, this model converges to a Lambertian surface.

        @attribute ALBEDO : 1D array or 2D array (depending on number of locations)
            Lambert albedo
            
        @attribute ROUGHNESS : 1D array or 2D array (depending on number of locations)
            Roughness mean slope angle (degrees)

        Methods
        -------
        Surface_0.assess()

        Surface_0.write_hdf5()
        Surface_0.read_hdf5()

        Surface_0.read_sur()
        Surface_0.read_hap()

        Surface_0.edit_EMISSIVITY()
        Surface_0.edit_SGLALB()
        Surface_0.edit_BS0()
        Surface_0.edit_hs()
        Surface_0.edit_BC0()
        Surface_0.edit_hc()
        Surface_0.edit_K()
        Surface_0.edit_ROUGHNESS()
        Surface_0.edit_G1()
        Surface_0.edit_G2()
        Surface_0.edit_F()
        
        Surface_0.select_location()
        """

        #Input parameters
        self.NLOCATIONS = NLOCATIONS
        self.GASGIANT = GASGIANT
        #self.ISPACE = WaveUnit(ISPACE) if not isinstance(ISPACE, WaveUnit) else ISPACE
        #self.LOWBC = LowerBoundaryCondition(LOWBC) if not isinstance(LOWBC, LowerBoundaryCondition) else LOWBC
        self.GALB = GALB
        self.NEM = NEM

        # Input the following profiles using the edit_ methods.
        self.LATITUDE = None   #float or (NLOCATIONS) 
        self.LONGITUDE = None  #float or (NLOCATIONS) 
        self.TSURF = None      #float or (NLOCATIONS) 
        self.VEM = None #(NEM) or (NEM)
        self.EMISSIVITY = None #(NEM) or (NEM,NLOCATIONS)

        #Hapke parameters
        self.SGLALB = None #(NEM) or (NEM,NLOCATIONS)
        self.BS0 = None #(NEM) or (NEM,NLOCATIONS)
        self.hs = None #(NEM) or (NEM,NLOCATIONS)
        self.BC0 = None #(NEM) or (NEM,NLOCATIONS)
        self.hc = None #(NEM) or (NEM,NLOCATIONS)
        self.K = None #(NEM) or (NEM,NLOCATIONS)
        self.ROUGHNESS = None #(NEM) or (NEM,NLOCATIONS)
        self.G1 = None #(NEM) or (NEM,NLOCATIONS)
        self.G2 = None #(NEM) or (NEM,NLOCATIONS)
        self.F = None #(NEM) or (NEM,NLOCATIONS)

        # private attributes
        self._ispace = None
        self._lowbc = None
        
        # set property values
        self.ISPACE = ISPACE if ISPACE is not None else WaveUnit.Wavenumber_cm
        self.LOWBC = LOWBC if LOWBC is not None else LowerBoundaryCondition.THERMAL
    
    
    @property
    def ISPACE(self) -> WaveUnit:
        return self._ispace
    
    @ISPACE.setter
    def ISPACE(self, value):
        self._ispace = WaveUnit(value)
    
    @property
    def LOWBC(self) -> LowerBoundaryCondition:
        return self._lowbc
    
    @LOWBC.setter
    def LOWBC(self, value):
        self._lowbc = LowerBoundaryCondition(value)
        


    def assess(self):
        """
        Assess whether the different variables have the correct dimensions and types
        """

        if self.GASGIANT==False:

            #Checking some common parameters to all cases
            assert isinstance(self.LOWBC, (int, np.integer, LowerBoundaryCondition)), 'LOWBC must be int or LowerBoundaryCondition'
            assert self.LOWBC in LowerBoundaryCondition, f'LOWBC must be one of {tuple(LowerBoundaryCondition)}'
            
            assert len(self.VEM) == self.NEM , 'VEM must have size (NEM)'
            
            assert isinstance(self.ISPACE, (int, np.integer, WaveUnit)), 'ISPACE must be int or WaveUnit'
            assert self.ISPACE in WaveUnit, f'ISPACE must be one of {tuple(WaveUnit)}'

            #Determining sizes based on the number of surface locations
            if self.NLOCATIONS<=0:
                raise ValueError('error :: NLOCATIONS must be greater than 0')

            elif self.NLOCATIONS==1:

                assert np.issubdtype(type(self.TSURF), np.float64) == True , \
                    'TSURF must be float'
                assert np.issubdtype(type(self.LATITUDE), np.float64) == True , \
                    'LATITUDE must be float'
                assert abs(self.LATITUDE) < 90.0 , \
                    'LATITUDE must be within -90 to 90 degrees'
                assert np.issubdtype(type(self.LONGITUDE), np.float64) == True , \
                    'LONGITUDE must be float'

                assert len(self.EMISSIVITY) == self.NEM , \
                    'EMISSIVITY must have size (NEM)'

                #Special case for Hapke reflection
                if self.LOWBC==LowerBoundaryCondition.HAPKE:
                    assert len(self.SGLALB) == self.NEM , \
                        'SGLALB must have size (NEM)'
                    assert len(self.ROUGHNESS) == self.NEM , \
                        'ROUGHNESS must have size (NEM)'
                    assert len(self.BS0) == self.NEM , \
                        'BS0 must have size (NEM)'
                    assert len(self.hs) == self.NEM , \
                        'hs must have size (NEM)'
                    assert len(self.BC0) == self.NEM , \
                        'BC0 must have size (NEM)'
                    assert len(self.hc) == self.NEM , \
                        'hc must have size (NEM)'
                    assert len(self.K) == self.NEM , \
                        'K must have size (NEM)'
                    assert len(self.G1) == self.NEM , \
                        'G1 must have size (NEM)'
                    assert len(self.G2) == self.NEM , \
                        'G2 must have size (NEM)'
                    assert len(self.F) == self.NEM , \
                        'F must have size (NEM)'
                        
                elif self.LOWBC==3:
                    assert len(self.ALBEDO) == self.NEM , \
                        'ALBEDO must have size (NEM)'
                    assert len(self.ROUGHNESS) == self.NEM , \
                        'ROUGHNESS must have size (NEM)'
                        
            else:
                assert len(self.TSURF) == self.NLOCATIONS , \
                    'TSURF must have size (NLOCATIONS)'
                assert len(self.LATITUDE) == self.NLOCATIONS , \
                    'LATITUDE must have size (NLOCATIONS)'
                assert len(self.LONGITUDE) == self.NLOCATIONS , \
                    'LONGITUDE must have size (NLOCATIONS)'
                
                assert self.EMISSIVITY.shape == (self.NEM,self.NLOCATIONS) , \
                    'EMISSIVITY must have size (NEM,NLOCATIONS)'
                
                #Special case for Hapke reflection
                if self.LOWBC==LowerBoundaryCondition.HAPKE:
                    assert self.SGLALB.shape == (self.NEM,self.NLOCATIONS) , \
                        'SGLALB must have size (NEM,NLOCATIONS)'
                    assert self.BS0.shape == (self.NEM,self.NLOCATIONS) , \
                        'BS0 must have size (NEM,NLOCATIONS)'
                    assert self.hs.shape == (self.NEM,self.NLOCATIONS) , \
                        'hs must have size (NEM,NLOCATIONS)'
                    assert self.BC0.shape == (self.NEM,self.NLOCATIONS) , \
                        'BC0 must have size (NEM,NLOCATIONS)'
                    assert self.hc.shape == (self.NEM,self.NLOCATIONS) , \
                        'hc must have size (NEM,NLOCATIONS)'
                    assert self.K.shape == (self.NEM,self.NLOCATIONS) , \
                        'K must have size (NEM,NLOCATIONS)'
                    assert self.ROUGHNESS.shape == (self.NEM,self.NLOCATIONS) , \
                        'ROUGHNESS must have size (NEM,NLOCATIONS)'
                    assert self.G1.shape == (self.NEM,self.NLOCATIONS) , \
                        'G1 must have size (NEM,NLOCATIONS)'
                    assert self.G2.shape == (self.NEM,self.NLOCATIONS) , \
                        'G2 must have size (NEM,NLOCATIONS)'
                    assert self.F.shape == (self.NEM,self.NLOCATIONS) , \
                        'F must have size (NEM,NLOCATIONS)'
                        
                #Special case for Oren-Nayar reflection
                elif self.LOWBC==3:
                    assert self.ALBEDO.shape == (self.NEM,self.NLOCATIONS) , \
                        'ALBEDO must have size (NEM,NLOCATIONS)'
                    assert self.ROUGHNESS.shape == (self.NEM,self.NLOCATIONS) , \
                        'ROUGHNESS must have size (NEM,NLOCATIONS)'

        else:
            assert self.TSURF == 0.0, \
                "if GASGIANT=True then no emission from surface"
            assert self.GALB == 0.0, \
                "if GASGIANT=True then surface absorbs everything"
            assert self.LOWBC == LowerBoundaryCondition.THERMAL , \
                'If GASGIANT=True then LOWBC=0 (i.e. No reflection)'

    def write_hdf5(self,runname):
        """
        Write the surface properties into an HDF5 file
        """

        import h5py

        #Assessing that all the parameters have the correct type and dimension
        self.assess()

        f = h5py.File(runname+'.h5','a')
        #Checking if Atmosphere already exists
        if ('/Surface' in f)==True:
            del f['Surface']   #Deleting the Atmosphere information that was previously written in the file

        if self.GASGIANT==False:

            grp = f.create_group("Surface")

            #Writing the lower boundary condition
            dset = grp.create_dataset('LOWBC',data=int(self.LOWBC))
            dset.attrs['title'] = "Lower Boundary Condition"
            if self.LOWBC==LowerBoundaryCondition.THERMAL:
                dset.attrs['type'] = 'Isotropic thermal emission (no reflection)'
            elif self.LOWBC==LowerBoundaryCondition.LAMBERTIAN:
                dset.attrs['type'] = 'Isotropic thermal emission and Lambert reflection'
            elif self.LOWBC==LowerBoundaryCondition.HAPKE:
                dset.attrs['type'] = 'Isotropic thermal emission and Hapke reflection'
            elif self.LOWBC==LowerBoundaryCondition.OREN_NAYAR:
                dset.attrs['type'] = 'Isotropic thermal emission and Oren-Nayar reflection'

            #Writing the spectral units
            dset = grp.create_dataset('ISPACE',data=int(self.ISPACE))
            dset.attrs['title'] = "Spectral units"
            if self.ISPACE==WaveUnit.Wavenumber_cm:
                dset.attrs['units'] = 'Wavenumber / cm-1'
            elif self.ISPACE==WaveUnit.Wavelength_um:
                dset.attrs['units'] = 'Wavelength / um'

            #Writing the spectral array
            dset = grp.create_dataset('VEM',data=self.VEM)
            dset.attrs['title'] = "Spectral array"
            if self.ISPACE==WaveUnit.Wavenumber_cm:
                dset.attrs['units'] = 'Wavenumber / cm-1'
            elif self.ISPACE==WaveUnit.Wavelength_um:
                dset.attrs['units'] = 'Wavelength / um'

            #Writing the number of locations
            dset = grp.create_dataset('NLOCATIONS',data=self.NLOCATIONS)
            dset.attrs['title'] = "Number of surface locations"

            #Writing the co-ordinates of the locations
            dset = grp.create_dataset('LATITUDE',data=self.LATITUDE)
            dset.attrs['title'] = "Latitude of the surface locations"
            dset.attrs['units'] = 'degrees'

            dset = grp.create_dataset('LONGITUDE',data=self.LONGITUDE)
            dset.attrs['title'] = "Longitude of the surface locations"
            dset.attrs['units'] = 'degrees'

            #Writing the surface temperature
            dset = grp.create_dataset('TSURF',data=self.TSURF)
            dset.attrs['title'] = "Surface Temperature"
            dset.attrs['units'] = 'K'

            #Writing the surface albedo
            dset = grp.create_dataset('GALB',data=self.GALB)
            dset.attrs['title'] = "Ground albedo"
            if self.GALB<0.0:
                dset.attrs['type'] = 'Surface albedo calculated as (1 - EMISSIVITY)'

            #Writing the emissivity
            dset = grp.create_dataset('EMISSIVITY',data=self.EMISSIVITY)
            dset.attrs['title'] = "Surface emissivity"
            dset.attrs['units'] = ''

            #Writing Hapke parameters if they are required
            if self.LOWBC==LowerBoundaryCondition.HAPKE:

                dset = grp.create_dataset('SGLALB',data=self.SGLALB)
                dset.attrs['title'] = "Single scattering albedo"
                dset.attrs['units'] = ''

                dset = grp.create_dataset('K',data=self.K)
                dset.attrs['title'] = "Porosity coefficient"
                dset.attrs['units'] = ''

                dset = grp.create_dataset('BS0',data=self.BS0)
                dset.attrs['title'] = "Amplitude of the opposition effect"
                dset.attrs['units'] = ''

                dset = grp.create_dataset('hs',data=self.hs)
                dset.attrs['title'] = "Width of the opposition surge"
                dset.attrs['units'] = ''

                dset = grp.create_dataset('BC0',data=self.BC0)
                dset.attrs['title'] = "Amplitude of the coherent backscatter opposition effect"
                dset.attrs['units'] = ''

                dset = grp.create_dataset('hc',data=self.hc)
                dset.attrs['title'] = "Width of the backscatter function"
                dset.attrs['units'] = ''

                dset = grp.create_dataset('ROUGHNESS',data=self.ROUGHNESS)
                dset.attrs['title'] = "Roughness mean slope angle"
                dset.attrs['units'] = 'degrees'

                dset = grp.create_dataset('G1',data=self.G1)
                dset.attrs['title'] = "Asymmetry factor of the first Henyey-Greenstein function defining the phase function"
                dset.attrs['units'] = ''

                dset = grp.create_dataset('G2',data=self.G2)
                dset.attrs['title'] = "Asymmetry factor of the second Henyey-Greenstein function defining the phase function"
                dset.attrs['units'] = ''

                dset = grp.create_dataset('F',data=self.F)
                dset.attrs['title'] = "Parameter defining the relative contribution of G1 and G2 of the double Henyey-Greenstein phase function"
                dset.attrs['units'] = ''

            elif self.LOWBC==LowerBoundaryCondition.OREN_NAYAR:

                dset = grp.create_dataset('ALBEDO',data=self.ALBEDO)
                dset.attrs['title'] = "Surface albedo"
                dset.attrs['units'] = ''
                
                dset = grp.create_dataset('ROUGHNESS',data=self.ROUGHNESS)
                dset.attrs['title'] = "Roughness mean slope angle"
                dset.attrs['units'] = 'degrees'

        f.close()

    def read_hdf5(self,runname):
        """
        Read the surface properties from an HDF5 file
        """

        import h5py

        f = h5py.File(runname+'.h5','r')

        #Checking if Surface exists
        e = "/Surface" in f
        if e==False:
            # If we have "no surface", planet is a gasgiant
            # so the surface is always perfectly isothermal
            # and absorbing.
            self.GASGIANT = True
            self.LOWBC = LowerBoundaryCondition.THERMAL
            self.TSURF = 0.0
            self.GALB = 0.0
        else:
            self.GASGIANT = False
            self.ISPACE = h5py_helper.retrieve_data(f, 'Surface/ISPACE', lambda x:  WaveUnit(np.int32(x)))
            self.LOWBC = h5py_helper.retrieve_data(f, 'Surface/LOWBC', lambda x:  LowerBoundaryCondition(np.int32(x)))
            self.NLOCATIONS = h5py_helper.retrieve_data(f, 'Surface/NLOCATIONS', np.int32)

            self.VEM = h5py_helper.retrieve_data(f, 'Surface/VEM', np.array)
            self.NEM = len(self.VEM)
            if self.NLOCATIONS==1:
                self.TSURF = h5py_helper.retrieve_data(f, 'Surface/TSURF', np.float64)
                self.LATITUDE = h5py_helper.retrieve_data(f, 'Surface/LATITUDE', np.float64)
                self.LONGITUDE = h5py_helper.retrieve_data(f, 'Surface/LONGITUDE', np.float64)
            else:
                self.TSURF = h5py_helper.retrieve_data(f, 'Surface/TSURF', np.array)
                self.LATITUDE = h5py_helper.retrieve_data(f, 'Surface/LATITUDE', np.array)
                self.LONGITUDE = h5py_helper.retrieve_data(f, 'Surface/LONGITUDE', np.array)

            self.EMISSIVITY = h5py_helper.retrieve_data(f, 'Surface/EMISSIVITY', np.array)

            if self.LOWBC==LowerBoundaryCondition.LAMBERTIAN:
                self.GALB = h5py_helper.retrieve_data(f, 'Surface/GALB', np.array)

            if self.LOWBC==LowerBoundaryCondition.HAPKE:
                self.SGLALB = h5py_helper.retrieve_data(f, 'Surface/SGLALB', np.array)
                self.BS0 = h5py_helper.retrieve_data(f, 'Surface/BS0', np.array)
                self.hs = h5py_helper.retrieve_data(f, 'Surface/hs', np.array)
                self.BC0 = h5py_helper.retrieve_data(f, 'Surface/BC0', np.array)
                self.hc = h5py_helper.retrieve_data(f, 'Surface/hc', np.array)
                self.K = h5py_helper.retrieve_data(f, 'Surface/K', np.array)
                self.ROUGHNESS = h5py_helper.retrieve_data(f, 'Surface/ROUGHNESS', np.array)
                self.G1 = h5py_helper.retrieve_data(f, 'Surface/G1', np.array)
                self.G2 = h5py_helper.retrieve_data(f, 'Surface/G2', np.array)
                self.F = h5py_helper.retrieve_data(f, 'Surface/F', np.array)
                
            if self.LOWBC==LowerBoundaryCondition.OREN_NAYAR:
                self.ALBEDO = h5py_helper.retrieve_data(f, 'Surface/ALBEDO', np.array)
                self.ROUGHNESS = h5py_helper.retrieve_data(f, 'Surface/ROUGHNESS', np.array)

        self.assess()


    def edit_EMISSIVITY(self, EMISSIVITY_array):
        """
        Edit the surface emissivity at each of the lat/lon points
        @param EMISSIVITY_array: 3D array
            Array defining the surface emissivity at each of the points
        """
        EMISSIVITY_array = np.array(EMISSIVITY_array)
        if self.NLOCATIONS==1:
            assert len(EMISSIVITY_array) == self.NEM , \
                'EMISSIVITY should have NEM elements'
        else:
            assert EMISSIVITY_array.shape == (self.NEM,self.NLOCATIONS) , \
                'EMISSIVITY should have (NEM,NLOCATIONS) elements'
        self.EMISSIVITY = EMISSIVITY_array 


    def edit_SGLALB(self, array):
        """
        Edit the single scattering albedo at each of the lat/lon points
        @param array: 1D or 2D array
        """
        array = np.array(array)
        if self.NLOCATIONS==1:
            assert len(array) == self.NEM , \
                'SGLALB should have NEM elements'
        else:
            assert array.shape == (self.NEM,self.NLOCATIONS) , \
                'SGLALB should have (NEM,NLOCATIONS) elements'
        self.SGLALB = array 

    def edit_ROUGHNESS(self, array):
        """
        Edit the roughness mean slope angle at each of the lat/lon points
        @param array: 1D or 2D array
        """
        array = np.array(array)
        if self.NLOCATIONS==1:
            assert len(array) == self.NEM , \
                'ROUGHNESS should have NEM elements'
        else:
            assert array.shape == (self.NEM,self.NLOCATIONS) , \
                'ROUGHNESS should have (NEM,NLOCATIONS) elements'
        self.ROUGHNESS = array 

    def edit_BS0(self, array):
        """
        Edit the amplitude of the opposition effect at each of the lat/lon points
        @param array: 1D or 2D array
        """
        array = np.array(array)
        if self.NLOCATIONS==1:
            assert len(array) == self.NEM , \
                'BS0 should have NEM elements'
        else:
            assert array.shape == (self.NEM,self.NLOCATIONS) , \
                'BS0 should have (NEM,NLOCATIONS) elements'
        self.BS0 = array 

    def edit_hs(self, array):
        """
        Edit the width of the opposition effect at each of the lat/lon points
        @param array: 1D or 2D array
        """
        array = np.array(array)
        if self.NLOCATIONS==1:
            assert len(array) == self.NEM , \
                'hs should have NEM elements'
        else:
            assert array.shape == (self.NEM,self.NLOCATIONS) , \
                'hs should have (NEM,NLOCATIONS) elements'
        self.hs = array 

    def edit_BC0(self, array):
        """
        Edit the amplitude of the backscatter opposition effect at each of the lat/lon points
        @param array: 1D or 2D array
        """
        array = np.array(array)
        if self.NLOCATIONS==1:
            assert len(array) == self.NEM , \
                'BC0 should have NEM elements'
        else:
            assert array.shape == (self.NEM,self.NLOCATIONS) , \
                'BC0 should have (NEM,NLOCATIONS) elements'
        self.BC0 = array 

    def edit_hc(self, array):
        """
        Edit the width of the backscatter opposition effect at each of the lat/lon points
        @param array: 1D or 2D array
        """
        array = np.array(array)
        if self.NLOCATIONS==1:
            assert len(array) == self.NEM , \
                'hc should have NEM elements'
        else:
            assert array.shape == (self.NEM,self.NLOCATIONS) , \
                'hc should have (NEM,NLOCATIONS) elements'
        self.hc = array 

    def edit_K(self, array):
        """
        Edit the porosity coefficient at each of the lat/lon points
        @param array: 1D or 2D array
        """
        array = np.array(array)
        if self.NLOCATIONS==1:
            assert len(array) == self.NEM , \
                'K should have NEM elements'
        else:
            assert array.shape == (self.NEM,self.NLOCATIONS) , \
                'K should have (NEM,NLOCATIONS) elements'
        self.K = array 

    def edit_G1(self, array):
        """
        Edit the first assymmetry parameter at each of the lat/lon points
        @param array: 1D or 2D array
        """
        array = np.array(array)
        if self.NLOCATIONS==1:
            assert len(array) == self.NEM , \
                'G1 should have NEM elements'
        else:
            assert array.shape == (self.NEM,self.NLOCATIONS) , \
                'G1 should have (NEM,NLOCATIONS) elements'
        self.G1 = array 

    def edit_G2(self, array):
        """
        Edit the second assymmetry parameter at each of the lat/lon points
        @param array: 1D or 2D array
        """
        array = np.array(array)
        if self.NLOCATIONS==1:
            assert len(array) == self.NEM , \
                'G2 should have NEM elements'
        else:
            assert array.shape == (self.NEM,self.NLOCATIONS) , \
                'G2 should have (NEM,NLOCATIONS) elements'
        self.G2 = array 

    def edit_F(self, array):
        """
        Edit the contribution from each H-G function at each of the lat/lon points
        @param array: 1D or 2D array
        """
        array = np.array(array)
        if self.NLOCATIONS==1:
            assert len(array) == self.NEM , \
                'F should have NEM elements'
        else:
            assert array.shape == (self.NEM,self.NLOCATIONS) , \
                'F should have (NEM,NLOCATIONS) elements'
        self.F = array 

    def calc_phase_angle(self,EMISS_ANG,SOL_ANG,AZI_ANG):
        """
        Calculate the phase angle based on the emission, incident and azimuth angles

        Inputs
        ------
        @param EMISS_ANG: 1D array or scalar
            Emission angle (deg)
        @param SOL_ANG: 1D array or scalar
            Solar zenith or incident angle (deg)
        @param AZI_ANG: 1D array or scalar
            Azimuth angle (deg)

        Outputs
        -------
        @param PHASE_ANG: 1D array or scalar
            Phase angle (deg)
        """

        #First of all let's calculate the scattering phase angle
        mu = np.cos(EMISS_ANG/180.*np.pi)   #Cosine of the reflection angle
        mu0 = np.cos(SOL_ANG/180.*np.pi)    #Coside of the incidence angle

        cg = mu * mu0 + np.sqrt(1. - mu**2.) * np.sqrt(1.-mu0**2.) * np.cos(AZI_ANG/180.*np.pi)
        iin = np.where(cg>1.0)
        cg[iin] = 1.0 
        g = np.arccos(cg)/np.pi*180.   #Scattering phase angle (degrees) (NTHETA)

        return g


    def select_location(self,iLOCATION):
        """
        Subroutine to select only one geometry from the Atmosphere class (and remove all the others)

        Inputs
        ------
        @param iLOCATION: int
            Index of the location to be selected

        Outputs
        -------
        Updated Surface class
        """
        
        if iLOCATION>self.NLOCATIONS-1:
            raise ValueError('error in select_location :: iLOCATION must be between 0 and NLOCATIONS-1',[0,self.NLOCATIONS-1])

        self.NLOCATIONS = 1
        
        self.LATITUDE = self.LATITUDE[iLOCATION]
        self.LONGITUDE = self.LONGITUDE[iLOCATION]
        self.TSURF = self.TSURF[iLOCATION]
        
        self.edit_EMISSIVITY(self.EMISSIVITY[:,iLOCATION])
        
        if self.LOWBC==LowerBoundaryCondition.HAPKE: #Hapke case
            
            self.edit_SGLALB(self.SGLALB[:,iLOCATION])
            self.edit_BS0(self.BS0[:,iLOCATION])
            self.edit_hs(self.hs[:,iLOCATION])
            self.edit_BC0(self.BC0[:,iLOCATION])
            self.edit_hc(self.hc[:,iLOCATION])
            self.edit_K(self.K[:,iLOCATION])
            self.edit_ROUGHNESS(self.ROUGHNESS[:,iLOCATION])
            self.edit_G1(self.G1[:,iLOCATION])
            self.edit_G2(self.G2[:,iLOCATION])
            self.edit_F(self.F[:,iLOCATION])
            
        if self.LOWBC==3: #Oren-Nayar case
            
            self.edit_ALBEDO(self.ALBEDO[:,iLOCATION])
            self.edit_ROUGHNESS(self.ROUGHNESS[:,iLOCATION])
            
        #Checking that everything went well
        self.assess()
            
        


    ##################################################################################################################
    ##################################################################################################################
    #                                                   EMISSIVITY
    ##################################################################################################################
    ##################################################################################################################

    def read_sur(self, runname):
        """
        Read the surface emissivity from the .sur file
        @param runname: str
            Name of the Nemesis run
        """
        
        #Opening file
        f = open(runname+'.sur','r')
        nem = int(np.fromfile(f,sep=' ',count=1,dtype='int'))
    
        vem = np.zeros([nem])
        emissivity = np.zeros([nem])
        for i in range(nem):
            tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
            vem[i] = tmp[0]
            emissivity[i] = tmp[1]

        self.NEM = nem
        self.VEM = vem
        self.EMISSIVITY = emissivity

    ##################################################################################################################

    def write_sur(self, runname):
        """
        Write the surface emissivity into the .sur file
        @param runname: str
            Name of the Nemesis run
        """
        
        #Opening file
        f = open(runname+'.sur','w')
        f.write('%i \n' % (self.NEM))
        for i in range(self.NEM):
            f.write('%7.4e \t %7.4e \n' % (self.VEM[i],self.EMISSIVITY[i]))
        f.close()

    ##################################################################################################################


    def calc_radground(self,ISPACE,WAVE=None):
        """
        Calculate the thermal emission from the surface

        @param ISPACE: int
            Units of wavelength array
            (0) Wavenumber in cm-1 (1) Wavelength in microns

        @param WAVE: 1D array
            Wavelength or wavenumber array
        """

        if WAVE is None:
            WAVE = self.VEM

        bbsurf = planck(ISPACE,WAVE,self.TSURF)

        # bbsurf has shape (self.NEM, self.NLOCATIONS)
        # so self.EMISSIVITY must have that shape as well if it is 2D
        # Therefore, spectral axis is 0th axis.
        f = interp1d(self.VEM,self.EMISSIVITY, axis=0)
        emissivity = f(WAVE)

        radground = bbsurf * emissivity

        return radground


    ##################################################################################################################
    ##################################################################################################################
    #                            BIDIRECTIONAL REFLECTANCE DISTRIBUTION FUNCTION (BRDF)
    ##################################################################################################################
    ##################################################################################################################

    def calc_reflectance(self,WAVE,SOL_ANG,EMISS_ANG,AZI_ANG,E0=None):
        """
        Calculate the reflected radiance from the surface based on the type of surface defined
        
        Inputs
        ------
        @param WAVE(NWAVE): 1D array
            Wavelengths (microns) or Wavenumbers (cm-1)
        @param SOL_ANG(NTHETA): 1D array
            Solar zenith angle (degrees)
        @param EMISS_ANG(NTHETA): 1D array
            Emission angle (degrees)
        @param AZI_ANG(NTHETA): 1D array    
            Azimuth angle (degrees)
            
        Optional inputs
        ---------------
        @param E0(NWAVE): 1D array
            Solar spectral irradiance (W/m2/cm-1) or (W/m2/um)
            
        Outputs
        -------
        @param R(NWAVE,NTHETA): 2D array
            Reflected radiance (W/m2/cm-1/sr or W/m2/um/sr) 
        """
        
        NWAVE = len(WAVE)
        NTHETA = len(EMISS_ANG)
        
        if E0 is None:
            E0 = np.ones(NWAVE)
        else:
            if len(E0) != NWAVE:
                raise ValueError('error in calc_reflectance :: E0 must have the same size as WAVE')
            
        BRDF = self.calc_BRDF(WAVE,SOL_ANG,EMISS_ANG,AZI_ANG)  #Bidirectional reflectance distribution function
        
        R = BRDF * np.cos(SOL_ANG/180.*np.pi)
        R *= E0[:,None]
        
        # Set R to 0 where SOL_ANG > 90
        R[:, SOL_ANG > 90] = 0
        
        return R


    def calc_BRDF(self,WAVE,SOL_ANG,EMISS_ANG,AZI_ANG):
        """
        Calculate the BRDF of the surface based on the type of surface defined
        
        Inputs
        ------
        @param WAVE(NWAVE): 1D array
            Wavelengths (microns) or Wavenumbers (cm-1)
        @param SOL_ANG(NTHETA): 1D array
            Solar zenith angle (degrees)
        @param EMISS_ANG(NTHETA): 1D array
            Emission angle (degrees)
        @param AZI_ANG(NTHETA): 1D array    
            Azimuth angle (degrees)
            
        Outputs
        -------
        @param BRDF(NWAVE,NTHETA): 2D array
            Bidirectional reflectance distribution function (BRDF)
        """
        
        NWAVE = len(WAVE)
        NTHETA = len(EMISS_ANG)
        
        BRDF = np.zeros((NWAVE,NTHETA))
        if self.LOWBC==LowerBoundaryCondition.LAMBERTIAN: #Lambertian reflection
            
            ALBEDO = self.calc_albedo()                 #Calculating albedo based on flags in class
            galbx = np.interp(WAVE,self.VEM,ALBEDO)  #Interpolating albedo to desired spectral array
            
            # Broadcasting to avoid loop
            BRDF[:,:] = (galbx[:, None] / np.pi)  # Shape (NWAVE, 1) broadcasted to (NWAVE, NTHETA)
            
        elif self.LOWBC==LowerBoundaryCondition.HAPKE: #Hapke reflection
            
            #Interpolating Hapke parameters to the desired spectral array
            SGLALB = np.interp(WAVE,self.VEM,self.SGLALB)
            K = np.interp(WAVE,self.VEM,self.K)
            BS0 = np.interp(WAVE,self.VEM,self.BS0)
            hs = np.interp(WAVE,self.VEM,self.hs)
            BC0 = np.interp(WAVE,self.VEM,self.BC0)
            hc = np.interp(WAVE,self.VEM,self.hc)
            ROUGHNESS = np.interp(WAVE,self.VEM,self.ROUGHNESS)
            G1 = np.interp(WAVE,self.VEM,self.G1)
            G2 = np.interp(WAVE,self.VEM,self.G2)
            F = np.interp(WAVE,self.VEM,self.F)
            
            #Calling the fortran module to calculate Hapke's BRDF
            BRDF[:,:] = calc_Hapke_BRDF(SGLALB,K,BS0,hs,BC0,hc,ROUGHNESS,G1,G2,F,\
                                    SOL_ANG,EMISS_ANG,AZI_ANG)
            
        elif self.LOWBC == 3: #Oren & Nayar reflection model
            
            #Interpolating Hapke parameters to the desired spectral array
            ALBEDO = np.interp(WAVE,self.VEM,self.ALBEDO)
            ROUGHNESS = np.interp(WAVE,self.VEM,self.ROUGHNESS)
            
            #Calling the fortran module to calculate Oren-Nayar's BRDF
            BRDF[:,:] = calc_OrenNayar_BRDF(ALBEDO,ROUGHNESS,SOL_ANG,EMISS_ANG,AZI_ANG)

        return BRDF


    ##################################################################################################################

    def calc_albedo(self):
        """
        Calculate the Lambert albedo of the surface based on the value on the class

        If GALB<0.0 then the Lambert albedo is calculated from the surface emissivity
        """

        if self.GALB>=0.0:
            ALBEDO = np.ones(self.NEM)*self.GALB
        else:
            ALBEDO = np.zeros(self.NEM)
            ALBEDO[:] = 1.0 - self.EMISSIVITY[:]

        return ALBEDO


    ##################################################################################################################
    ##################################################################################################################
    #                                         HAPKE BIDIRECTIONAL-REFLECTANCE
    ##################################################################################################################
    ##################################################################################################################    


    def read_hap(self, runname):
        """
        Read the Hapke parameters of the surface from the .hap file
        @param runname: str
            Name of the Nemesis run
        """
        
        #Opening file
        f = open(runname+'.hap','r')

        #Reading number of wavelengths
        nem = int(np.fromfile(f,sep=' ',count=1,dtype='int'))
    
        #Defining all fields
        vem = np.zeros(nem)
        sglalb = np.zeros(nem)
        k = np.zeros(nem)
        bso = np.zeros(nem)
        hs = np.zeros(nem)
        bco = np.zeros(nem)
        hc = np.zeros(nem)
        roughness = np.zeros(nem)
        g1 = np.zeros(nem)
        g2 = np.zeros(nem)
        fhg = np.zeros(nem)

        #Reading Hapke parameters
        for i in range(nem):
            tmp = np.fromfile(f,sep=' ',count=11,dtype='float')
            vem[i] = tmp[0]
            sglalb[i] = tmp[1]
            k[i] = tmp[2]
            bso[i] = tmp[3]
            hs[i] = tmp[4]
            bco[i] = tmp[5]
            hc[i] = tmp[6]
            roughness[i] = tmp[7]
            g1[i] = tmp[8]
            g2[i] = tmp[9]
            fhg[i] = tmp[10]

        f.close()

        #Storing parameters in the class
        self.NEM = nem
        self.VEM = vem
        self.SGLALB = sglalb
        self.K = k
        self.BS0 = bso
        self.hs = hs
        self.BC0 = bco
        self.hc = hc
        self.ROUGHNESS = roughness
        self.G1 = g1
        self.G2 = g2
        self.F = fhg

    ##################################################################################################################

    def write_hap(self,runname):
        """
        Read the Hapke parameters stored in the class into the .hap file
        @param runname: str
            Name of the Nemesis run
        """

        f = open(runname+'.hap','w')
        f.write('%i \n' % (self.NEM))
        for i in range(self.NEM):
            f.write('%7.4e \t %7.4e \t %7.4e \t %7.4e \t %7.4e \t %7.4e \t %7.4e \t %7.4e \t %7.4e \t %7.4e \t %7.4e \n' % \
                (self.VEM[i],self.SGLALB[i],self.K[i],self.BS0[i],self.hs[i],self.BC0[i],self.hc[i],self.ROUGHNESS[i],self.G1[i],self.G2[i],self.F[i]))
        f.close()


    ##################################################################################################################
    ##################################################################################################################
    #                                              PLOTTING FUNCTIONS
    ##################################################################################################################
    ################################################################################################################## 

    def plot_tsurf_map(self,subobs_lat=None,subobs_lon=None,cmap='viridis'):
        """
        Function to plot the surface temperature on a map 
        """

        from mpl_toolkits.basemap import Basemap
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig,ax1 = plt.subplots(1,1,figsize=(4,4))

        #Plotting the geometry
        if((subobs_lat is not None) & (subobs_lon is not None)):
            map = Basemap(projection='ortho', resolution=None,
                lat_0=subobs_lat, lon_0=subobs_lon)
        else:
            map = Basemap(projection='ortho', resolution=None,
                lat_0=np.mean(self.LATITUDE), lon_0=np.mean(self.LONGITUDE))
            
        lats = map.drawparallels(np.linspace(-90, 90, 13))
        lons = map.drawmeridians(np.linspace(-180, 180, 13))

        im = map.scatter(self.LONGITUDE,self.LATITUDE,latlon=True,c=self.TSURF,cmap=cmap)

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("bottom", size="5%", pad=0.15)
        cbar2 = plt.colorbar(im,cax=cax,orientation='horizontal')
        cbar2.set_label('Surface Temperature (K)')

        ax1.grid()
        plt.tight_layout()
        plt.show()
        
    def plot_emissivity_map(self,subobs_lat=None,subobs_lon=None,cmap='viridis',iWAVE=0):
        """
        Function to plot the surface emissivity on a map 
        """

        from mpl_toolkits.basemap import Basemap
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig,ax1 = plt.subplots(1,1,figsize=(4,4))

        #Plotting the geometry
        if((subobs_lat is not None) & (subobs_lon is not None)):
            map = Basemap(projection='ortho', resolution=None,
                lat_0=subobs_lat, lon_0=subobs_lon)
        else:
            map = Basemap(projection='ortho', resolution=None,
                lat_0=np.mean(self.LATITUDE), lon_0=np.mean(self.LONGITUDE))
            
        lats = map.drawparallels(np.linspace(-90, 90, 13))
        lons = map.drawmeridians(np.linspace(-180, 180, 13))

        im = map.scatter(self.LONGITUDE,self.LATITUDE,latlon=True,c=self.EMISSIVITY[iWAVE,:],cmap=cmap)

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("bottom", size="5%", pad=0.15)
        cbar2 = plt.colorbar(im,cax=cax,orientation='horizontal')
        cbar2.set_label('Surface Emissivity')

        ax1.grid()
        plt.tight_layout()
        plt.show()
        
        
    
        
        
################################################################################################################################
################################################################################################################################
#                                              OTHER FUNCTIONS OUTSIDE CLASS
################################################################################################################################
################################################################################################################################

#Conditional JIT compilation
# Define a flag to control JIT compilation
use_jit = True

def conditional_jit(nopython=True):
    def decorator(func):
        if use_jit:
            return jit(nopython=nopython)(func)
        else:
            return func
    return decorator


##################################################################################################################
#Thermal emission
##################################################################################################################

def planck(ispace,wave,temp):

    """
    FUNCTION NAME : planck()

    DESCRIPTION : Function to calculate the blackbody radiation given by the Planck function

    INPUTS : 

        ispace :: Flag indicating the spectral units
                  (0) Wavenumber (cm-1)
                  (1) Wavelength (um)
        wave(nwave) :: Wavelength or wavenumber array
        temp(ntemp) :: Temperature of the blackbody (K)

    OPTIONAL INPUTS:  none

    OUTPUTS : 

	    bb(nwave,ntemp) :: Planck function (W cm-2 sr-1 (cm-1)-1 or W cm-2 sr-1 um-1)
 
    CALLING SEQUENCE:

	    bb = planck(ispace,wave,temp)
 
    MODIFICATION HISTORY : Juan Alday (29/07/2021)

    """

    if(np.isscalar(temp)==True):  #Only one temperature value bb(nwave)
        ntemp = 0
    else:  #Several temperature values bb(nwave,ntemp)
        ntemp = len(temp)
        wave = np.repeat(wave[:, np.newaxis],ntemp,axis=1)

    c1 = 1.1911e-12
    c2 = 1.439
    if ispace==0:
        y = wave
        a = c1 * (y**3.)
    elif ispace==1:
        y = 1.0e4/wave
        a = c1 * (y**5.) / 1.0e4
    else:
        raise ValueError('error in planck :: ISPACE must be either 0 or 1')

    tmp = c2 * y / temp
    b = np.exp(tmp) - 1
    bb = a/b

    return bb

###############################################################################################
def planckg(ispace,wave,temp):
    """
    FUNCTION NAME : planckg()

    DESCRIPTION : Function to calculate the blackbody radiation given by the Planck function
                    as well as its derivative with respect to temperature

    INPUTS : 

        ispace :: Flag indicating the spectral units
                  (0) Wavenumber (cm-1)
                  (1) Wavelength (um)
        wave(nwave) :: Wavelength or wavenumber array
        temp :: Temperature of the blackbody (K)

    OPTIONAL INPUTS:  none

    OUTPUTS : 

	    bb(nwave) :: Planck function (W cm-2 sr-1 (cm-1)-1 or W cm-2 sr-1 um-1)
        dBdT(nwave) :: Temperature gradient (W cm-2 sr-1 (cm-1)-1 or W cm-2 sr-1 um-1)/K
 
    CALLING SEQUENCE:

	    bb,dBdT = planckg(ispace,wave,temp)
 
    MODIFICATION HISTORY : Juan Alday (29/07/2021)

    """

    c1 = 1.1911e-12
    c2 = 1.439
    if ispace==0:
        y = wave
        a = c1 * (y**3.)
        ap = c1 * c2 * (y**4.)/temp**2.
    elif ispace==1:
        y = 1.0e4/wave
        a = c1 * (y**5.) / 1.0e4
        ap = c1 * c2 * (y**6.) / 1.0e4 / temp**2.
    else:
        raise ValueError('error in planck :: ISPACE must be either 0 or 1')

    tmp = c2 * y / temp
    b = np.exp(tmp) - 1
    bb = a/b

    tmpp = c2 * y / temp
    bp = (np.exp(tmp) - 1.)**2.
    tp = np.exp(tmpp) * ap
    dBdT = tp/bp

    return bb,dBdT

##################################################################################################################
#Hapke BDRF calculations
##################################################################################################################
 
#@njit(fastmath=True)
@conditional_jit(nopython=True)
def calc_Hapke_BRDF(w,K,BS0,hs,BC0,hc,ROUGHNESS,G1,G2,F,i,e,phi):
    """
    Calculate the bidirectional-reflectance distribution function for a Hapke surface.
    The method used here is described in Hapke (2012): Theory of Reflectance and Emittance
    Spectroscopy, in chapter 12.3.1 (disk-resolved photometry)
    
    Inputs
    ______

    w(nwave) :: Single scattering albedo
    K(nwave) :: Porosity coefficient
    BS0(nwave), hs(nwave) :: Amplitude of opposition effect and width of the opposition surge
    BC0(nwave), hc(nwave) :: Amplitude of the coherent backscatter opposition effect and width of the backscatter function
    ROUGHNESS(nwave) :: Roughness mean slope angle
    G1(nwave),G2(nwave),F(nwave) :: Parameters describing the double Henyey-Greenstein phase function
    i(ntheta),e(ntheta),phi(ntheta) :: Incident, reflection and azimuth angle (degrees)
    
    Outputs
    _______
    
    BRDF(nwave,ntheta) :: Bidirectional reflectance
    
    """
    
    nwave = len(w)
    ntheta = len(i)
    
    BRDF = np.zeros((nwave,ntheta))
    
    for iwave in range(nwave):
        for itheta in range(ntheta):
            
            #print(i[itheta],e[itheta],phi[itheta])
            BRDF[iwave,itheta] = calc_Hapke_BRDFx(w[iwave],K[iwave],BS0[iwave],hs[iwave],BC0[iwave],hc[iwave],ROUGHNESS[iwave],\
                                                  G1[iwave],G2[iwave],F[iwave],i[itheta],e[itheta],phi[itheta])

    return BRDF

##################################################################################################################
    
#@njit(fastmath=True)
@conditional_jit(nopython=True)
def calc_Hapke_BRDFx(w,K,BS0,hs,BC0,hc,ROUGHNESS,G1,G2,F,i,e,phi_nemesis):
    """
    Calculate the bidirectional-reflectance distribution function for a Hapke surface.
    The method used here is described in Hapke (2012): Theory of Reflectance and Emittance
    Spectroscopy, in chapter 12.3.1 (disk-resolved photometry)
    
    Note that in the Hapke model the azimuth angle is defined opposite to NEMESIS.
    In NEMESIS forward scattering is phi=0 and backward scattering is phi=180.
    In Hapke forward scattering is phi=180 and backward scattering is phi=0.
    This is why we need to change the azimuth angle in the Hapke model to be consistent with NEMESIS.

    Inputs
    ______

    w :: Single scattering albedo
    K :: Porosity coefficient
    BS0, hs :: Amplitude of opposition effect and width of the opposition surge
    BC0, hc :: Amplitude of the coherent backscatter opposition effect and width of the backscatter function
    ROUGHNESS :: Roughness mean slope angle
    G1,G2,F :: Parameters describing the double Henyey-Greenstein phase function
    i,e,phi :: Incident, reflection and azimuth angle (degrees)
    
    Outputs
    _______
    
    BRDF :: Bidirectional reflectance
    
    """

    phi = 180. - phi_nemesis  #Changing the azimuth angle to be consistent with Hapke model

    if( (e>=90.) or (i>=90.) ):
    
        BRDF = 0.0
    
    else:
        
        #Calculating the cosine of the angl
        mu = np.cos(e/180.*np.pi)
        mu0 = np.cos(i/180.*np.pi)
        
        #Correcting the azimuth angle to be within 0-180 degrees
        if phi>180.:
            phix = 180. - (phi-180.)
        elif phi<0.:
            phix = -phi
        else:
            phix = phi
            
        #Calculating the phase angle
        cg = mu * mu0 + np.sqrt(1. - mu**2.) * np.sqrt(1. - mu0**2.) * np.cos(phix/180.*np.pi) 
        if cg>1.0:
            cg = 1.0
        if cg<0.0:
            cg = 0.0
        g = np.arccos(cg)/np.pi*180.   #Phase angle (degrees) (NTHETA)
        
        #Calculate some of the input parameters for the Hapke formalism
        gamma = np.sqrt(1. - w)
        r0 = (1. - gamma)/(1. + gamma)
        theta_bar = ROUGHNESS * (1. - r0)
        chi = 1./np.sqrt(1. + np.pi * np.tan(theta_bar/180.*np.pi)**2.)
        if np.abs(phix)==180.:
            fphi = 0.0
        else:
            fphi = np.exp(-2.*np.abs(np.tan(phix/2./180.*np.pi)))  #f(phi)

        #Calculating the E-functions
        E1e = calc_Hapke_E1(e,theta_bar)
        E2e = calc_Hapke_E2(e,theta_bar)
        E1i = calc_Hapke_E1(i,theta_bar)
        E2i = calc_Hapke_E2(i,theta_bar)

        #Calculating the nu functions
        nue = calc_Hapke_nu(e,theta_bar,E1e,E2e,chi)
        nui = calc_Hapke_nu(i,theta_bar,E1i,E2i,chi)
        
        #Calculating the effective incidence and reflection angles
        mu0eff, mueff = calc_Hapke_eff_angles(i,e,phix,theta_bar,E1e,E1i,E2e,E2i,chi)

        #Calculating the shadowing function S
        if i<=e:
            S = mueff/nue * mu0/nui * chi / (1.0 - fphi + fphi*chi*mu0/nui)
        else:
            S = mueff/nue * mu0/nui * chi / (1.0 - fphi + fphi*chi*mu/nue)

        #Calculating the shadow-hiding opposition function Bs
        Bs = BS0 / ( 1. + (1./hs) * np.tan( g/2./180.*np.pi) )
 
        #Calculating the backscatter anfular function Bc
        Bc = BC0 / ( 1. + (1.3 + K) * ( (1./hc*np.tan( g/2./180.*np.pi)) + (1./hc*np.tan( g/2./180.*np.pi))**2.0 ) )

        #Calculating the Ambartsumian–Chandrasekhar H function
        H0e = calc_Hapke_H(w,mu0eff/K,r0)
        He = calc_Hapke_H(w,mueff/K,r0)
         
        #Calculate phase function (double Henyey-Greenstein function)
        phase = calc_Hapke_hgphase(g,G1,G2,F)

        #Calculating the bidirectional reflectance
        r = K * w / (4.*np.pi) * mu0eff / (mu0eff + mueff) * ( phase*(1.+Bs) + (H0e*He-1.) ) * (1.+Bc) * S
        
        #Dividing the BRDF by the cosine of the incidence angle to account for the relation r(i,e,phi) = BRDF(i,e,phi) * cos(i)
        BRDF = r / np.cos(i/180.*np.pi)

    return BRDF


##################################################################################################################
    
#@njit(fastmath=True)
@conditional_jit(nopython=True)
def calc_Hapke_H(SGLALB,x,r0):
    """
    Calculate the Ambartsumian–Chandrasekhar H function of the Hapke formalism (Hapke, 2012; p. 333)

    Inputs
    ------

    SGLALB :: Single scattering albedo
    x :: Value at which the H-function must be evaluated
    r0 :: r0 parameter from the Hapke formalism

    Outputs
    -------

    @param H: 1D array or real scalar
        H function

    """

    H = 1.0 / ( 1.0 - SGLALB*x * (r0 + (1.0 - 2.0*r0*x)/2.0*np.log((1.0+x)/x)) )

    return H

##################################################################################################################

#@njit(fastmath=True)
@conditional_jit(nopython=True)
def calc_Hapke_thetabar(ROUGHNESS,r0):
    """
    Calculate the theta_bar parameter of the Hapke formalism (Hapke, 2012; p. 333)
    This parameter is the corrected roughness mean slope angle

    Inputs
    ------

    ROUGHNESS :: Roughness mean slope angle (degrees)
    r0 :: Diffusive reflectance

    Outputs
    -------

    theta_bar :: Corrected Roughness mean slope angle (degrees)

    """

    theta_bar = ROUGHNESS * (1.0 - r0)

    return theta_bar

##################################################################################################################

#@njit(fastmath=True)
@conditional_jit(nopython=True)
def calc_Hapke_gamma(SGLALB):
    """
    Calculate the gamma parameter of the Hapke formalism (Hapke, 2012; p. 333)
    This parameter is just a factor calculated from the albedo

    Inputs
    ------

    SGLALB :: Single scattering albedo

    Outputs
    -------

    gamma :: Gamma factor

    """

    gamma = np.sqrt(1.0 - SGLALB)

    return gamma

##################################################################################################################

#@njit(fastmath=True)
@conditional_jit(nopython=True)
def calc_Hapke_r0(gamma):
    """
    Calculate the r0 parameter of the Hapke formalism (Hapke, 2012; p. 333)
    This parameter is called the diffusive reflectance

    Inputs
    ------

    gamma :: Gamma factor

    Outputs
    -------

    r0 :: Diffusive reflectance

    """

    r0 = (1.0 - gamma)/(1.0 + gamma)

    return r0

##################################################################################################################


#@njit(fastmath=True)
@conditional_jit(nopython=True)
def calc_Hapke_eff_angles(i,e,phi,theta_bar,E1e,E1i,E2e,E2i,chi):
    """
    Calculate the effective incidence and reflection angles 

    Inputs
    ------

    i :: Incidence angle (degrees)  
    e :: Reflection angle (degrees)
    phi :: Azimuth angle (degrees)
    theta_bar :: Corrected roughness mean slope angle (degrees)
    E1e,E1i,E2e,E2i,chi :: Several different coefficients from the Hapke formalism         

    Outputs
    -------

    mu0_eff :: Cosine of the effective incidence angle
    mu_eff :: Cosine of the effective reflection angle

    """

    #Calculating some initial parameters
    irad = i / 180. * np.pi  
    erad = e / 180. * np.pi
    phirad = phi / 180. * np.pi 
    tbarrad = theta_bar / 180. * np.pi

    #There are two possible cases
    if i<=e:

        mu0eff = chi * ( np.cos(irad) + np.sin(irad) * np.tan(tbarrad) * \
            (np.cos(phirad) * E2e + np.sin(phirad/2.)**2. *E2i) / (2.0 - E1e - phirad/np.pi*E1i)  )

        mueff = chi * ( np.cos(erad) + np.sin(erad) * np.tan(tbarrad) * \
            (E2e - np.sin(phirad/2.)**2. *E2i) / (2.0 - E1e - phirad/np.pi*E1i)  )

    elif i>e:

        mu0eff = chi * ( np.cos(irad) + np.sin(irad) * np.tan(tbarrad) * \
            (E2i - np.sin(phirad/2.)**2. *E2e) / (2.0 - E1i - phirad/np.pi*E1e)  )

        mueff = chi * ( np.cos(erad) + np.sin(erad) * np.tan(tbarrad) * \
            (np.cos(phirad) * E2i + np.sin(phirad/2.)**2. *E2e) / (2.0 - E1i - phirad/np.pi*E1e)  )


    return mu0eff, mueff

##################################################################################################################

#@njit(fastmath=True)
@conditional_jit(nopython=True)
def calc_Hapke_nu(x,theta_bar,E1x,E2x,chi):
    """
    Calculate the nu function from the Hapke formalism (Hapke 2012 p.333) 

    Inputs
    ------

    x :: Incidence or reflection angles (degrees)
    theta_bar :: Corrected roughness mean slope angle (degrees)
    E1x,E1x,chi :: Several different coefficients from the Hapke formalism (evaluated at the angle x)       

    Outputs
    -------

    nu :: Nu parameter from the Hapke formalism

    """

    #Calculating some initial parameters
    xrad = x / 180. * np.pi
    tbarrad = theta_bar / 180. * np.pi

    nu = chi * ( np.cos(xrad) + np.sin(xrad) * np.tan(tbarrad) * \
        (E2x) / (2.0 - E1x)  )

    return nu


##################################################################################################################

#@njit(fastmath=True)
@conditional_jit(nopython=True)
def calc_Hapke_E1(x,theta_bar):
    """
    Calculate the E1 function of the Hapke formalism (Hapke, 2012; p. 333)

    Inputs
    ------

    x :: Angle (degrees)
    theta_bar :: Mean slope angle (degrees)

    Outputs
    -------

    E1 :: Parameter E1 in the Hapke formalism

    """

    if( (theta_bar==0.0) or (x==0.0) ):
        E1 = 0.0
    else:
        E1 = np.exp(-2.0/np.pi * 1.0/np.tan(theta_bar/180.*np.pi) * 1./np.tan(x/180.*np.pi))

    return E1


##################################################################################################################

#@njit(fastmath=True)
@conditional_jit(nopython=True)
def calc_Hapke_E2(x,theta_bar):
    """
    Calculate the E2 function of the Hapke formalism (Hapke, 2012; p. 333)

    Inputs
    ------

    x :: Angle (degrees)
    theta_bar :: Mean slope angle (degrees)

    Outputs
    -------

    E2 :: Parameter E2 in the Hapke formalism

    """

    if( (theta_bar==0.0) or (x==0.0) ):
        E2 = 0.0
    else:
        E2 = np.exp(-1.0/np.pi * 1.0/np.tan(theta_bar/180.*np.pi)**2. * 1./np.tan(x/180.*np.pi)**2.)

    return E2

##################################################################################################################

#@njit(fastmath=True)
@conditional_jit(nopython=True)
def calc_Hapke_chi(theta_bar):
    """
    Calculate the chi function of the Hapke formalism (Hapke, 2012; p. 333)

    Inputs
    ------

    theta_bar :: Corrected roughness mean slope angle (degrees)

    Outputs
    -------

    chi :: Parameter chi in the Hapke formalism

    """

    chi = 1./np.sqrt(1.0 + np.pi * np.tan(theta_bar/180.*np.pi)**2.)

    return chi


##################################################################################################################

#@njit(fastmath=True)
@conditional_jit(nopython=True)
def calc_Hapke_hgphase(Theta,G1,G2,F):
    """
    Calculate the phase function at Theta angles given the double Henyey-Greenstein parameters
    
    Inputs
    ------
    
    Theta :: Scattering angle (degrees)
    
    Outputs
    -------

    phase :: Phase function evaluated at Theta
    """

    t1 = (1.-G1**2.)/(1. - 2.*G1*np.cos(Theta/180.*np.pi) + G1**2.)**1.5
    t2 = (1.-G2**2.)/(1. - 2.*G2*np.cos(Theta/180.*np.pi) + G2**2.)**1.5
    
    phase = F * t1 + (1.0 - F) * t2

    return phase


##################################################################################################################
#Oren & Nayar BDRF calculations
##################################################################################################################

@jit(nopython=True)
def calc_OrenNayar_BRDF(A,ROUGHNESS,i,e,phi):
    """
    Calculate the bidirectional-reflectance distribution function for a surface following the
    method described by Oren & Nayar (1994). This method is a generalisation of the Lambertian model
    for rough surfaces.

    Inputs
    ______

    A(nwave) :: Lambertian albedo
    ROUGHNESS(nwave) :: Roughness parameter (degrees)
    i(ntheta),e(ntheta),phi(ntheta) :: Incident, reflection and azimuth angle (degrees)
    
    Outputs
    _______
    
    BRDF(nwave,ntheta) :: Bidirectional reflectance
    
    """
    
    nwave = len(A)
    ntheta = len(i)
    
    BRDF = np.zeros((nwave,ntheta))
    
    for iwave in range(nwave):
        for itheta in range(ntheta):
            
            BRDF[iwave,itheta] = calc_OrenNayar_BRDFx(A[iwave],ROUGHNESS[iwave],i[itheta],e[itheta],phi[itheta])
            
    return BRDF


@jit(nopython=True)
def calc_OrenNayar_BRDFx(A,ROUGHNESS,i,e,phi):
    """
    Calculate the bidirectional-reflectance distribution function for a surface following the
    method described by Oren & Nayar (1994). This method is a generalisation of the Lambertian model
    for surface with roughness

    Inputs
    ______

    A :: Lambertian albedo
    ROUGHNESS :: Roughness parameter (degrees)
    i,e,phi :: Incident, reflection and azimuth angle (degrees)
    
    Outputs
    _______
    
    BRDF :: Bidirectional reflectance
    
    """
    
    #Converting angles and roughness to radians
    irad = i / 180. * np.pi
    erad = e / 180. * np.pi
    phirad = phi / 180. * np.pi
    sigma = ROUGHNESS / 180. * np.pi
    
    #Calculating initial parameters
    alpha = max([irad,erad])
    beta = min([irad,erad])
    
    #Calculating the C parameters
    C1 = 1.0 - 0.5 * (sigma)**2. / ( (sigma)**2. + 0.33 )
    
    C2 = 0.45 * (sigma)**2. / ( (sigma)**2. + 0.09 )
    if np.cos(phirad)>=0:
        C2 *= np.sin(alpha)
    else:
        C2 *= (np.sin(alpha) - (2. * beta/ np.pi)**3. )
    
    C3 = 0.125 * sigma**2. / ( sigma**2. + 0.09 ) * (4.*alpha*beta/np.pi**2.)**2.
    
    #Calculating the L terms
    BRDF1 = A/np.pi * ( C1 + np.cos(phirad) * C2 * np.tan(beta) + (1. - np.abs(np.cos(phirad))) * C3 * np.tan((alpha+beta)/2.) )
    BRDF2 = 0.17 * A**2. /np.pi * sigma**2. /  ( sigma**2. + 0.13 ) * (1.0 - np.cos(phirad) * (2.*beta/np.pi)**2. )

    BRDF = BRDF1 + BRDF2
    
    return BRDF