#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from __future__ import annotations #  for 3.9 compatability
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

from archnemesis.helpers import h5py_helper
from archnemesis.enums import (
    LayerType,
    LayerIntegrationScheme,
    InterpolationMethod,
)

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

AVOGAD = 6.02214076e23
"""
Object to store layering scheme settings and averaged properties of each layer.
"""
class Layer_0:
    def __init__(self, RADIUS=None, LAYHT=0, LAYTYP=LayerType.EQUAL_LOG_PRESSURE, LAYINT=LayerIntegrationScheme.ABSORBER_WEIGHTED_AVERAGE, NLAY=20, NINT=101,
                LAYANG=0.0, H_base=None, P_base=None):
        """
        After creating a Layer object, call the method
            integrate(self, H, P, T, LAYANG, ID, VMR) to calculate
            averaged layer properties.

        Inputs
        ------

        @param RADIUS: real
            Reference planetary radius where H=0.  Usually at surface for
            terrestrial planets, or at 1 bar pressure level for gas giants.
        @param LAYHT: real
            Height of the base of the lowest layer. Default 0.0.
        @param LAYTYP: LayerType (enum)
            Integer specifying how to split up the layers. Default LayerType.EQUAL_LOG_PRESSURE.
            LayerType.EQUAL_PRESSURE = by equal changes in pressure
            LayerType.EQUAL_LOG_PRESSURE = by equal changes in log pressure
            LayerType.EQUAL_HEIGHT = by equal changes in height
            LayerType.EQUAL_PATH_LENGTH = by equal changes in path length at LAYANG
            LayerType.BASE_PRESSURE = layer base pressure levels specified by P_base
            LayerType.BASE_HEIGHT = layer base height levels specified by H_base
            Note BASE_PRESSURE and BASE_HEIGHT force NLAY = len(P_base) or len(H_base).
        @param LAYINT: LayerIntegrationScheme (enum)
            Layer integration scheme
            LayerIntegrationScheme.MID_PATH = use properties at mid path
            LayerIntegrationScheme.ABSORBER_WEIGHTED_AVERAGE = use absorber amount weighted average values
        @param LAYANG: real
            Angle from zenith at which to split the layers (Default is 0.0)
        @param NLAY: int
            Number of layers to split the atmosphere into. Default 20.
        @param NINT: int
            Number of integration points to be used if LAYINT=LayerIntegrationScheme.ABSORBER_WEIGHTED_AVERAGE.
        @param H_base: 1D array
            Heights of the layer bases defined by user. Default None.
        @param P_base: 1D array
            Pressures of the layer bases defined by user. Default None.


        Attributes
        ------
    
        @param BASEH: 1D array (NLAY)
            Base altitude of each of the layers in the atmosphere (m)
        @param BASEP: 1D array (NLAY)
            Base pressure of each of the layers in the atmosphere (Pa)
        @param DELH: 1D array (NLAY)
            Width of each of the layers in the atmosphere (m)
        @param HEIGHT: 1D array (NLAY)
            Effective altitude each of the layers in the atmosphere (m)
        @param PRESS: 1D array (NLAY)
            Effective pressure of each layer (Pa) (the one at which the radiative transfer calculations are performed)
        @param TEMP: 1D array (NLAY)
            Effective temperature of each layer (Pa) (the one at which the radiative transfer calculations are performed)
        @param TOTAM: 1D array (NLAY)
            Gaseous line-of-sight column density in each layer (particles/m2)
        @param AMOUNT: 2D array (NLAY,NVMR)
            Gaseous line-of-sight column density of each gas in each layer (particles/m2)
        @param PP: 2D array (NLAY,NVMR)
            Partial pressure of each gas in each layer (Pa)
        @param CONT: 2D array (NLAY,NDUST)
            Gaseous line-of-sight column density of each dust population in each layer (particles/m2)

        @param DTE: 2D array (NLAY,NP)
            Matrix relating the temperatures in each layer wrt those in the input profiles
        @param DTE: 2D array (NLAY,NP)
            Matrix relating the gaseous abundances in each layer wrt those in the input profiles
        @param DCO: 2D array (NLAY,NP)
            Matrix relating the dust abundances in each layer wrt those in the input profiles
        @param DPH: 2D array (NLAY,NP)
            Matrix relating the para-H2 fractions in each layer wrt those in the input profiles

        @param TAURAY: 2D array (NWAVE,NLAY)
            Rayleigh scattering optical depth
        @param TAUSCAT: 2D array (NWAVE,NLAY)
            Aerosol scattering optical depth
        @param TAUCLSCAT: 3D array (NWAVE,NLAY,NDUST)
            Aerosol scattering optical depth by each aerosol type
        @param TAUDUST: 2D array (NWAVE,NLAY)
            Aerosol extinction optical depth (absorption + scattering)
        @param TAUCIA: 2D array (NWAVE,NLAY)
            CIA optical depth
        @param TAUGAS: 3D array (NWAVE,NG,NLAY)
            Gas optical depth
        @param TAUTOT: 3D array (NWAVE,NG,NLAY)
            Total optical depth


        Methods
        -------

        Layer_0.assess()
        Layer_0.write_hdf5()
        Layer_0.read_hdf5()
        Layer_0.integrate()
        Layer_0.integrateg()
        Layer_0.layer_split()
        Layer_0.layer_average()
        Layer_0.layer_averageg()

        """
        self.RADIUS = RADIUS
        #self.LAYTYP = LAYTYP
        self.NLAY = NLAY
        #self.LAYINT = LAYINT
        self.NINT = NINT
        self.LAYHT = LAYHT
        self.H_base = H_base
        self.P_base = P_base
        self.LAYANG = None     #Angle used to split the layers. Usually it is zero unless it is a limb calculation that it is 90.
        
        #Properties of the layers
        self.BASEH = None      #(NLAY) Base altitude of each layer (m)
        self.BASEP = None      #(NLAY) Base pressure of each layer (Pa)
        self.DELH = None       #(NLAY) Width of each layer (m)
        self.HEIGHT = None     #(NLAY) Effective height of each layer (m) 
        self.PRESS = None      #(NLAY) Effective pressure of each layer (Pa) (the one at which the radiative transfer calculations are performed)
        self.TEMP = None       #(NLAY) Effective temperature of each layer (K) (the one at which the radiative transfer calculations are performed)
        self.TOTAM = None      #(NLAY) Gaseous line-of-sight column density in each layer (m-2)
        self.AMOUNT = None     #(NLAY,NVMR) Gaseous line-of-sight column density of each gas in each layer (m-2)
        self.PP = None         #(NLAY,NVMR) Effective partial pressure of each gas in each layer (Pa)
        self.CONT = None       #(NLAY,NDUST)  Line-of-sight column density of each dust population in each layer (particles/m2)
        self.FRAC = None     #(NLAY) Fraction of para-h2 in each layer
        
        #Matrices relating the properties at the reference input profiles wrt the properties of each layer 
        self.DTE = None        #(NLAY,NP)
        self.DAM = None        #(NLAY,NP)
        self.DCO = None        #(NLAY,NP)
        self.DPH = None        #(NLAY,NP)

        #Optical depths in each layer
        self.NWAVE = None  #Number of calculation wavelengths
        self.TAURAY = None  #(NWAVE,NLAY) Rayleigh scattering optical depth
        self.TAUSCAT = None #(NWAVE,NLAY) Aerosol scattering optical depth
        self.TAUDUST = None #(NWAVE,NLAY) Aerosol absorption + scattering optical depth
        self.TAUCIA = None  #(NWAVE,NLAY) CIA optical depth
        self.TAUGAS = None  #(NWAVE,NG,NLAY) Gas optical depth
        self.TAUTOT = None  #(NWAVE,NG,NLAY) Total optical depth

        #Units of the dust
        self.DUST_UNITS_FLAG = None   #If None, dust units are expected to be particles m-3. If -1 they are expected to be particles per gram of atmosphere

        # private attributes
        self._laytyp = None
        self._layint = None
        
        # set properties
        self.LAYTYP = LAYTYP
        self.LAYINT = LAYINT
    
    @property
    def LAYTYP(self) -> LayerType:
        return self._laytyp
    
    @LAYTYP.setter
    def LAYTYP(self, value):
        self._laytyp = LayerType(value)
    
    @property
    def LAYINT(self) -> LayerIntegrationScheme:
        return self._layint
    
    @LAYINT.setter
    def LAYINT(self, value):
        self._layint = LayerIntegrationScheme(value)

    ####################################################################################################

    def assess(self):
        """
        Assess whether the different variables have the correct dimensions and types
        """

        #Checking some common parameters to all cases
        assert np.issubdtype(type(self.NLAY), np.integer) == True , \
            'NLAY must be int'
        assert self.NLAY > 0 , \
            'NLAY must be >0'
        
        #Checking some common parameters to all cases
        assert isinstance(self.LAYTYP, (int, LayerType)), \
            'LAYTYP must be int or LayerType'
        assert self.LAYTYP >= LayerType.EQUAL_PRESSURE , \
            'LAYTYP must be >= LayerType.EQUAL_PRESSURE'
        assert self.LAYTYP <= LayerType.BASE_HEIGHT , \
            'LAYTYP must be <= LayerType.BASE_HEIGHT'
        
        #Checking some common parameters to all cases
        assert isinstance(self.LAYINT, (int, LayerIntegrationScheme)), \
            'LAYINT must be int or LayerIntegrationScheme'
        assert self.LAYINT >= LayerIntegrationScheme.MID_PATH , \
            'LAYINT must be >= LayerIntegrationScheme.MID_PATH'
        assert self.LAYINT <= LayerIntegrationScheme.ABSORBER_WEIGHTED_AVERAGE , \
            'LAYINT must be <= LayerIntegrationScheme.ABSORBER_WEIGHTED_AVERAGE'
        
        if self.LAYTYP==LayerType.BASE_PRESSURE:
            assert len(self.P_base) == self.NLAY , \
                'P_base must have size (NLAY) if LAYTYP=LayerType.BASE_PRESSURE'
        
        if self.LAYTYP==LayerType.BASE_HEIGHT:
            assert len(self.H_base) == self.NLAY , \
                'H_base must have size (NLAY) if LAYTYP=LayerType.BASE_HEIGHT'


    ####################################################################################################
    
    def summary_info(self):
        """
        Print summary information about the class in the screen
        """
        
        _lgr.info(f'Number of layers ::  {(self.NLAY)}')

        if self.LAYTYP==LayerType.EQUAL_PRESSURE:
            _lgr.info('Layers calculated by equal changes in pressure')
        elif self.LAYTYP==LayerType.EQUAL_LOG_PRESSURE:
            _lgr.info('Layers calculated by equal changes in log pressure')
        elif self.LAYTYP==LayerType.EQUAL_HEIGHT:
            _lgr.info('Layers calculated by equal changes in altitude')
        elif self.LAYTYP==LayerType.EQUAL_PATH_LENGTH:
            _lgr.info('Layers calculated by equal changes in line-of-sight path intervals')
        elif self.LAYTYP==LayerType.BASE_PRESSURE:
            _lgr.info('Layers calculated by specified base pressures')
        elif self.LAYTYP==LayerType.BASE_HEIGHT:
            _lgr.info('Layers calculated by specified base altitudes')

        if self.LAYINT==LayerIntegrationScheme.MID_PATH:
            _lgr.info('Layer properties calculated at the middle of the layer')
        elif self.LAYINT==LayerIntegrationScheme.ABSORBER_WEIGHTED_AVERAGE:
            _lgr.info('Layer properties calculated through mass weighted averages')
            
        if self.BASEH is not None:
            _lgr.info(f"BASEH(km) {('BASEP(bar)','DELH(km)','P(bar)','T(K)','TOTAM(m-2)','DUST_TOTAM(m-2)')}")
            for i in range(self.NLAY):
                _lgr.info(self.BASEH[i]/1.0e3,self.BASEP[i]/1.0e5,self.DELH[i]/1.0e3,self.PRESS[i]/1.0e5,self.TEMP[i],np.sum(self.AMOUNT[i,:]),np.sum(self.CONT[i,:]))


    ####################################################################################################

    def write_hdf5(self,runname):
        """
        Write the information about the layering of the atmosphere in the HDF5 file
        """

        import h5py

        self.assess()

        #Writing the information into the HDF5 file
        with h5py.File(runname+'.h5', 'a') as f:
            #Checking if Layer already exists
            if ('/Layer' in f)==True:
                del f['Layer']   #Deleting the Layer information that was previously written in the file

            grp = f.create_group("Layer")

            #Writing the layering type
            dset = h5py_helper.store_data(grp, 'LAYTYP', int(self.LAYTYP))
            dset.attrs['title'] = "Layer splitting calculation type"
            if self.LAYTYP==LayerType.EQUAL_PRESSURE:
                dset.attrs['type'] = 'Split layers by equal changes in pressure'
            elif self.LAYTYP==LayerType.EQUAL_LOG_PRESSURE:
                dset.attrs['type'] = 'Split layers by equal changes in log-pressure'
            elif self.LAYTYP==LayerType.EQUAL_HEIGHT:
                dset.attrs['type'] = 'Split layers by equal changes in height'
            elif self.LAYTYP==LayerType.EQUAL_PATH_LENGTH: 
                dset.attrs['type'] = 'Split layers by equal changes in path length (at LAYANG)'
            elif self.LAYTYP==LayerType.BASE_PRESSURE:
                dset.attrs['type'] = 'Split layers by defining base pressure of each layer'
            elif self.LAYTYP==LayerType.BASE_HEIGHT:
                dset.attrs['type'] = 'Split layers by defining base altitude of each layer'

            #Writing the layering integration type
            dset = h5py_helper.store_data(grp, 'LAYINT', int(self.LAYINT))
            dset.attrs['title'] = "Layer integration calculation type"
            if self.LAYINT==LayerIntegrationScheme.MID_PATH:
                dset.attrs['type'] = 'Layer properties calculated at mid-point between layer boundaries'
            if self.LAYINT==LayerIntegrationScheme.ABSORBER_WEIGHTED_AVERAGE:
                dset.attrs['type'] = 'Layer properties calculated by weighting in terms of atmospheric mass (i.e. Curtis-Godson path)'
        
            #Writing the number of layers
            dset = h5py_helper.store_data(grp, 'NLAY', self.NLAY)
            dset.attrs['title'] = "Number of layers"

            #Writing the altitude of bottom layer
            dset = h5py_helper.store_data(grp, 'LAYHT', self.LAYHT)
            dset.attrs['title'] = "Altitude at base of bottom layer"
            dset.attrs['units'] = "m"

            #Writing the base properties of the layers in special cases
            if self.LAYTYP==LayerType.BASE_PRESSURE:
                dset = h5py_helper.store_data(grp, 'P_base', self.P_base)
                dset.attrs['title'] = "User defined pressure at the base of each layer"
                dset.attrs['units'] = "Pressure / Pa"
            
            if self.LAYTYP==LayerType.BASE_HEIGHT:
                dset = h5py_helper.store_data(grp, 'H_base', self.H_base)
                dset.attrs['title'] = "User defined altitude at the base of each layer"
                dset.attrs['units'] = "Altitude / m"
            
            dset = h5py_helper.store_data(grp, 'BASEH', self.BASEH)
            dset.attrs['title'] = "Altitude at the base of each layer"
            dset.attrs['units'] = "Altitude / m"
            
            dset = h5py_helper.store_data(grp, 'BASEP', self.BASEP)
            dset.attrs['title'] = "Altitude at the base of each layer"
            dset.attrs['units'] = "Pressure / Pa"
            
            dset = h5py_helper.store_data(grp, 'TAUTOT', self.TAUTOT)
            dset.attrs['title'] = 'Total opacity of each wavelength at each G-ordinate at each layer. NOTE: G-ordinate is an interval of cumulative fraction of absorption coefficient (see correlated-k approximation)'
            dset.attrs['units'] = 'cm^{-2}'
        
        return
    
    
    def read_hdf5(self,runname):
        """
        Read the Layer properties from an HDF5 file
        """

        import h5py

        with h5py.File(runname+'.h5', 'r') as f:
            #Checking if Surface exists
            e = "/Layer" in f
            if e==False:
                raise ValueError('error :: Layer is not defined in HDF5 file')
            else:

                self.NLAY = h5py_helper.retrieve_data(f, 'Layer/NLAY', np.int32, default=20)
                self.LAYTYP = h5py_helper.retrieve_data(f, 'Layer/LAYTYP', lambda x:  LayerType(np.int32(x)), default=LayerType.EQUAL_LOG_PRESSURE)
                self.LAYINT = h5py_helper.retrieve_data(f, 'Layer/LAYINT', lambda x:  LayerIntegrationScheme(np.int32(x)), default=LayerIntegrationScheme.ABSORBER_WEIGHTED_AVERAGE)
                self.LAYHT = h5py_helper.retrieve_data(f, 'Layer/LAYHT', np.float64)
                if self.LAYTYP==LayerType.BASE_PRESSURE:
                    self.P_base = h5py_helper.retrieve_data(f, 'Layer/P_base', np.array)
                if self.LAYTYP==LayerType.BASE_HEIGHT:
                    self.H_base = h5py_helper.retrieve_data(f, 'Layer/H_base', np.array)
                self.BASEH = h5py_helper.retrieve_data(f, 'Layer/BASEH', np.array)
                self.BASEP = h5py_helper.retrieve_data(f, 'Layer/BASEP', np.array)
                self.TAUTOT = h5py_helper.retrieve_data(f, 'Layer/TAUTOT', np.array)


    ####################################################################################################
 
    def calc_layering(self, H, P, T, ID, VMR, DUST, PARAH2, MOLWT=None):
        """
        Function to split atmosphere into layer and calculate their effective properties.
        
        Inputs
        ______
        
        @param H: 1D array (NP)
            Input profile heights (m)
        @param P: 1D array (NP)
            Input profile pressures (Pa)
        @param T: 1D array (NP)
            Input profile temperatures (K) 
        @param ID: 1D array (NVMR)
            Gas identifiers.
        @param VMR: 2D array (NP,NVMR)
            VMR[i,j] is Volume Mixing Ratio of gas j at vertical point i
            the column j corresponds to the gas with RADTRANS ID ID[j].
        @param DUST: 2D array (NP,NDUST)
            DUST[i,j] is abundance of the dust population j at vertical point i
            the column j corresponds to the dust population.
        @param MOLWT: 1D array (NP)
            Molecular weight at each altitude level in kg mol-1 (only required if DUST_UNITS are in particles per gram of atm)
        """
        
        #Splitting the atmosphere into layers - Calculating the BASEH and BASEP of each layer
        self.layer_split(H=H, P=P, T=T, LAYANG=self.LAYANG)
        
        #Once we have the layers, we calculate the effective properties of each layer
        self.layer_average(ID=ID, VMR=VMR, DUST=DUST, PARAH2=PARAH2, MOLWT=MOLWT)

    ####################################################################################################

    def calc_layeringg(self, H, P, T, ID, VMR, DUST, PARAH2, MOLWT=None):
        """
        
        Function to split atmosphere into layer and calculate their effective properties.
        In addition, we calculate the matrices relating the properties at each of the layers
        with respect to the properties in the input profiles.
        
        Inputs
        ______
        
        @param H: 1D array
            Input profile heights (m)
        @param P: 1D array
            Input profile pressures (Pa)
        @param T: 1D array
            Input profile temperatures (K)
        @param LAYANG: real
            Zenith angle in degrees defined at LAYHT.
        @param ID: 1D array
            Gas identifiers.
        @param VMR: 2D array
            VMR[i,j] is Volume Mixing Ratio of gas j at vertical point i
            the column j corresponds to the gas with RADTRANS ID ID[j].
        @param DUST: 2D array
            DUST[i,j] is abundance of the dust population j at vertical point i
            the column j corresponds to the dust population.
        @param MOLWT: 1D array (NP)
            Molecular weight at each altitude level in kg mol-1 (only required if DUST_UNITS are in particles per gram of atm)
        """
        
        #Splitting the atmosphere into layers - Calculating the BASEH and BASEP of each layer
        self.layer_split(H=H, P=P, T=T, LAYANG=self.LAYANG)
        
        #Once we have the layers, we calculate the effective properties of each layer
        self.layer_averageg(ID=ID, VMR=VMR, DUST=DUST, PARAH2=PARAH2, MOLWT=MOLWT)

    ####################################################################################################

    def layer_split(self, H, P, T, LAYANG):
        """
        Function to split atmosphere into layers and calculate the base altitudes and pressures
        
        Inputs
        ______
        
        @param H: 1D array
            Input profile heights (m)
        @param P: 1D array
            Input profile pressures (Pa)
        @param T: 1D array
            Input profile temperatures (K)
        @param LAYANG: real
            Zenith angle in degrees defined at LAYHT.
        """
        self.H = H
        self.P = P
        self.T = T
        self.LAYANG = LAYANG
        # get layer base height and base pressure
        BASEH, BASEP = layer_split(RADIUS=self.RADIUS, H=H, P=P, LAYANG=LAYANG,
            LAYHT=self.LAYHT, NLAY=self.NLAY, LAYTYP=self.LAYTYP,
            H_base=self.H_base, P_base=self.P_base)
        self.BASEH = BASEH
        self.BASEP = BASEP

    ####################################################################################################

    def layer_average(self, ID, VMR, DUST, PARAH2, MOLWT=None):
        """
        Function to calculate the effective properties of each layer based on the
        layer boundaries, and the reference input profiles
        
        Inputs
        ______
        
        @param ID: 1D array
            Gas identifiers.
        @param VMR: 2D array
            VMR[i,j] is Volume Mixing Ratio of gas j at vertical point i
            the column j corresponds to the gas with RADTRANS ID ID[j].
        @param DUST: 2D array
            DUST[i,j] is abundance of the dust population j at vertical point i
            the column j corresponds to the dust population.
        @param PARAH2: 1D array
            Para-H2 fraction at each altitude level
        @param MOLWT: 1D array
            Molecular weight at each altitude level 
            (only required if DUST_UNITS is in partcles per gram of atm)
        """

        HEIGHT,PRESS,TEMP,TOTAM,AMOUNT,PP,CONT,FRAC,DELH,BASET,LAYSF\
            = layer_average(RADIUS=self.RADIUS, H=self.H, P=self.P, T=self.T,
                ID=ID, VMR=VMR, DUST=DUST, PARAH2=PARAH2, BASEH=self.BASEH, BASEP=self.BASEP,
                LAYANG=self.LAYANG, LAYINT=self.LAYINT, LAYHT=self.LAYHT,
                NINT=self.NINT, DUST_UNITS = self.DUST_UNITS_FLAG, XMOLWT=MOLWT)
        self.HEIGHT = HEIGHT
        self.PRESS = PRESS
        self.TEMP = TEMP
        self.TOTAM = TOTAM
        self.AMOUNT = AMOUNT
        self.PP = PP
        self.CONT = CONT
        self.FRAC = FRAC
        self.DELH = DELH
        self.BASET = BASET
        self.LAYSF = LAYSF

    ####################################################################################################

    def layer_averageg(self, ID, VMR, DUST, PARAH2, MOLWT=None):
        """
        Function to calculate the effective properties of each layer based on the
        layer boundaries, and the reference input profiles. In addition, this function
        calculates the matrices relating the effective properties at each layer
        and the values in the input profiles.
        
        Inputs
        ______
        
        @param ID: 1D array
            Gas identifiers.
        @param VMR: 2D array
            VMR[i,j] is Volume Mixing Ratio of gas j at vertical point i
            the column j corresponds to the gas with RADTRANS ID ID[j].
        @param DUST: 2D array
            DUST[i,j] is abundance of the dust population j at vertical point i
            the column j corresponds to the dust population.
        @param MOLWT: 1D array
            Molecular weight at each altitude level 
            (only required if DUST_UNITS is in partcles per gram of atm)
        """
        # get averaged layer properties
        HEIGHT,PRESS,TEMP,TOTAM,AMOUNT,PP,CONT,FRAC,DELH,BASET,LAYSF,DTE,DAM,DCO,DPH\
            = layer_averageg(RADIUS=self.RADIUS, H=self.H, P=self.P, T=self.T,
                ID=ID, VMR=VMR, DUST=DUST, PARAH2=PARAH2, BASEH=self.BASEH, BASEP=self.BASEP,
                LAYANG=self.LAYANG, LAYINT=self.LAYINT, LAYHT=self.LAYHT,
                NINT=self.NINT, DUST_UNITS = self.DUST_UNITS_FLAG, XMOLWT=MOLWT)
        self.HEIGHT = HEIGHT
        self.PRESS = PRESS
        self.TEMP = TEMP
        self.TOTAM = TOTAM
        self.AMOUNT = AMOUNT
        self.PP = PP
        self.CONT = CONT
        self.FRAC = FRAC
        self.DELH = DELH
        self.BASET = BASET
        self.LAYSF = LAYSF
        self.DTE = DTE
        self.DAM = DAM
        self.DCO = DCO
        self.DPH = DPH
    ####################################################################################################

    def plot_Layer(self):
        """
        Make a summary plot summarising the information about the layers in the atmosphere
        """
        
        #Making a summary plot 
        fig,ax = plt.subplots(1,4,figsize=(10,3),sharey=True)

        #Pressure
        for i in range(self.NLAY):
            ax[0].axhline(self.BASEH[i]/1.0e3,c='black',linestyle='--',linewidth=0.5)
        ax[0].plot(self.PRESS/1.0e5,self.HEIGHT/1.0e3,marker='o',markersize=4.,c='tab:blue')
        ax[0].set_xscale('log')
        ax[0].set_xlabel('Effective pressure (bar)')
        ax[0].set_ylabel('Altitude (km)')
        ax[0].set_facecolor('lightgray')

        #Temperature
        for i in range(self.NLAY):
            ax[1].axhline(self.BASEH[i]/1.0e3,c='black',linestyle='--',linewidth=0.5)
        ax[1].plot(self.TEMP,self.HEIGHT/1.0e3,marker='o',markersize=4.,c='tab:red')
        ax[1].set_xlabel('Effective temperature (K)')
        #ax[1].set_ylabel('Altitude (km)')
        ax[1].set_facecolor('lightgray')

        #Gaseous column density
        for i in range(self.NLAY):
            ax[2].axhline(self.BASEH[i]/1.0e3,c='black',linestyle='--',linewidth=0.5)
        for i in range(self.PP.shape[1]):
            ax[2].plot(self.AMOUNT[:,i],self.HEIGHT/1.0e3,marker='o',markersize=4.)
        ax[2].set_xscale('log')
        ax[2].set_xlabel('Gaseous column density (m$^{-2}$)')
        #ax[2].set_ylabel('Altitude (km)')
        ax[2].set_facecolor('lightgray')

        #Dust column density
        for i in range(self.NLAY):
            ax[3].axhline(self.BASEH[i]/1.0e3,c='black',linestyle='--',linewidth=0.5)
        for i in range(self.CONT.shape[1]):
            ax[3].plot(self.CONT[:,i],self.HEIGHT/1.0e3,marker='o',markersize=4.)
        ax[3].set_xscale('log')
        ax[3].set_xlabel('Aerosol column density (m$^{-2}$)')
        #ax[3].set_ylabel('Altitude (km)')
        ax[3].set_facecolor('lightgray')

        plt.tight_layout()


#########################################################################################
# END OF LAYER CLASS

# USEFUL FUNCTIONS FOR THE LAYERING CLASS
#########################################################################################

def interp(X_data, Y_data, X, ITYPE=InterpolationMethod.LINEAR):
    """
    Routine for 1D interpolation using the SciPy library.

    Inputs
    ------
    @param X_data: 1D array

    @param Y_data: 1D array

    @param X: real

    @param ITYPE: InterpolationMethod (enum)
        InterpolationMethod.LINEAR=linear interpolation
        InterpolationMethod.QUADRATIC_SPLINE=quadratic spline interpolation
        InterpolationMethod.CUBIC_SPLINE=cubic spline interpolation
    """
    
    from scipy.interpolate import interp1d
    
    if ITYPE == InterpolationMethod.LINEAR:
        f = interp1d(X_data, Y_data, kind='linear', fill_value='extrapolate')
        Y = f(X)

    elif ITYPE == InterpolationMethod.QUADRATIC_SPLINE:
        f = interp1d(X_data, Y_data, kind='quadratic', fill_value='extrapolate')
        Y = f(X)

    elif ITYPE == InterpolationMethod.CUBIC_SPLINE:
        f = interp1d(X_data, Y_data, kind='cubic', fill_value='extrapolate')
        Y = f(X)

    return Y

#########################################################################################

@jit(nopython=True)
def interp_numba(X_data, Y_data, X):
    """
    Optimised routine for 1D linear interpolation using Numba JIT.

    Inputs
    ------
    @param X_data: 1D array (monotonic increasing values)
    @param Y_data: 1D array
    @param X: 1D array (values to interpolate)

    Returns
    -------
    @return Y: Interpolated values at X
    @return J: Indices of the left bounding points
    @return F: Fractional distances between interpolation points
    """

    # Ensure X is always an array
    instance=False
    if isinstance(X, (float, np.float64, int)):
        instance=True
        X = np.array([X])  # Convert single value to array
        
    if np.any(np.diff(X_data) <= 0):
        raise ValueError("error in interp :: X_data must be strictly increasing.")
        
    NX = len(X)
    Y = np.zeros(NX)
    J = np.zeros(NX, dtype=np.int32)
    F = np.zeros(NX)

    for IX in range(NX):
        j = 0
        while j < len(X_data) and X_data[j] <= X[IX]:
            j += 1
        j = min(j, len(X_data) - 1)  # Ensure j does not exceed bounds

        if j == 0:
            j = 1  # Avoid out-of-bounds access

        J[IX] = j - 1
        F[IX] = (X[IX] - X_data[j - 1]) / (X_data[j] - X_data[j - 1])
        Y[IX] = (1.0 - F[IX]) * Y_data[j - 1] + F[IX] * Y_data[j]

    if instance==True:
        return Y[0] 
    else:
        return Y

#########################################################################################

@jit(nopython=True)
def interpg(X_data, Y_data, X):
    """
    Optimised routine for 1D linear interpolation using Numba JIT.

    Inputs
    ------
    @param X_data: 1D array (monotonic increasing values)
    @param Y_data: 1D array
    @param X: 1D array (values to interpolate)

    Returns
    -------
    @return Y: Interpolated values at X
    @return J: Indices of the left bounding points
    @return F: Fractional distances between interpolation points
    """

    NX = len(X)
    Y = np.zeros(NX)
    J = np.zeros(NX, dtype=np.int32)
    F = np.zeros(NX)

    for IX in range(NX):
        j = 0
        while j < len(X_data) and X_data[j] <= X[IX]:
            j += 1
        j = min(j, len(X_data) - 1)  # Ensure j does not exceed bounds

        if j == 0:
            j = 1  # Avoid out-of-bounds access

        J[IX] = j - 1
        F[IX] = (X[IX] - X_data[j - 1]) / (X_data[j] - X_data[j - 1])
        Y[IX] = (1.0 - F[IX]) * Y_data[j - 1] + F[IX] * Y_data[j]

    return Y, J, F

#########################################################################################

def layer_average(RADIUS, H, P, T, ID, VMR, DUST, PARAH2, BASEH, BASEP,
                  LAYANG=0.0, LAYINT=LayerIntegrationScheme.MID_PATH, LAYHT=0.0, NINT=101, DUST_UNITS=None, XMOLWT=None):
    """
    Calculates average layer properties.
    Takes an atmosphere profile and a layering shceme specified by
    layer base altitudes and pressures and returns averaged height,
    pressure, temperature, VMR for each layer.

    Inputs
    ------
    @param RADIUS: real
        Reference planetary radius where H=0.  Usually at surface for
        terrestrial planets, or at 1 bar pressure level for gas giants.
    @param H: 1D array
        Input profile heights
    @param P: 1D array
        Input profile pressures
    @param T: 1D array
        Input profile temperatures
    @param ID: 1D array
        Gas identifiers.
    @param VMR: 2D array
        VMR[i,j] is Volume Mixing Ratio of gas j at vertical point i
        the column j corresponds to the gas with RADTRANS ID ID[j].
    @param DUST: 2D array
        DUST[i,j] is dust density of dust popoulation j at vertical point i  (particles/m3)
    @param PARAH2: 1D array
        Para-H2 fraction
    @param BASEH: 1D array
        Heights of the layer bases.
    @param BASEP: 1D array
        Pressures of the layer bases.
    @param LAYANG: real
        Zenith angle in degrees defined at LAYHT.
    @param LAYINT: LayerIntegrationScheme (enum)
        Layer integration scheme
        LayerIntegrationScheme.MID_PATH = use properties at mid path
        LayerIntegrationScheme.ABSORBER_WEIGHTED_AVERAGE = use absorber amount weighted average values
    @param LAYHT: real
        Height of the base of the lowest layer. Default 0.0.
    @param NINT: int
        Number of integration points to be used if LAYINT=LayerIntegrationScheme.ABSORBER_WEIGHTED_AVERAGE.
    @param XMOLWT: 1D array
        Molecular weight vertical profile in kg mol-1 (only required if DUST_UNITS is activated)

    Returns
    -------
    @param HEIGHT: 1D array
        Representative height for each layer
    @param PRESS: 1D array
        Representative pressure for each layer
    @param TEMP: 1D array
        Representative pressure for each layer
    @param TOTAM: 1D array
        Total gaseous absorber amounts along the line-of-sight path, i.e.
        number of molecules per area.
    @param AMOUNT: 1D array
        Representative absorber amounts of each gas at each layer.
        AMOUNT[I,J] is the representative number of gas J molecules
        in layer I in the form of number of molecules per area.
    @param PP: 1D array
        Representative partial pressure for each gas at each layer.
        PP[I,J] is the representative partial pressure of gas J in layer I.
    @param DELH: 1D array
        Layer thicnkness.
    @param BASET: 1D array
        Layer base temperature.
    @param LAYSF: 1D array
        Layer scaling factor.
    """

    from scipy.integrate import simpson
    from archnemesis.Data.gas_data import Calc_mmw
    
    k_B = 1.38065e-23

    # Calculate layer geometric properties
    NLAY = len(BASEH)
    NPRO = len(H)
    DELH = np.zeros(NLAY)           # layer thickness
    DELH[0:-1] = (BASEH[1:]-BASEH[:-1])
    DELH[-1] = (H[-1]-BASEH[-1])

    sin = np.sin(LAYANG*np.pi/180)  # sin(viewing angle)
    cos = np.cos(LAYANG*np.pi/180)  # cos(viewing angle)

    z0 = RADIUS + LAYHT    # distance from planet centre to lowest layer base
    zmax = RADIUS+H[-1]    # maximum height defined in the profile

    SMAX = np.sqrt(zmax**2-(z0*sin)**2)-z0*cos # total path length
    BASES = np.sqrt((RADIUS+BASEH)**2-(z0*sin)**2)-z0*cos # Path lengths at base of layer
    DELS = np.zeros(NLAY)           # path length in each layer
    DELS[0:-1] = (BASES[1:]-BASES[:-1])
    DELS[-1] = (SMAX-BASES[-1])
    LAYSF = DELS/DELH               # Layer Scaling Factor
    BASET = interp(H,T,BASEH)       # layer base temperature

    # Note number of: profile points, layers, VMRs, aerosol types, flags
    if VMR.ndim == 1:
        NVMR = 1
    else:
        NVMR = len(VMR[0])

    if DUST.ndim == 1:
        NDUST = 1
    else:
        NDUST = len(DUST[0])

    #Checking consistency of DUST_UNITS and XMOLWT
    if DUST_UNITS is not None:
        for j in range(NDUST):
            if DUST_UNITS[j]==-1:
                if XMOLWT is None:
                    raise ValueError('error in layer_average :: if DUST_UNITS=-1 (particles per gram of atm), the XMOLWT must be defined')
        XMOLWT *= 1000. #Converting to molecular weight in g mol-1
    else:
        XMOLWT = np.zeros(NPRO)
    
    # HEIGHT = average layer height
    HEIGHT = np.zeros(NLAY)
    # PRESS = average layer pressure
    PRESS  = np.zeros(NLAY)
    # TEMP = average layer temperature
    TEMP   = np.zeros(NLAY)
    # TOTAM = total no. of molecules/aera
    TOTAM  = np.zeros(NLAY)
    # DUDS = no. of molecules per area per distance
    DUDS   = np.zeros(NLAY)
    # AMOUNT = no. of molecules/aera for each gas (molecules/m2)
    AMOUNT = np.zeros((NLAY, NVMR))
    # PP = gas partial pressures
    PP     = np.zeros((NLAY, NVMR))
    # MOLWT = mean molecular weight
    MOLWT  = np.zeros(NLAY)
    # CONT = no. of particles/area for each dust population (particles/m2)
    CONT = np.zeros((NLAY,NDUST))
    
    # FRAC = fraction of para-H2 in each layer
    FRAC   = np.zeros(NLAY)
    if PARAH2 is None:
        PARAH2 = np.zeros(NPRO)

    # Calculate average properties depending on intergration type
    if LAYINT == LayerIntegrationScheme.MID_PATH:
        # use layer properties at half path length in each layer
        S = np.zeros(NLAY)
        S[:-1] = (BASES[1:] + BASES[:-1])/2
        S[-1] = (SMAX+BASES[-1])/2
        # Derive other properties from S
        HEIGHT = np.sqrt(S**2+z0**2+2*S*z0*cos) - RADIUS
        PRESS = interp(H,P,HEIGHT)
        TEMP = interp(H,T,HEIGHT)
        FRAC = interp(H,PARAH2,HEIGHT)
        MOLWT = interp(H,XMOLWT,HEIGHT)
        
        # Ideal gas law: N/(Area*Path_length) = P/(k_B*T)
        DUDS = PRESS/(k_B*TEMP)
        TOTAM = DUDS*DELS
        # Use the volume mixing ratio information
        if VMR.ndim > 1:
            AMOUNT = np.zeros((NLAY, NVMR))
            for J in range(NVMR):
                AMOUNT[:,J] = interp(H, VMR[:,J], HEIGHT)
            PP = (AMOUNT.T * PRESS).T
            AMOUNT = (AMOUNT.T * TOTAM).T
        else:
            AMOUNT = interp(H, VMR, HEIGHT)
            PP = AMOUNT * PRESS
            AMOUNT = AMOUNT * TOTAM

        #Use the dust density information
        if DUST.ndim > 1:
            for J in range(NDUST):
                DD = interp(H, DUST[:,J],HEIGHT)  
                if DUST_UNITS is not None:
                    if DUST_UNITS[J] == -1:  #dust units are particles per gram of atm
                        CONT[:,J] =  DD * TOTAM * MOLWT / AVOGAD
                    else:
                         CONT[:,J] = DD * DELS  
                else:  
                    CONT[:,J] = DD * DELS
        else:
            DD = interp(H, DUST,HEIGHT)  
            if DUST_UNITS is not None:
                if DUST_UNITS[0] == -1: #dust units are particles per gram of atm
                    CONT =  DD * TOTAM * MOLWT / AVOGAD
                else:
                    CONT = DD * DELS
            else:
                CONT = DD * DELS

    elif LAYINT == LayerIntegrationScheme.ABSORBER_WEIGHTED_AVERAGE:
        # Curtis-Godson equivalent path for a gas with constant mixing ratio
        for I in range(NLAY):
            S0 = BASES[I]
            if I < NLAY-1:
                S1 = BASES[I+1]
            else:
                S1 = SMAX
            # sub-divide each layer into NINT layers
            S = np.linspace(S0, S1, NINT)
            h = np.sqrt(S**2+z0**2+2*S*z0*cos)-RADIUS
            p = interp(H,P,h)
            temp = interp(H,T,h)
            frac = interp(H,PARAH2,h)
            molwt = interp(H,XMOLWT,h)
            
            duds = p/(k_B*temp)

            amount = np.zeros((NINT, NVMR))
            
            TOTAM[I] = simpson(duds,x=S)
            HEIGHT[I]  = simpson(h*duds,x=S)/TOTAM[I]
            PRESS[I] = simpson(p*duds,x=S)/TOTAM[I]
            TEMP[I]  = simpson(temp*duds,x=S)/TOTAM[I]
            FRAC[I]  = simpson(frac*duds,x=S)/TOTAM[I]
            MOLWT[I] = simpson(molwt*duds,x=S)/TOTAM[I]
            
            if VMR.ndim > 1:
                amount = np.zeros((NINT, NVMR))
                for J in range(NVMR):
                    amount[:,J] = interp(H, VMR[:,J], h)
                    AMOUNT[I,J] = simpson(amount[:,J]*duds,x=S)
                pp = (amount.T * p).T     # gas partial pressures
                for J in range(NVMR):
                    PP[I, J] = simpson(pp[:,J]*duds,x=S)/TOTAM[I]
            else:
                amount = interp(H, VMR, h)
                pp = amount * p
                AMOUNT[I] = simpson(amount*duds,x=S)
                PP[I] = simpson(pp*duds,x=S)/TOTAM[I]
                        
            if DUST.ndim > 1:
                dd = np.zeros((NINT,NDUST))
                for J in range(NDUST):
                    dd[:,J] = interp(H, DUST[:,J], h)
                    if DUST_UNITS is not None:
                        if DUST_UNITS[J] == -1: #dust units are particles per gram of atm
                            CONT[I,J] = simpson(dd[:,J]*duds*molwt/AVOGAD,x=S)
                        else:
                            CONT[I,J] = simpson(dd[:,J],x=S)
                    else:
                        CONT[I,J] = simpson(dd[:,J],x=S)     
            else:
                dd = interp(H, DUST, h) 
                if DUST_UNITS is not None:
                    if DUST_UNITS[0] == -1: #dust units are particles per gram of atm
                        CONT[I] = simpson(dd*duds*molwt/AVOGAD,x=S)
                    else:
                        CONT[I] = simpson(dd,x=S)
                else:
                    CONT[I] = simpson(dd,x=S)
            
    # Scale back to vertical layers
    TOTAM = TOTAM / LAYSF
    if VMR.ndim > 1:
        AMOUNT = (AMOUNT.T * LAYSF**-1 ).T
    else:
        AMOUNT = AMOUNT/LAYSF

    if DUST.ndim > 1:
        CONT = (CONT.T * LAYSF**-1 ).T
    else:
        CONT = CONT/LAYSF
    return HEIGHT,PRESS,TEMP,TOTAM,AMOUNT,PP,CONT,FRAC,DELH,BASET,LAYSF

#########################################################################################

def layer_averageg(RADIUS, H, P, T, ID, VMR, DUST, PARAH2, BASEH, BASEP,
                  LAYANG=0.0, LAYINT=LayerIntegrationScheme.MID_PATH, LAYHT=0.0, NINT=101, DUST_UNITS=None, XMOLWT=None):
    """
    Calculates average layer properties.
    Takes an atmosphere profile and a layering shceme specified by
    layer base altitudes and pressures and returns averaged height,
    pressure, temperature, VMR for each layer.

    Inputs
    ------
    @param RADIUS: real
        Reference planetary radius where H=0.  Usually at surface for
        terrestrial planets, or at 1 bar pressure level for gas giants.
    @param H: 1D array
        Input profile heights
    @param P: 1D array
        Input profile pressures
    @param T: 1D array
        Input profile temperatures
    @param ID: 1D array
        Gas identifiers.
    @param VMR: 2D array
        VMR[i,j] is Volume Mixing Ratio of gas j at vertical point i
        the column j corresponds to the gas with RADTRANS ID ID[j].
    @param DUST: 2D array
        DUST[i,j] is dust density of dust popoulation j at vertical point i  (particles/m3)
    @param PARAH2: 1D array
        Para-H2 fraction
    @param BASEH: 1D array
        Heights of the layer bases.
    @param BASEP: 1D array
        Pressures of the layer bases.
    @param LAYANG: real
        Zenith angle in degrees defined at LAYHT.
    @param LAYINT: LayerIntegrationScheme (enum)
        Layer integration scheme
        LayerIntegrationScheme.MID_PATH = use properties at mid path
        LayerIntegrationScheme.ABSORBER_WEIGHTED_AVERAGE = use absorber amount weighted average values
    @param LAYHT: real
        Height of the base of the lowest layer. Default 0.0.
    @param NINT: int
        Number of integration points to be used if LAYINT=LayerIntegrationScheme.ABSORBER_WEIGHTED_AVERAGE.
    @param XMOLWT: 1D array
        Molecular weight vertical profile in kg mol-1 (only required if DUST_UNITS is activated)

    Returns
    -------
    @param HEIGHT: 1D array
        Representative height for each layer
    @param PRESS: 1D array
        Representative pressure for each layer
    @param TEMP: 1D array
        Representative pressure for each layer
    @param TOTAM: 1D array
        Total gaseous absorber amounts along the line-of-sight path, i.e.
        number of molecules per area.
    @param AMOUNT: 1D array
        Representative absorber amounts of each gas at each layer.
        AMOUNT[I,J] is the representative number of gas J molecules
        in layer I in the form of number of molecules per area.
    @param PP: 1D array
        Representative partial pressure for each gas at each layer.
        PP[I,J] is the representative partial pressure of gas J in layer I.
    @param DELH: 1D array
        Layer thicnkness.
    @param BASET: 1D array
        Layer base temperature.
    @param LAYSF: 1D array
        Layer scaling factor.
    @param DTE: 2D array
        Matrix relating the temperature in each layer (TEMP) to the input temperature profile (T)
    @param DAM: 2D array
        Matrix relating the absorber amounts in each layer (AMOUNT) to the input VMR profile (VMR)
    @param DCO: 2D array
        Matrix relating the dust amounts in each layer (CONT) to the input dust profile (DUST)
    """

    from scipy.integrate import simpson
    from archnemesis.Data.gas_data import Calc_mmw

    k_B = 1.38065e-23

    # Calculate layer geometric properties
    NLAY = len(BASEH)
    NPRO = len(H)
    DELH = np.zeros(NLAY)           # layer thickness
    DELH[0:-1] = (BASEH[1:]-BASEH[:-1])
    DELH[-1] = (H[-1]-BASEH[-1])

    sin = np.sin(LAYANG*np.pi/180)  # sin(viewing angle)
    cos = np.cos(LAYANG*np.pi/180)  # cos(viewing angle)

    z0 = RADIUS + LAYHT    # distance from planet centre to lowest layer base
    zmax = RADIUS+H[-1]    # maximum height defined in the profile

    SMAX = np.sqrt(zmax**2-(z0*sin)**2)-z0*cos # total path length
    BASES = np.sqrt((RADIUS+BASEH)**2-(z0*sin)**2)-z0*cos # Path lengths at base of layer
    DELS = np.zeros(NLAY)           # path length in each layer
    DELS[0:-1] = (BASES[1:]-BASES[:-1])
    DELS[-1] = (SMAX-BASES[-1])
    LAYSF = DELS/DELH               # Layer Scaling Factor
    BASET = interp(H,T,BASEH)       # layer base temperature

    # Note number of: profile points, layers, VMRs, aerosol types, flags
    if VMR.ndim == 1:
        NVMR = 1
    else:
        NVMR = len(VMR[0])

    if DUST.ndim == 1:
        NDUST = 1
    else:
        NDUST = len(DUST[0])

    #Checking consistency of DUST_UNITS and XMOLWT
    if DUST_UNITS is not None:
        for j in range(NDUST):
            if DUST_UNITS[j]==-1:
                if XMOLWT is None:
                    raise ValueError('error in layer_average :: if DUST_UNITS=-1 (particles per gram of atm), the XMOLWT must be defined')
        XMOLWT *= 1000. #Converting to molecular weight in g mol-1
    else:
        XMOLWT = np.zeros(NPRO)

    # HEIGHT = average layer height
    HEIGHT = np.zeros(NLAY)
    # PRESS = average layer pressure
    PRESS  = np.zeros(NLAY)
    # TEMP = average layer temperature
    TEMP   = np.zeros(NLAY)
    # TOTAM = total no. of molecules/aera
    TOTAM  = np.zeros(NLAY)
    # DUDS = no. of molecules per area per distance
    DUDS   = np.zeros(NLAY)
    # AMOUNT = no. of molecules/aera for each gas (molecules/m2)
    AMOUNT = np.zeros((NLAY, NVMR))
    # PP = gas partial pressures
    PP     = np.zeros((NLAY, NVMR))
    # MOLWT = mean molecular weight
    MOLWT  = np.zeros(NLAY)
    # CONT = no. of particles/area for each dust population (particles/m2)
    CONT = np.zeros((NLAY,NDUST))
    # FRAC = fraction of para-H2 in each layer
    FRAC = np.zeros(NLAY)
    if PARAH2 is None:
        PARAH2 = np.zeros(NPRO)
    #DTE = matrix to relate the temperature in each layer (TEMP) to the temperature in the input profiles (T)
    DTE = np.zeros((NLAY, NPRO))
    #DCO = matrix to relate the dust abundance in each layer (CONT) to the dust abundance in the input profiles (DUST)
    DCO = np.zeros((NLAY, NPRO))
    #DAM = matrix to relate the gaseous abundance in each layer (AMOUNT) to the gas VMR in the input profiles (VMR)
    DAM = np.zeros((NLAY, NPRO))
    #DPH = matrix to relate para-H2 in each layer (FRAC) to that in the input profiles (PARAH2)
    DPH = np.zeros((NLAY, NPRO))

    # Ensure NINT is odd for Simpson's rule
    if NINT % 2 == 0:
        raise ValueError("NINT must be odd for Simpson's rule.")
    # Initialize weights for Simpson's rule
    w = np.ones(NINT)  # Start with all ones
    # Set 4s for odd indices (1-based, so 1,3,5,... -> 0-based 1,3,5,...)
    w[1:-1:2] = 4.
    # Set 2s for even indices (1-based 2,4,6,... -> 0-based 2,4,6,...)
    w[2:-1:2] = 2.

    # Calculate average properties depending on intergration type
    if LAYINT == LayerIntegrationScheme.MID_PATH:
        # use layer properties at half path length in each layer
        S = np.zeros(NLAY)
        S[:-1] = (BASES[1:] + BASES[:-1])/2
        S[-1] = (SMAX+BASES[-1])/2
        # Derive other properties from S
        HEIGHT = np.sqrt(S**2+z0**2+2*S*z0*cos) - RADIUS
        PRESS = interp(H,P,HEIGHT)
        MOLWT = interp(H,XMOLWT,HEIGHT)
        TEMP,J,F = interpg(H,T,HEIGHT)
        for ilay in range(NLAY):
            DTE[ilay,J[ilay]] = DTE[ilay,J[ilay]] + (1.0-F[ilay])
            DTE[ilay,J[ilay]+1] = DTE[ilay,J[ilay]+1] + (F[ilay])

        FRAC,J,F = interpg(H,PARAH2,HEIGHT)
        for ilay in range(NLAY):
            DPH[ilay,J[ilay]] = DPH[ilay,J[ilay]] + (1.0-F[ilay])
            DPH[ilay,J[ilay]+1] = DPH[ilay,J[ilay]+1] + (F[ilay])
            
        # Ideal gas law: N/(Area*Path_length) = P/(k_B*T)
        DUDS = PRESS/(k_B*TEMP) 
        TOTAM = DUDS*DELS
        # Use the volume mixing ratio information
        if VMR.ndim > 1:
            AMOUNT = np.zeros((NLAY, NVMR))
            for J in range(NVMR):
                AMOUNT[:,J],JJ,F = interpg(H, VMR[:,J], HEIGHT)
            PP = (AMOUNT.T * PRESS).T
            AMOUNT = (AMOUNT.T * TOTAM).T
        else:
            AMOUNT,JJ,F = interpg(H, VMR, HEIGHT)
            PP = AMOUNT * PRESS
            AMOUNT = AMOUNT * TOTAM

        for ilay in range(NLAY):
            DAM[ilay,JJ[ilay]] = DAM[ilay,JJ[ilay]] + (1.0-F[ilay])*TOTAM[ilay]
            DAM[ilay,JJ[ilay]+1] = DAM[ilay,JJ[ilay]+1] + (F[ilay])*TOTAM[ilay]

        #Use the dust density information
        if DUST.ndim > 1:
            for J in range(NDUST):
                DD,JJ,F = interpg(H, DUST[:,J],HEIGHT)  
                CONT[:,J] = DD * DELS
        else:
            #DD = interp(H, DUST,HEIGHT)  
            DD,JJ,F = interpg(H, DUST,HEIGHT)
            CONT = DD * DELS            


        #Use the dust density information
        dust_units_flag = False
        if DUST.ndim > 1:
            for J in range(NDUST):
                DD = interp(H, DUST[:,J],HEIGHT)  
                if DUST_UNITS is not None:
                    if DUST_UNITS[J] == -1:  #dust units are particles per gram of atm
                        dust_units_flag = True
                        CONT[:,J] =  DD * TOTAM * MOLWT / AVOGAD
                    else:
                         CONT[:,J] = DD * DELS  
                else:  
                    CONT[:,J] = DD * DELS
        else:
            DD = interp(H, DUST,HEIGHT)  
            if DUST_UNITS is not None:
                if DUST_UNITS[0] == -1: #dust units are particles per gram of atm
                    dust_units_flag = True
                    CONT =  DD * TOTAM * MOLWT / AVOGAD
                else:
                    CONT = DD * DELS
            else:
                CONT = DD * DELS

        if dust_units_flag is False:
            for ilay in range(NLAY):
                DCO[ilay,JJ[ilay]] = DCO[ilay,JJ[ilay]] + (1.0-F[ilay])
                DCO[ilay,JJ[ilay]+1] = DCO[ilay,JJ[ilay]+1] + (F[ilay])
        else:
                DCO[ilay,JJ[ilay]] = DCO[ilay,JJ[ilay]] + (1.0-F[ilay]) * TOTAM * MOLWT /AVOGAD
                DCO[ilay,JJ[ilay]+1] = DCO[ilay,JJ[ilay]+1] + (F[ilay]) * TOTAM * MOLWT /AVOGAD
            
    elif LAYINT == LayerIntegrationScheme.ABSORBER_WEIGHTED_AVERAGE:
        
        # Curtis-Godson equivalent path for a gas with constant mixing ratio
        for I in range(NLAY):
            S0 = BASES[I]
            if I < NLAY-1:
                S1 = BASES[I+1]
            else:
                S1 = SMAX
            # sub-divide each layer into NINT layers
            S = np.linspace(S0, S1, NINT)
            h = np.sqrt(S**2+z0**2+2*S*z0*cos)-RADIUS   #altitude of the NINT points (m)
            p = interp(H,P,h)                           #pressure at the NINT points (Pa)
            molwt = interp(H,XMOLWT,h)                           #molecular weight at the NINT points (g mol-1)
            temp,JJ,F = interpg(H,T,h)                  #temperature at the NINT points (K)
            frac,JJP,FP = interpg(H,PARAH2,h)           #fraction of para-H2 at the NINT points 
            duds = p/(k_B*temp)                         #number density at the NINT points (m-3)

            amount = np.zeros((NINT, NVMR))

            TOTAM[I] = simpson(duds,x=S)                #column density (m-2)
            HEIGHT[I]  = simpson(h*duds,x=S)/TOTAM[I]   #effective altitude of the layer (m)
            PRESS[I] = simpson(p*duds,x=S)/TOTAM[I]     #effective pressure of the layer (Pa)
            TEMP[I]  = simpson(temp*duds,x=S)/TOTAM[I]  #effective temperature of the layer (K)
            FRAC[I]  = simpson(frac*duds,x=S)/TOTAM[I]  #effective fraction of para-H2 in the layer

            for iint in range(NINT):
                DTE[I,JJ[iint]] = DTE[I,JJ[iint]] + (1.-F[iint])*w[iint]*duds[iint]
                DTE[I,JJ[iint]+1] = DTE[I,JJ[iint]+1] + (F[iint])*w[iint]*duds[iint]

                DPH[I,JJP[iint]] = DPH[I,JJP[iint]] + (1.-FP[iint])*w[iint]*duds[iint]
                DPH[I,JJP[iint]+1] = DPH[I,JJP[iint]+1] + (FP[iint])*w[iint]*duds[iint]

                
            #Computing the gas column densities and gradients
            if VMR.ndim > 1:
                amount = np.zeros((NINT, NVMR))
                for J in range(NVMR):
                    amount[:,J],JJ,F = interpg(H, VMR[:,J], h)    #VMR of gas J at the NINT points
                    AMOUNT[I,J] = simpson(amount[:,J]*duds,x=S)   #column density of gas J in the layer (m-2)
                pp = (amount.T * p).T     # gas partial pressures at the NINT points
                for J in range(NVMR):
                    PP[I, J] = simpson(pp[:,J]*duds,x=S)/TOTAM[I]   #effective partial pressure of gas J in the layer
            else:
                amount,JJ,F = interpg(H, VMR, h)   
                pp = amount * p
                AMOUNT[I] = simpson(amount*duds,x=S)
                PP[I] = simpson(pp*duds,x=S)/TOTAM[I]

            for iint in range(NINT):
                DAM[I,JJ[iint]] = DAM[I,JJ[iint]] + (1.-F[iint])*duds[iint]*w[iint]
                DAM[I,JJ[iint]+1] = DAM[I,JJ[iint]+1] + (F[iint])*duds[iint]*w[iint]


            #Computing the aerosol column densities and gradients
            dust_units_flag = False
            if DUST.ndim > 1:
                dd = np.zeros((NINT,NDUST))
                for J in range(NDUST):
                    dd[:,J],JJ,F = interpg(H, DUST[:,J], h)   #dust density (m-3) of dust population J at the NINT points
                    if DUST_UNITS is not None:
                        if DUST_UNITS[J] == -1: #dust units are in particles per gram of atm
                            dust_units_flag = True
                            CONT[I,J] = simpson(dd[:,J]*duds*molwt/AVOGAD,x=S)  #column density of dust population J in the layer (m-2)
                        else:
                            CONT[I,J] = simpson(dd[:,J],x=S)          #column density of dust population J in the layer (m-2)
                    else:
                        CONT[I,J] = simpson(dd[:,J],x=S)          #column density of dust population J in the layer (m-2)
            else:
                dd,JJ,F = interpg(H, DUST, h) 
                if DUST_UNITS is not None:
                    if DUST_UNITS[0] == -1: #dust units are in particles per gram of atm
                        dust_units_flag = True
                        CONT[I] = simpson(dd*duds*molwt/AVOGAD,x=S)
                    else:
                        CONT[I] = simpson(dd,x=S)
                else:
                    CONT[I] = simpson(dd,x=S)
                
            if dust_units_flag is False:
                for iint in range(NINT):
                    DCO[I,JJ[iint]] = DCO[I,JJ[iint]] + (1.-F[iint])*w[iint]
                    DCO[I,JJ[iint]+1] = DCO[I,JJ[iint]+1] + (F[iint])*w[iint]
            else:
                for iint in range(NINT):
                    DCO[I,JJ[iint]] = DCO[I,JJ[iint]] + (1.-F[iint])*w[iint]*duds[iint]*molwt[iint]/AVOGAD
                    DCO[I,JJ[iint]+1] = DCO[I,JJ[iint]+1] + (F[iint])*w[iint]*duds[iint]*molwt[iint]/AVOGAD

        #Finishing the integration for the matrices
        for IPRO in range(NPRO):
            DTE[:,IPRO] = DTE[:,IPRO] * DELS / (NINT-1.) / 3. / TOTAM
            DPH[:,IPRO] = DPH[:,IPRO] * DELS / (NINT-1.) / 3. / TOTAM
            DAM[:,IPRO] = DAM[:,IPRO] * DELS / (NINT-1.) / 3.
            DCO[:,IPRO] = DCO[:,IPRO] * DELS / (NINT-1.) / 3.

    # Scale back to vertical layers
    TOTAM = TOTAM / LAYSF
    if VMR.ndim > 1:
        AMOUNT = (AMOUNT.T * LAYSF**-1 ).T
    else:
        AMOUNT = AMOUNT/LAYSF

    if DUST.ndim > 1:
        CONT = (CONT.T * LAYSF**-1 ).T
    else:
        CONT = CONT/LAYSF

    for IPRO in range(NPRO):
        DAM[:,IPRO] = DAM[:,IPRO] / LAYSF
        DCO[:,IPRO] = DCO[:,IPRO] / LAYSF

    return HEIGHT,PRESS,TEMP,TOTAM,AMOUNT,PP,CONT,FRAC,DELH,BASET,LAYSF,DTE,DAM,DCO,DPH

#########################################################################################

def layer_split(RADIUS, H, P, LAYANG=0.0, LAYHT=0.0, NLAY=20,
        LAYTYP=LayerType.EQUAL_LOG_PRESSURE, H_base=None, P_base=None):
    """
    Splits an atmosphere into NLAY layers.
    Takes a set of altitudes H with corresponding pressures P and returns
    the altitudes and pressures of the base of the layers.

    Inputs
    ------
    @param RADIUS: real
        Reference planetary radius where H=0.  Usually at surface for
        terrestrial planets, or at 1 bar pressure level for gas giants.
    @param H: 1D array
        Heights at which the atmosphere profile is specified.
        (At altitude H[i] the pressure is P[i].)
    @param P: 1D array
        Pressures at which the atmosphere profile is specified.
        (At pressure P[i] the altitude is H[i].)
    @param LAYANG: real
        Zenith angle in degrees defined at LAYHT.
        Default 0.0 (nadir geometry). Only needed for layer type LayerType.EQUAL_PATH_LENGTH.
    @param LAYHT: real
        Height of the base of the lowest layer. Default 0.0.
    @param NLAY: int
        Number of layers to split the atmosphere into. Default 20.
    @param LAYTYP: LayerType (enum)
        Integer specifying how to split up the layers. Default LayerType.EQUAL_LOG_PRESSURE.
        LayerType.EQUAL_PRESSURE = by equal changes in pressure
        LayerType.EQUAL_LOG_PRESSURE = by equal changes in log pressure
        LayerType.EQUAL_HEIGHT = by equal changes in height
        LayerType.EQUAL_PATH_LENGTH = by equal changes in path length at LAYANG
        LayerType.BASE_PRESSURE = layer base pressure levels specified by P_base
        LayerType.BASE_HEIGHT = layer base height levels specified by H_base
        Note BASE_PRESSURE and BASE_HEIGHT force NLAY = len(P_base) or len(H_base).
    @param H_base: 1D array
        Heights of the layer bases defined by user. Default None.
    @param P_base: 1D array
        Pressures of the layer bases defined by user. Default None.

    Returns
    -------
    @param BASEH: 1D array
        Heights of the layer bases.
    @param BASEP: 1D array
        Pressures of the layer bases.
    """

    if LAYHT<H[0]:
        _lgr.warning(' from layer_split() :: LAYHT < H(0). Resetting LAYHT')
        LAYHT = H[0]

    if LAYTYP == LayerType.EQUAL_PRESSURE: # split by equal pressure intervals
        PBOT = interp(H,P,LAYHT)  # pressure at base of lowest layer
        BASEP = np.linspace(PBOT,P[-1],NLAY+1)[:-1]
        BASEH = interp(P[::-1],H[::-1],BASEP)

    elif LAYTYP == LayerType.EQUAL_LOG_PRESSURE: # split by equal log pressure intervals
        PBOT = interp(H,P,LAYHT)  # pressure at base of lowest layer
        BASEP = np.logspace(np.log10(PBOT),np.log10(P[-1]),NLAY+1)[:-1]
        BASEH = interp(P[::-1],H[::-1],BASEP)

    elif LAYTYP == LayerType.EQUAL_HEIGHT: # split by equal height intervals
        BASEH = np.linspace(LAYHT, H[-1], NLAY+1)[:-1]
        BASEP = interp(H,P,BASEH)

    elif LAYTYP == LayerType.EQUAL_PATH_LENGTH: # split by equal line-of-sight path intervals
        assert LAYANG<=90 and LAYANG>=0,\
            'Zennith angle should be in [0,90]'
        sin = np.sin(LAYANG*np.pi/180)      # sin(zenith angle)
        cos = np.cos(LAYANG*np.pi/180)      # cos(zenith angle)
        z0 = RADIUS + LAYHT                 # distance from centre to lowest layer's base
        zmax = RADIUS+H[-1]                 # maximum height
        SMAX = np.sqrt(zmax**2-(z0*sin)**2)-z0*cos # total path length
        BASES = np.linspace(0, SMAX, NLAY+1)[:-1]
        BASEH = np.sqrt(BASES**2+z0**2+2*BASES*z0*cos) - RADIUS
        logBASEP = interp(H,np.log(P),BASEH)
        BASEP = np.exp(logBASEP)

    elif LAYTYP == LayerType.BASE_PRESSURE: # split by specifying input base pressures
        if np.isnan(P_base) is True:
            raise ValueError('error in layer_split() :: need input layer base pressures')
        assert  (P_base[-1] >= P[-1]) and (P_base[0] <= P[0]), \
            'Input layer base pressures out of range of atmosphere profile'
        BASEP = P_base
        NLAY = len(BASEP)
        BASEH = interp(P[::-1],H[::-1],BASEP)

    elif LAYTYP == LayerType.BASE_HEIGHT: # split by specifying input base heights
        BASEH = H_base
        logBASEP = interp(H,np.log(P),BASEH)
        BASEP = np.exp(logBASEP)

    else:
        raise('Layering scheme not defined')

    return BASEH, BASEP

#########################################################################################

def read_hlay():

    """
        FUNCTION NAME : read_hlay()
        
        DESCRIPTION : Read the height.lay file used to set the altitude of the base of the layers
                      in a Nemesis run
        
        INPUTS : None
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            nlay :: Number of layers in atmospheric model

        
        CALLING SEQUENCE:
        
            nlay,hbase = read_hlay()
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """

    f = open('height.lay','r')

    header = f.readline().split()

    s = f.readline().split()
    nlay = int(s[0])
    hbase = np.zeros(nlay)
    for i in range(nlay):
        s = f.readline().split()
        hbase[i] = float(s[0])

    return nlay,hbase

#########################################################################################


def read_play():

    """
        FUNCTION NAME : read_play()
        
        DESCRIPTION : Read the height.lay file used to set the altitude of the base of the layers
                      in a Nemesis run
        
        INPUTS : None
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            nlay :: Number of layers in atmospheric model

        
        CALLING SEQUENCE:
        
            nlay,hbase = read_play()
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """

    f = open('pressure.lay','r')

    header = f.readline().split()

    s = f.readline().split()
    nlay = int(s[0])
    pbase = np.zeros(nlay)
    for i in range(nlay):
        s = f.readline().split()
        pbase[i] = float(s[0])

    return nlay,pbase

#########################################################################################