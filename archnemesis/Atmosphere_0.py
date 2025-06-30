#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:27:12 2021

@author: jingxuanyang and juanaldayparejo

Atmosphere Class.
"""
from __future__ import annotations #  for 3.9 compatability
from archnemesis import Data
from archnemesis import *
import numpy as np
from scipy.special import legendre

from archnemesis.enums import PlanetEnum, AtmosphericProfileFormatEnum, AtmosphericProfileType

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

class Atmosphere_0:
    """
    Clear atmosphere. Simplest possible profile.
    """
    def __init__(
            self, 
            runname='', 
            Fortran = False, 
            NP=10, 
            NVMR=6, 
            NDUST=0, 
            NLOCATIONS=1, 
            IPLANET=PlanetEnum.UNDEFINED, 
            AMFORM=AtmosphericProfileFormatEnum.CALC_MOLECULAR_WEIGHT_SCALE_VMR_TO_ONE
        ):
        """
        Set up an atmosphere profile with NP points and NVMR gases.
        Use the class methods to edit Height, Pressure, Temperature and
        gas Volume Mixing Ratios after creating an Atmosphere_0 object, e.g.
            atm = Atmosphere_0('Jupiter',100,6,[1,2,4,5,39,40],
                      [0,0,0,0,0,0],0,1,1)
            atm.edit_H(your_height_profile)
            atm.edit_P(your_pressure_profile)
            atm.edit_T(your_temperature_profile)
            atm.edit_VMR(your_VMR_profile)
        Can write to a *.prf file in Nemesis format.

        Inputs
        ------
        @param runname: str
            Name of this particular profile (no space)
        @param NP: int,
            Number of points defined in the profile
        @param NVMR: int
            Number of gases in the profile (expressed as volume mixing ratios)
        @param NDUST: int
            Number of aerosol species in the profile (expressed as particles per cubic meter)
        @param IPLANET: int
            Planet ID
        @param LATITUDE: real
            Planetocentric latitude
        @param AMFORM: int,
            Format of the atmospheric profile, default AMFORM=1:
            assume that at each level the VMRs add up to 1.0.

        Attributes
        ----------
        @attribute ID: 1D array
            Gas ID for each gas to be defined in the profile
        @attribute ISO: 1D array
            Isotope ID for each gas, default 0 for all
            isotopes in terrestrial relative abundance
        @attribute H: 1D array
            Height in m of each points above reference level:
            ground for terrestrial planets,
            usually 1-bar level for gas giants
        @attribute P: 1D array
            Pressure in Pa of each point defined in the profile
        @attribute T: 1D array
            Temperature in K at each point defined in the profile
        @attribute VMR: 2D array
            VMR[i,j] is Volume Mixing Ratio of gas j at vertical point i
            the column j corresponds to the gas with RADTRANS ID ID[j]
            and RADTRANS isotope ID ISO[j]
        @attribute MOLWT: float
            Molecular weight of the atmosphere in kg mol-1
        @attribute PARAH2: 1D array
            Verical profile of para-H2 fraction (para-H2/total amount of H2). 
            The value 1-PARAH2 would be the ortho-H2 fraction.

        
        Methods
        -------
        Atmosphere_0.assess()
        Atmosphere_0.summary_info()
        Atmosphere_0.write_hdf5()
        Atmosphere_0.read_hdf5()

        Atmosphere_0.edit_H()
        Atmosphere_0.edit_P()
        Atmosphere_0.edit_T()
        Atmosphere_0.edit_VMR()
        Atmosphere_0.edit_DUST()

        Atmosphere_0.adjust_VMR()
        Atmosphere_0.adjust_hydrostatP()
        Atmosphere_0.adjust_hydrostatH()
        Atmosphere_0.calc_molwt()
        Atmosphere_0.calc_rho()
        Atmosphere_0.calc_numdens()
        Atmosphere_0.calc_radius()
        Atmosphere_0.calc_grav()

        Atmosphere_0.locate_gas()
        Atmosphere_0.add_gas()
        Atmosphere_0.remove_gas()
        Atmosphere_0.update_gas()
        
        Atmosphere_0.select_location()
        Atmosphere_0.calc_coldens()
        
        Atmosphere_0.read_ref()
        Atmosphere_0.write_ref()
        Atmosphere_0.read_aerosol()
        Atmosphere_0.write_aerosol()
        Atmosphere_0.read_parah2()
        Atmosphere_0.write_parah2()
        Atmosphere_0.read_vpf()

        Atmosphere_0.plot_Atm()
        Atmosphere_0.plot_Dust()
        Atmosphere_0.plot_map()
        """

        self.runname = runname
        self.NP = NP
        self.NVMR = NVMR
        self.NDUST = NDUST
        #self.IPLANET : PlanetEnum = IPLANET
        #self.AMFORM : AtmosphericProfileFormatEnum = AMFORM
        self.NLOCATIONS = NLOCATIONS
        self.Fortran = Fortran

        # Input the following profiles using the edit_ methods.
        self.RADIUS = None    #float of (NLOCATIONS) #m
        self.LATITUDE = None  #float or (NLOCATIONS)
        self.LONGITUDE = None #float or (NLOCATIONS)
        self.ID = None #np.zeros(NVMR)
        self.ISO = None #np.zeros(NVMR)
        self.H = None # np.zeros(NP) or np.zeros((NP,NLOCATIONS)) #m
        self.P = None # np.zeros(NP) or np.zeros((NP,NLOCATIONS)) #Pa
        self.T =  None # np.zeros(NP) or np.zeros((NP,NLOCATIONS)) #K
        self.MOLWT = None #np.zeros(NP) or np.zeros((NP,NLOCATIONS)) #kg mol-1
        self.GRAV = None #np.zeros(NP) or np.zeros((NP,NLOCATIONS))    
        self.VMR = None # np.zeros((NP,NVMR)) or np.zeros((NP,NVMR,NLOCATIONS)) 
        self.DUST = None # np.zeros((NP,NDUST)) or np.zeros((NP,NDUST,NLOCATIONS)) #particles per m3
        self.DUST_UNITS_FLAG = None # np.zeros(NDUST), -1 for legacy units (g cm^-3), None for standard (number m^-3)
        self.DUST_RENORMALISATION = {} # flags for normalising clouds to specific opacity
        self.PARAH2 = None # np.zeros(NP) 
        
        self.SVP = {} # Flags for limiting gas profiles to saturated profiles (from .vpf file)
        
        
        # private attributes
        self._iplanet = None
        self._amform = None
        
        # set properties
        self.IPLANET = IPLANET
        self.AMFORM = AMFORM
    ##################################################################################

    @property
    def IPLANET(self) -> PlanetEnum:
        return self._iplanet
    
    @IPLANET.setter
    def IPLANET(self, value):
        self._iplanet = PlanetEnum(value)
    
    @property
    def AMFORM(self) -> AtmosphericProfileFormatEnum:
        return self._amform
    
    @AMFORM.setter
    def AMFORM(self, value):
        self._amform = AtmosphericProfileFormatEnum(value)

    def assess(self):
        """
        Assess whether the different variables have the correct dimensions and types
        """

        #Checking some common parameters to all cases
        assert np.issubdtype(type(self.NP), np.integer) == True , \
            'NP must be int'
        assert self.NP > 0 , \
            'NP must be >0'
        
        assert np.issubdtype(type(self.NVMR), np.integer) == True , \
            'NVMR must be int'
        assert self.NP > 0 , \
            'NVMR must be >0'
        
        assert np.issubdtype(type(self.NLOCATIONS), np.integer) == True , \
            'NLOCATIONS must be int'
        assert self.NLOCATIONS > 0 , \
            'NLOCATIONS must be >0'
        
        assert self.AMFORM in AtmosphericProfileFormatEnum, \
            f"AMFORM must be one of {tuple(AtmosphericProfileFormatEnum)}"
        
        assert self.IPLANET in PlanetEnum, \
            f"IPLANET must be one of {tuple(PlanetEnum)}"
        
        assert len(self.ID) == self.NVMR , \
            'ID must have size (NVMR)'
        assert len(self.ISO) == self.NVMR , \
            'ISO must have size (NVMR)'
        
        if self.NLOCATIONS==1:

            assert np.issubdtype(type(self.LATITUDE), float) == True , \
                'LATITUDE must be float'
            assert abs(self.LATITUDE) < 90.0 , \
                'LATITUDE must be within -90 to 90 degrees'
            assert np.issubdtype(type(self.LONGITUDE), float) == True , \
                'LONGITUDE must be float'
            
            assert len(self.H) == self.NP , \
                'H must have size (NP)'
            assert len(self.P) == self.NP , \
                'P must have size (NP)'
            assert len(self.T) == self.NP , \
                'T must have size (NP)'

            if self.AMFORM==AtmosphericProfileFormatEnum.MOLECULAR_WEIGHT_DEFINED:                       
                assert self.MOLWT is not None , \
                    f'MOLWT must be defined if AMFORM={AtmosphericProfileFormatEnum.MOLECULAR_WEIGHT_DEFINED}'
            
            assert self.VMR.shape == (self.NP,self.NVMR) , \
                'VMR must have size (NP,NVMR)'
            
            if self.NDUST>0:
                assert self.DUST.shape == (self.NP,self.NDUST) , \
                    'DUST must have size (NP,NDUST)'
                    
            if self.PARAH2 is not None:
                assert len(self.PARAH2) == self.NP , \
                    'PARAH2 must have size (NP)'
                
        elif self.NLOCATIONS>1:

            assert len(self.LATITUDE) == self.NLOCATIONS , \
                'LATITUDE must have size (NLOCATIONS)'
            
            assert len(self.LONGITUDE) == self.NLOCATIONS , \
                'LONGITUDE must have size (NLOCATIONS)'
            
            assert self.H.shape == (self.NP,self.NLOCATIONS) , \
                'H must have size (NP,NLOCATIONS)'
            
            assert self.P.shape == (self.NP,self.NLOCATIONS) , \
                'P must have size (NP,NLOCATIONS)'
            
            assert self.T.shape == (self.NP,self.NLOCATIONS) , \
                'T must have size (NP,NLOCATIONS)'
            
            if self.AMFORM==AtmosphericProfileFormatEnum.MOLECULAR_WEIGHT_DEFINED:          
                assert self.MOLWT is not None , \
                    'MOLWT must be define if AMFORM=0'
                
            assert self.VMR.shape == (self.NP,self.NVMR,self.NLOCATIONS) , \
                'VMR must have size (NP,NVMR,NLOCATIONS)'
            
            if self.NDUST>0:
                assert self.DUST.shape == (self.NP,self.NDUST,self.NLOCATIONS) , \
                    'DUST must have size (NP,NDUST,NLOCATIONS)'
            
            if self.PARAH2 is not None:
                assert self.PARAH2.shape == (self.NP,self.NLOCATIONS) , \
                    'PARAH2 must have size (NP,NLOCATIONS)'

    ##################################################################################

    def summary_info(self):
        """
        Subroutine to print summary of information about the class
        """      
        
        from archnemesis.Data.gas_data import gas_info
        from archnemesis.Data.planet_data import planet_info

        data = planet_info[str(int(self.IPLANET))]
        _lgr.info('Planet :: '+data['name'])
        _lgr.info(f'Number of profiles ::  {(self.NLOCATIONS)}')
        _lgr.info(f'Latitude of profiles ::  {(self.LATITUDE)}')
        _lgr.info(f'Number of altitude points ::  {(self.NP)}')
        _lgr.info(f'Minimum/maximum heights (km) ::  {(self.H.min()/1.0e3,self.H.max()/1.0e3)}')
        _lgr.info(f'Maximum/minimum pressure (atm) ::  {(self.P.max()/101325.,self.P.min()/101325.)}')
        _lgr.info(f'Maximum/minimum temperature (K) {( self.T.max(),self.T.min())}')
        if self.GRAV is not None:
            _lgr.info(f'Maximum/minimum gravity (m/s2) ::  {(np.round(self.GRAV.max(),2),np.round(self.GRAV.min(),2))}')
        if self.MOLWT is not None:
            _lgr.info(f'Maximum/minimum molecular weight ::  {(self.MOLWT.max(),self.MOLWT.min())}')
        _lgr.info(f'Number of gaseous species ::  {(self.NVMR)}')
        gasname = ['']*self.NVMR
        for i in range(self.NVMR):
            gasname1 = gas_info[str(self.ID[i])]['name']
            if self.ISO[i]!=0:
                gasname1 = gasname1+' ('+str(self.ISO[i])+')'
            gasname[i] = gasname1
        _lgr.info(f'Gaseous species ::  {(gasname)}')
        if self.DUST is not None:
            _lgr.info(f'Number of aerosol populations ::  {( self.NDUST)}')
        else:
            _lgr.info(f'Number of aerosol populations ::  {( 0)}')

    ##################################################################################

    def write_hdf5(self,runname,inside_telluric=False):
        """
        Write the Atmosphere properties into an HDF5 file
        """

        import h5py
        from archnemesis.Data.gas_data import gas_info
        from archnemesis.Data.planet_data import planet_info

        #Assessing that all the parameters have the correct type and dimension
        self.assess()

        f = h5py.File(runname+'.h5','a')
        
        if inside_telluric is False:
            
            #Checking if Atmosphere already exists
            if ('/Atmosphere' in f)==True:
                del f['Atmosphere']   #Deleting the Atmosphere information that was previously written in the file

            grp = f.create_group("Atmosphere")
            
        else:
            
            #The Atmosphere class must be inserted inside the Telluric class
            if ('/Telluric/Atmosphere' in f)==True:
                del f['Telluric/Atmosphere']   #Deleting the Atmosphere information that was previously written in the file

            grp = f.create_group("Telluric/Atmosphere")

        #Writing the main dimensions
        dset = grp.create_dataset('NP',data=self.NP)
        dset.attrs['title'] = "Number of vertical points in profiles"

        dset = grp.create_dataset('NVMR',data=self.NVMR)
        dset.attrs['title'] = "Number of gaseous species in atmosphere"

        dset = grp.create_dataset('NLOCATIONS',data=self.NLOCATIONS)
        dset.attrs['title'] = "Number of different vertical profiles in atmosphere"

        dset = grp.create_dataset('NDUST',data=self.NDUST)
        dset.attrs['title'] = "Number of aerosol populations in atmosphere"

        dset = grp.create_dataset('IPLANET',data=self.IPLANET)
        dset.attrs['title'] = "Planet ID"
        dset.attrs['type'] = planet_info[str(int(self.IPLANET))]["name"]

        dset = grp.create_dataset('AMFORM',data=self.AMFORM)
        dset.attrs['title'] = "Type of Molecular Weight calculation"
        if self.AMFORM==AtmosphericProfileFormatEnum.MOLECULAR_WEIGHT_DEFINED:
            dset.attrs['type'] = "Explicit definition of the molecular weight MOLWT"
            dset = grp.create_dataset('MOLWT',data=self.MOLWT)
            dset.attrs['title'] = "Molecular weight"
            dset.attrs['units'] = "kg mol-1"
        elif self.AMFORM==AtmosphericProfileFormatEnum.CALC_MOLECULAR_WEIGHT_SCALE_VMR_TO_ONE:
            dset.attrs['type'] = "Internal calculation of molecular weight with scaling of VMRs to 1"
        elif self.AMFORM==AtmosphericProfileFormatEnum.CALC_MOLECULAR_WEIGHT_DO_NOT_SCALE_VMR:
            dset.attrs['type'] = "Internal calculation of molecular weight without scaling of VMRs"

        dset = grp.create_dataset('ID',data=self.ID)
        dset.attrs['title'] = "ID of the gaseous species"

        dset = grp.create_dataset('ISO',data=self.ISO)
        dset.attrs['title'] = "Isotope ID of the gaseous species"

        dset = grp.create_dataset('LATITUDE',data=self.LATITUDE)
        dset.attrs['title'] = "Latitude of each vertical profile"
        dset.attrs['units'] = "Degrees"

        dset = grp.create_dataset('LONGITUDE',data=self.LONGITUDE)
        dset.attrs['title'] = "Longitude of each vertical profile"
        dset.attrs['units'] = "Degrees"

        dset = grp.create_dataset('P',data=self.P)
        dset.attrs['title'] = "Pressure"
        dset.attrs['units'] = "Pa"

        dset = grp.create_dataset('T',data=self.T)
        dset.attrs['title'] = "Temperature"
        dset.attrs['units'] = "K"

        dset = grp.create_dataset('H',data=self.H)
        dset.attrs['title'] = "Altitude"
        dset.attrs['units'] = "m"

        dset = grp.create_dataset('VMR',data=self.VMR)
        dset.attrs['title'] = "Volume mixing ratio"
        dset.attrs['units'] = ""

        if self.NDUST>0:
            dset = grp.create_dataset('DUST',data=self.DUST)
            dset.attrs['title'] = "Aerosol abundance"
            dset.attrs['units'] = "particles m-3"
            
        if self.PARAH2 is not None:
            dset = grp.create_dataset('PARAH2',data=self.PARAH2)
            dset.attrs['title'] = "Para-H2 fraction"

        f.close()

    ##################################################################################

    def read_hdf5(self,runname,inside_telluric=False):
        """
        Read the Atmosphere properties from an HDF5 file
        """

        import h5py

        f = h5py.File(runname+'.h5','r')
        
        if inside_telluric is True:
            name = '/Telluric/Atmosphere'
        else:
            name = '/Atmosphere'

        #Checking if Atmosphere exists
        e = name in f
        if e==False:
            raise ValueError('error :: Atmosphere is not defined in HDF5 file')
        else:

            self.NP = np.int32(f.get(name+'/NP'))
            self.NLOCATIONS = np.int32(f.get(name+'/NLOCATIONS'))
            self.NVMR = np.int32(f.get(name+'/NVMR'))
            self.NDUST = np.int32(f.get(name+'/NDUST'))
            self.AMFORM : AtmosphericProfileFormatEnum = AtmosphericProfileFormatEnum(np.int32(f.get(name+'/AMFORM')))
            self.IPLANET : PlanetEnum = PlanetEnum(np.int32(f.get(name+'/IPLANET')))

            if self.NLOCATIONS==1:
                self.LATITUDE = np.float64(f.get(name+'/LATITUDE'))
                self.LONGITUDE = np.float64(f.get(name+'/LONGITUDE'))
            else:
                self.LATITUDE = np.array(f.get(name+'/LATITUDE'))
                self.LONGITUDE = np.array(f.get(name+'/LONGITUDE'))

            self.ID = np.array(f.get(name+'/ID'))
            self.ISO = np.array(f.get(name+'/ISO'))

            self.H = np.array(f.get(name+'/H'))
            self.P = np.array(f.get(name+'/P'))
            self.T = np.array(f.get(name+'/T'))
            self.VMR = np.array(f.get(name+'/VMR'))

            if self.NDUST>0:
                self.DUST = np.array(f.get(name+'/DUST'))
                self.DUST_UNITS_FLAG = None  #if reading from the HDF5 files units are assumed to be in number density (m^{-3})
                
            parah2 = name+'/PARAH2'
            if parah2 in f:
                self.PARAH2 = np.array(f.get(name+'/PARAH2'))

            if self.AMFORM==AtmosphericProfileFormatEnum.MOLECULAR_WEIGHT_DEFINED:
                self.MOLWT = np.array(f.get(name+'/MOLWT'))
            if self.AMFORM in (
                    AtmosphericProfileFormatEnum.CALC_MOLECULAR_WEIGHT_SCALE_VMR_TO_ONE, 
                    AtmosphericProfileFormatEnum.CALC_MOLECULAR_WEIGHT_DO_NOT_SCALE_VMR
                ):
                self.calc_molwt()

            self.calc_grav()   

            self.assess()  

        f.close()       

    ##################################################################################

    def ipar_to_atm_profile_type(
            self, 
            ipar : int
        ) -> tuple[AtmosphericProfileType, None|int]:
        """
            Decodes `ipar` from a magic number to a profile type and an index of that profile type
            
            ## ARGUMENTS ##
            
                ipar :: Atmospheric parameter to be changed
                    (0 to NVMR-1) :: Gas VMR
                    (NVMR) :: Temperature
                    (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                    (NVMR+NDUST) :: Para-H2
                    (NVMR+NDUST+1) :: Fractional cloud coverage
            
            ## RETURNS ##
            
                atm_profile_type : AtmosphericProfileType
                    An ENUM specifiying the type of the profile.
                
                atm_profile_idx : int | None
                    An integer denoting which profile of `atm_profile_type` to select, or `None`
                    if that profile type cannot have multiple entries.
            
            ## EXAMPLE ##
            
                (atm_profile_type, atm_profile_idx) = Atmosphere.ipar_to_atm_profile_type(ipar)
                
                
        """
        _lgr.debug(f'{ipar=}')
        _lgr.debug(f'{self.NVMR=} {self.NDUST=}')
        if ipar >=0 and ipar < self.NVMR:
            return AtmosphericProfileType.GAS_VOLUME_MIXING_RATIO, ipar
        
        if ipar == self.NVMR:
            return AtmosphericProfileType.TEMPERATURE, 0
        
        if ipar > self.NVMR and ipar <= self.NVMR+self.NDUST:
            return AtmosphericProfileType.AEROSOL_DENSITY, ipar - (self.NVMR+1)
        
        if ipar == self.NVMR+self.NDUST+1:
            return AtmosphericProfileType.PARA_H2_FRACTION, None # only ever one of these profiles
        
        if ipar == self.NVMR+self.NDUST+2:
            return AtmosphericProfileType.FRACTIONAL_CLOUD_COVERAGE, None # only ever one of these profiles
        
        raise ValueError(f'Atmosphere_0 :: ipar_to_atm_profile_type :: {ipar=} is not a supported value')

    def edit_H(self, array):
        """
        Edit the Height profile.
        @param H_array: 1D or 2D array
            Heights of the vertical points in m
        """
        array = np.array(array)

        if self.NLOCATIONS==1:
            assert len(array) == self.NP, 'H should have NP elements'
            assert ((array[1:]-array[:-1])>0).all(),\
                'H should be strictly increasing'
        elif self.NLOCATIONS>1:
            assert array.shape == (self.NP,self.NLOCATIONS), 'H should have (NP,NLOCATIONS) elements'
        
        self.H = array

    ##################################################################################

    def edit_P(self, array):
        """
        Edit the Pressure profile.
        @param P_array: 1D or 2D array
            Pressures of the vertical points in Pa
        """
        array = np.array(array)

        if self.NLOCATIONS==1:
            assert len(array) == self.NP, 'P should have NP elements'
        elif self.NLOCATIONS>1:
            assert array.shape == (self.NP,self.NLOCATIONS), 'P should have (NP,NLOCATIONS) elements'
        
        self.P = array

    ##################################################################################

    def edit_T(self, array):
        """
        Edit the Temperature profile.
        @param T_array: 1D or 2D array
            Temperature of the vertical points in K
        """
        array = np.array(array)

        if self.NLOCATIONS==1:
            assert len(array) == self.NP, 'T should have NP elements'
        elif self.NLOCATIONS>1:
            assert array.shape == (self.NP,self.NLOCATIONS), 'T should have (NP,NLOCATIONS) elements'
        
        self.T = array

    ##################################################################################

    def edit_VMR(self, array):
        """
        Edit the gas Volume Mixing Ratio profile.
        @param VMR_array: 2D or 3D array
            NP by NVMR array containing the Volume Mixing Ratio of gases.
            VMR_array[i,j] is the volume mixing ratio of gas j at point i.
        """
        array = np.array(array)

        if self.NLOCATIONS==1:
            assert array.shape == (self.NP,self.NVMR), 'VMR should have (NP,NVMR) elements'
        elif self.NLOCATIONS>1:
            assert array.shape == (self.NP,self.NVMR,self.NLOCATIONS), 'VMR should have (NP,NVMR,NLOCATIONS) elements'
        
        self.VMR = array

    ##################################################################################

    def edit_DUST(self, array):
        """
        Edit the aerosol abundance profile.
        @param DUST_array: 2D or 3D array
            Dust abundance in particles m-3. Array size must be (NP,NDUST) or (NP,NDUST,NLOCATIONS)
        """
        array = np.array(array)

        if self.NLOCATIONS==1:
            assert array.shape == (self.NP,self.NDUST), 'DUST should have (NP,NDUST) elements'
        elif self.NLOCATIONS>1:
            assert array.shape == (self.NP,self.NDUST,self.NLOCATIONS), 'DUST should have (NP,NDUST,NLOCATIONS) elements'
        
        self.DUST = array

    ##################################################################################

    def adjust_VMR(self, ISCALE=[1,1,1,1,1,1]):

        """
        Subroutine to adjust the vmrs at a particular level to add up to 1.0.
        Also limits vmrs to svp if specified in .vpf.
        ISCALE :: Flag to indicate if gas vmr can be scaled(1) or not (0).
        """

        ISCALE = np.array(ISCALE)
        jvmr1 = np.where(ISCALE==1)
        jvmr2 = np.where(ISCALE==0)

        if self.NLOCATIONS==1:

            vmr = np.zeros([self.NP,self.NVMR])
            vmr[:,:] = self.VMR
            for ipro in range(self.NP):

                sumtot = np.sum(self.VMR[ipro,:])
                sum1 = np.sum(self.VMR[ipro,jvmr2])

                if sumtot!=1.0:
                    #Need to adjust the VMRs of those gases that can be scaled to
                    #bring the total sum to 1.0
                    xfac = (1.0-sum1)/(sumtot-sum1)
                    vmr[ipro,jvmr1] = self.VMR[ipro,jvmr1] * xfac
                    #self.VMR[ipro,jvmr1] = self.VMR[ipro,jvmr1] * xfac
            
            self.edit_VMR(vmr)

        elif self.NLOCATIONS>1:

            vmr = np.zeros((self.NP,self.NVMR,self.NLOCATIONS))
            vmr[:,:,:] = self.VMR
            for iloc in range(self.NLOCATIONS):
                for ipro in range(self.NP):

                    sumtot = np.sum(self.VMR[ipro,:,iloc])
                    sum1 = np.sum(self.VMR[ipro,jvmr2,iloc])

                    if sumtot!=1.0:
                        #Need to adjust the VMRs of those gases that can be scaled to
                        #bring the total sum to 1.0
                        xfac = (1.0-sum1)/(sumtot-sum1)
                        vmr[ipro,jvmr1,iloc] = self.VMR[ipro,jvmr1,iloc] * xfac
                        #self.VMR[ipro,jvmr1] = self.VMR[ipro,jvmr1] * xfac

            self.edit_VMR(vmr)
            
            
        for i in range(len(self.ID)):
            if (self.ID[i],self.ISO[i]) in self.SVP.keys():
                try:
                    vp,svpflag = self.SVP[(self.ID[i],self.ISO[i])]
                    a,b,c,d = Data.gas_data.svp_coefficients[self.ID[i]]
                except:
                    raise ValueError(f'error :: Could not find saturation vapour coefficients for gas {self.ID[i]}')
                
                svp = vp*np.exp(a + b/self.T + c*self.T + d*self.T**2)
                pp = self.VMR[:,i]*self.P/101325
                self.VMR[:,i] = np.where(pp > svp, svp/(self.P/101325), self.VMR[:,i])
            

    ##################################################################################

    def calc_molwt(self):
        """
        Subroutine to calculate the molecular weight of the atmosphere (kg/mol)
        """
        
        from archnemesis.Data.gas_data import gas_info

        if self.NLOCATIONS==1:

            molwt = np.zeros(self.NP)
            vmrtot = np.zeros(self.NP)
            for i in range(self.NVMR):
                if self.ISO[i]==0:
                    molwt1 = gas_info[str(self.ID[i])]['mmw']
                else:
                    molwt1 = gas_info[str(self.ID[i])]['isotope'][str(self.ISO[i])]['mass']

                vmrtot[:] = vmrtot[:] + self.VMR[:,i]
                molwt[:] = molwt[:] + self.VMR[:,i] * molwt1

        elif self.NLOCATIONS>1:

            molwt1 = np.zeros(self.NVMR)
            for i in range(self.NVMR):
                if self.ISO[i]==0:
                    molwt1[i] = gas_info[str(self.ID[i])]['mmw']
                else:
                    molwt1[i] = gas_info[str(self.ID[i])]['isotope'][str(self.ISO[i])]['mass']

            molwt = np.zeros((self.NP,self.NLOCATIONS))
            vmrtot = np.zeros((self.NP,self.NLOCATIONS))

            for i in range(self.NVMR):
                vmrtot[:,:] = vmrtot[:,:] + self.VMR[:,i,:]
                molwt[:,:] = molwt[:,:] + self.VMR[:,i,:] * molwt1[i]

        molwt = molwt / vmrtot
        self.MOLWT = molwt / 1000.

    ##################################################################################

    def calc_rho(self):
        """
        Subroutine to calculate the atmospheric density (kg/m3) at each level
        """
        
        from archnemesis.Data.gas_data import const
        
        R = const["R"]
        rho = self.P * self.MOLWT / R / self.T

        return rho
    
    ##################################################################################
    
    def calc_numdens(self):
        """
        Subroutine to calculate the atmospheric number density (m-3) at each level
        """
        
        from archnemesis.Data.gas_data import const
        
        k_B = const["k_B"]
        numdens = self.P / k_B / self.T

        return numdens

    ##################################################################################

    def calc_radius(self):
        """
        Subroutine to calculate the radius of the planet at the required latitude
        """
        
        from archnemesis.Data.planet_data import planet_info

        #Getting the information about the planet
        data = planet_info[str(int(self.IPLANET))]
        xradius = data["radius"] * 1.0e5   #cm
        xellip=1.0/(1.0-data["flatten"])

        #Calculating some values to account for the latitude dependence
        lat = 2 * np.pi * self.LATITUDE/360.      #Latitude in rad
        latc = np.arctan(np.tan(lat)/xellip**2.)   #Converts planetographic latitude to planetocentric
        slatc = np.sin(latc)
        clatc = np.cos(latc)
        Rr = np.sqrt(clatc**2 + (xellip**2. * slatc**2.))  #ratio of radius at equator to radius at current latitude
        radius = (xradius/Rr)*1.0e-5     #Radius of the planet at the given distance (km)

        self.RADIUS = radius * 1.0e3     #Metres

    ##################################################################################

    def calc_grav(self):
        """
        Subroutine to calculate the gravity at each level following the method
        of Lindal et al., 1986, Astr. J., 90 (6), 1136-1146
        """

        from archnemesis.Data.gas_data import const
        from archnemesis.Data.planet_data import planet_info

        #Reading data and calculating some parameters
        Grav = const["G"]
        data = planet_info[str(int(self.IPLANET))]
        xgm = data["mass"] * Grav * 1.0e24 * 1.0e6
        xomega = 2.*np.pi / (data["rotation"]*24.*3600.)
        xellip=1.0/(1.0-data["flatten"])
        Jcoeff = data["Jcoeff"]
        xcoeff = np.zeros(3)
        xcoeff[0] = Jcoeff[0] / 1.0e3
        xcoeff[1] = Jcoeff[1] / 1.0e6
        xcoeff[2] = Jcoeff[2] / 1.0e8
        xradius = data["radius"] * 1.0e5   #cm
        isurf = data["isurf"]
        name = data["name"]


        #Calculating some values to account for the latitude dependence
        lat = 2 * np.pi * self.LATITUDE/360.      #Latitude in rad [float or (NLOCATIONS)]
        latc = np.arctan(np.tan(lat)/xellip**2.)   #Converts planetographic latitude to planetocentric
        slatc = np.sin(latc)
        clatc = np.cos(latc)
        Rr = np.sqrt(clatc**2 + (xellip**2. * slatc**2.))  #ratio of radius at equator to radius at current latitude [float or (NLOCATIONS)]
        r = (xradius+self.H*1.0e2)/Rr    #Radial distance of each altitude point to centre of planet (cm) [(NP) or (NP,NLOCATIONS)]
        radius = (xradius/Rr)*1.0e-5     #Radius of the planet at the given distance (km)  [float or (NLOCATIONS)]

        self.RADIUS = radius * 1.0e3    

        #Calculating Legendre polynomials
        pol = np.zeros((6,self.NLOCATIONS))
        for i in range(6):
            Pn = legendre(i+1)
            pol[i,:] = Pn([slatc])

        #Evaluate radial contribution from summation
        # for first three terms,
        #then subtract centrifugal effect.
        g = np.ones(self.NLOCATIONS)
        for i in range(3):
            ix = i + 1
            g[:] = g[:] - ((2*ix+1) * Rr**(2 * ix) * xcoeff[ix-1] * pol[2*ix-1,:])

        #gradial = np.zeros((self.NP,self.NLOCATIONS))
        gradial = (g * xgm/r**2.) - (r * xomega**2. * clatc**2.)

        #Evaluate latitudinal contribution for
        # first three terms, then add centrifugal effects

        gtheta1 = np.zeros(self.NLOCATIONS)
        for i in range(3):
            ix = i + 1
            gtheta1 = gtheta1 - (4. * ix**2 * Rr**(2 * ix) * xcoeff[ix-1] * (pol[2*ix-1-1] - slatc * pol[2*ix-1])/clatc)

        gtheta = (gtheta1 * xgm/r**2) + (r * xomega**2 * clatc * slatc)

        #Combine the two components and write the result
        gtot = np.sqrt(gradial**2. + gtheta**2.)*0.01   #m/s2

        self.GRAV = gtot

    ##################################################################################

    def adjust_hydrostatP(self,htan,ptan):
        """
        Subroutine to rescale the pressures of a H/P/T profile according to
        the hydrostatic equation above and below a specified altitude
        given the pressure at that altitude
            htan :: specified altitude (m)
            ptan :: Pressure at specified altitude (Pa)


        Note :: Only valid if NLOCATIONS = 1
            
        """
        
        from archnemesis.Data.gas_data import const

        #if self.NLOCATIONS>1:
        #    raise ValueError('error :: adjust_hydrostatP only works if NLOCATIONS = 1')

        if self.NLOCATIONS==1:

            #First find the level below the reference altitude
            ialt = np.argmin(np.abs(self.H-htan))
            alt0 = self.H[ialt]
            if ( (alt0>htan) & (ialt>0)):
                ialt = ialt -1

            #Calculating the gravity at each altitude level
            self.calc_grav()

            #Calculate the scaling factor
            R = const["R"]
            scale = R * self.T / (self.MOLWT * self.GRAV)   #scale height (m)

            sh =  0.5*(scale[ialt]+scale[ialt+1])
            delh = self.H[ialt+1]-htan
            p = np.zeros(self.NP)
            p[ialt+1] = ptan*np.exp(-delh/sh)
            delh = self.H[ialt]-htan
            p[ialt] = ptan*np.exp(-delh/sh)

            for i in range(ialt+2,self.NP):
                sh =  0.5*(scale[i-1]+scale[i])
                delh = self.H[i]-self.H[i-1]
                p[i] = p[i-1]*np.exp(-delh/sh)

            for i in range(ialt-1,-1,-1):
                sh =  0.5*(scale[i+1]+scale[i])
                delh = self.H[i]-self.H[i+1]
                p[i] = p[i+1]*np.exp(-delh/sh)

            self.edit_P(p)
            
        else:
            
            #Checking that htan and ptan are arrays with size NLOCATIONS
            if len(htan)!=self.NLOCATIONS:
                raise ValueError('error in adjus_thydrostatP :: htan must have NLOCATION elements')

            if len(ptan)!=self.NLOCATIONS:
                raise ValueError('error in adjus_thydrostatP :: ptan must have NLOCATION elements')
            
            for iLOC in range(self.NLOCATIONS):
            
                #First find the level below the reference altitude
                ialt = np.argmin(np.abs(self.H[:,iLOC]-htan[iLOC]))
                alt0 = self.H[ialt,iLOC]
                if ( (alt0>htan[iLOC]) & (ialt>0)):
                    ialt = ialt -1
                
                #Calculating the gravity at each altitude level
                self.calc_grav()

                #Calculate the scaling factor
                R = const["R"]
                scale = R * self.T[:,iLOC] / (self.MOLWT[:,iLOC] * self.GRAV[:,iLOC])   #scale height (m)

                sh =  0.5*(scale[ialt]+scale[ialt+1])
                delh = self.H[ialt+1,iLOC]-htan[iLOC]
                p = np.zeros(self.NP)
                p[ialt+1] = ptan[iLOC]*np.exp(-delh/sh)
                delh = self.H[ialt,iLOC]-htan[iLOC]
                p[ialt] = ptan[iLOC]*np.exp(-delh/sh)

                for i in range(ialt+2,self.NP):
                    sh =  0.5*(scale[i-1]+scale[i])
                    delh = self.H[i,iLOC]-self.H[i-1,iLOC]
                    p[i] = p[i-1]*np.exp(-delh/sh)

                for i in range(ialt-1,-1,-1):
                    sh =  0.5*(scale[i+1]+scale[i])
                    delh = self.H[i,iLOC]-self.H[i+1,iLOC]
                    p[i] = p[i+1]*np.exp(-delh/sh)

                #self.edit_P(p)
                self.P[:,iLOC] = p[:]

    ##################################################################################

    def adjust_hydrostatH(self):
        """
        Subroutine to rescale the heights of a H/P/T profile according to
        the hydrostatic equation above and below the level where height=0.

        Note : Only valid if NLOCATIONS = 1
        """

        from archnemesis.Data.gas_data import gas_info, const

        if self.NLOCATIONS==1:

            #First find the level closest to the 0m altitudef
            ialt = np.argmin(np.abs(self.H-0.0))
            alt0 = self.H[ialt]
            if ( (alt0>0.0) & (ialt>0)):
                ialt = ialt

            xdepth = 100.
            while xdepth>1:

                h = np.zeros(self.NP)
                p = np.zeros(self.NP)
                h[:] = self.H
                p[:] = self.P

                #Calculating the atmospheric depth
                atdepth = h[self.NP-1] - h[0]

                #Calculate the gravity at each altitude level
                self.calc_grav()
                #Calculate the scale height
                R = const["R"]
                scale = R * self.T / (self.MOLWT * self.GRAV)   #scale height (m)

                p[:] = self.P
                if ((ialt>0) & (ialt<self.NP-1)):
                    h[ialt] = 0.0

                nupper = self.NP - ialt - 1
                for i in range(ialt+1,self.NP):
                    sh = 0.5 * (scale[i-1] + scale[i])
                    #self.H[i] = self.H[i-1] - sh * np.log(self.P[i]/self.P[i-1])
                    h[i] = h[i-1] - sh * np.log(p[i]/p[i-1])

                for i in range(ialt-1,-1,-1):
                    sh = 0.5 * (scale[i+1] + scale[i])
                    #self.H[i] = self.H[i+1] - sh * np.log(self.P[i]/self.P[i+1])
                    h[i] = h[i+1] - sh * np.log(p[i]/p[i+1])

                #atdepth1 = self.H[self.NP-1] - self.H[0]
                atdepth1 = h[self.NP-1] - h[0]

                xdepth = 100.*abs((atdepth1-atdepth)/atdepth)

                self.H = h[:]

                #Re-Calculate the gravity at each altitude level
                self.calc_grav()
                
        else:
            
            
            #It is assumed that the 0.0 altitude level is at the level position in all profiles
            ialtx = np.argmin(np.abs(self.H),axis=0)
            ialt = np.unique(ialtx)
            
            if len(ialt)!=1:
                raise ValueError('error in adjust_hydrostatH :: when using multiple locations it is assumed that the z = 0.0 km is at the same level index')
            else:
                ialt = int(ialt[0])

            altx = self.H[ialt,:]
            ialtx = np.zeros(self.NLOCATIONS,dtype='int32')
            for iLOC in range(self.NLOCATIONS):
                if((altx[iLOC]>0.0) & (ialt>0)):
                    ialtx[iLOC] = ialt -1
                    
            ialt = np.unique(ialtx)
            if len(ialt)!=1:
                raise ValueError('error in adjust_hydrostatH :: when using multiple locations it is assumed that the z = 0.0 km is at the same level index')
            else:
                ialt = int(ialt[0])


            xdepth = np.ones(self.NLOCATIONS)*100.
            while xdepth.max()>1.0:
                
                h = np.zeros(self.H.shape)
                p = np.zeros(self.H.shape)
                h[:,:] = self.H[:,:]
                p[:,:] = self.P[:,:]
            
                #Calculating the atmospheric depth
                atdepth = self.H[self.NP-1,:] - self.H[0,:]

                #Calculate the gravity at each altitude level
                self.calc_grav()

                #Calculate the scale height
                R = const["R"]
                scale = R * self.T / (self.MOLWT * self.GRAV)   #scale height (m)

                if ((ialt>0) & (ialt<self.NP-1)):
                    h[ialt,:] = 0.0

                nupper = self.NP - ialt - 1
                for i in range(ialt+1,self.NP):
                    sh = 0.5 * (scale[i-1,:] + scale[i,:])
                    h[i,:] = h[i-1,:] - sh[:] * np.log(p[i,:]/p[i-1,:])

                for i in range(ialt-1,-1,-1):
                    sh = 0.5 * (scale[i+1,:] + scale[i,:])
                    h[i,:] = h[i+1,:] - sh[:] * np.log(p[i,:]/p[i+1,:])

                atdepth1 = h[self.NP-1,:] - h[0,:]

                xdepth = 100.*abs((atdepth1-atdepth)/atdepth)

                self.H[:,:] = h[:,:]

                #Re-Calculate the gravity at each altitude level
                self.calc_grav()

    ##################################################################################

    def locate_gas(self,gasID,isoID):
        """
        Subroutine to locate a gas in the reference atmosphere
            gasID :: Radtran ID of the gas
            isoID :: Radtran isotopologue ID of the gas
        """

        for i, (id_val, iso_val) in enumerate(zip(self.ID, self.ISO)):
            if id_val == gasID and iso_val == isoID:
                return i
        return None

    ##################################################################################

    def add_gas(self,gasID,isoID,vmr):
        """
        Subroutine to add a gas into the reference atmosphere
            gasID :: Radtran ID of the gas
            isoID :: Radtran isotopologue ID of the gas
            vmr(NP) :: Volume mixing ratio of the gas at each altitude level (alternatively it can be (NP,NLOCATIONS))
        """

        ngas = self.NVMR + 1

        if self.NLOCATIONS==1:

            if len(vmr)!=self.NP:
                raise ValueError('error in Atmosphere.add_gas() :: Number of altitude levels in vmr must be the same as in Atmosphere')
            else:
                vmr1 = np.zeros([self.NP,ngas])
                gasID1 = np.zeros(ngas,dtype='int32')
                isoID1 = np.zeros(ngas,dtype='int32')
                vmr1[:,0:self.NVMR] = self.VMR
                vmr1[:,ngas-1] = vmr[:]
                gasID1[0:self.NVMR] = self.ID
                isoID1[0:self.NVMR] = self.ISO
                gasID1[ngas-1] = gasID
                isoID1[ngas-1] = isoID
                self.NVMR = ngas
                self.ID = gasID1
                self.ISO = isoID1
                self.edit_VMR(vmr1)

        elif self.NLOCATIONS>1:

            if vmr.shape!=(self.NP,self.NLOCATIONS):
                raise ValueError('error in Atmosphere.add_gas() :: vmr must have size (NP,NLOCATIONS)')
            else:
                vmr1 = np.zeros([self.NP,ngas,self.NLOCATIONS])
                gasID1 = np.zeros(ngas,dtype='int32')
                isoID1 = np.zeros(ngas,dtype='int32')
                vmr1[:,0:self.NVMR,:] = self.VMR[:,:,:]
                vmr1[:,ngas-1,:] = vmr[:,:]
                gasID1[0:self.NVMR] = self.ID
                isoID1[0:self.NVMR] = self.ISO
                gasID1[ngas-1] = gasID
                isoID1[ngas-1] = isoID
                self.NVMR = ngas
                self.ID = gasID1
                self.ISO = isoID1
                self.edit_VMR(vmr1)

    ##################################################################################

    def remove_gas(self,gasID,isoID):
        """
        Subroutine to remove a gas from the reference atmosphere
            gasID :: Radtran ID of the gas
            isoID :: Radtran isotopologue ID of the gas
        """

        igas = np.where( (self.ID==gasID) & (self.ISO==isoID) )
        igas = igas[0]

        if len(igas)>1:
            _lgr.warning(' in Atmosphere.remove_gas() :: Two gases with same Gas ID and Iso ID. Removing the second one by default')
            igas = igas[1]

        if len(igas)==0:
            _lgr.info('error in Atmosphere.remove_gas() :: Gas ID and Iso ID not found in reference atmosphere')
        else:

            self.NVMR = self.NVMR - 1
            self.ID = np.delete(self.ID,igas,axis=0)
            self.ISO = np.delete(self.ISO,igas,axis=0)
            self.edit_VMR(np.delete(self.VMR,igas,axis=1))

    ##################################################################################

    def update_gas(self,gasID,isoID,vmr):
        """
        Subroutine to update a gas into the reference atmosphere
            gasID :: Radtran ID of the gas
            isoID :: Radtran isotopologue ID of the gas
            vmr(NP) :: Volume mixing ratio of the gas at each altitude level (or (NP,NLOCATIONS))
        """

        igas = np.where( (self.ID==gasID) & (self.ISO==isoID) )
        igas = igas[0]


        if len(vmr)!=self.NP:
            raise ValueError('error in Atmosphere.update_gas() :: Number of altitude levels in vmr must be the same as in Atmosphere')

        if len(igas)==0:
            raise ValueError('error in Atmosphere.update_gas() :: Gas ID and Iso ID not found in reference atmosphere')
        else:
            if self.NLOCATIONS==1:
                vmr1 = np.zeros((self.NP,self.NVMR))
                vmr1[:,:] = self.VMR
                vmr1[:,igas[0]] = vmr[:]
                self.edit_VMR(vmr1)
            elif self.NLOCATIONS>1:
                vmr1 = np.zeros((self.NP,self.NVMR,self.NLOCATIONS))
                vmr1[:,:,:] = self.VMR
                vmr1[:,igas[0],:] = vmr[:,:]
                self.edit_VMR(vmr1)

    ##################################################################################

    def normalise_dust(self,idust):
        """
        Subroutine to normalise the column of a given aerosol population so that the column optical
        depth is given by the values in Scatter.KEXT
        
        This function is very useful if wanting to work with optical depths rather than with aerosol density.
        
        If normalising the aerosol profile with this function, the column optical depth for the aerosol population
        will be given by the value in Scatter.KEXT. 
    
        Inputs
        ______
        
        idust :: Index of the aerosol population to normalise
        """

        from scipy import integrate

        if self.NLOCATIONS==1:

            #We assume that what is in DUST is in particles m-3, and we integrate it over altitude to get dust column in particles m-2
            dust_col = integrate.simpson(self.DUST[:,idust], x=self.H) 

            #Normalising the aerosol profile so that the column is 1 particles m-2
            self.DUST[:,idust] /= dust_col
            
            #Applying a factor of 1.0e4 because the values in Scatter.KEXT are in cm2
            self.DUST[:,idust] *= 1.0e4
            
            #This way we will have that the column optical depth of this aerosol population is given by Scatter.KEXT 

        else:
            
            for ilocation in range(self.NLOCATIONS):
            
                #We assume that what is in DUST is in particles m-3, and we integrate it over altitude to get dust column in particles m-2
                dust_col = integrate.simpson(self.DUST[:,idust,ilocation], x=self.H[:,ilocation]) 

                #Normalising the aerosol profile so that the column is 1 particles m-2
                self.DUST[:,idust,ilocation] /= dust_col
                
                #Applying a factor of 1.0e4 because the values in Scatter.KEXT are in cm2
                self.DUST[:,idust,ilocation] *= 1.0e4
                
                #This way we will have that the column optical depth of this aerosol population is given by Scatter.KEXT 

    ##################################################################################

    def select_location(self,iLOCATION):
        """
        Subroutine to select only one geometry from the Atmosphere class (and remove all the others)
        """
        
        if iLOCATION>self.NLOCATIONS-1:
            raise ValueError('error in select_location :: iLOCATION must be between 0 and NLOCATIONS-1',[0,self.NLOCATIONS-1])

        self.NLOCATIONS = 1
        self.edit_P(self.P[:,iLOCATION])
        self.edit_T(self.T[:,iLOCATION])
        self.edit_H(self.H[:,iLOCATION])
        self.edit_VMR(self.VMR[:,:,iLOCATION])
        
        self.LATITUDE = self.LATITUDE[iLOCATION]
        self.LONGITUDE = self.LONGITUDE[iLOCATION]
        
        if self.NDUST>0:
            self.edit_DUST(self.DUST[:,:,iLOCATION])
        
        if self.AMFORM in (
                AtmosphericProfileFormatEnum.CALC_MOLECULAR_WEIGHT_SCALE_VMR_TO_ONE, 
                AtmosphericProfileFormatEnum.CALC_MOLECULAR_WEIGHT_DO_NOT_SCALE_VMR
            ):
            self.calc_molwt()
            
        self.calc_grav()
        self.calc_radius()
        
        self.assess()
        
    ##################################################################################

    def read_ref(self):
        """
        Fills the parameters of the Atmospheric class by reading the .ref file
        """


        #Checking if there are lines starting with #
        with open(self.runname+'.ref', 'r') as file:
            
            i0 = 0
            for line in file:
                # Skip lines starting with #
                if line.startswith('#'):
                    i0 = i0 + 1
                else:
                    break
                
        
        #Opening file
        f = open(self.runname+'.ref','r')
        
        #Skipping all lines starting with #
        for i in range(i0):
            header = f.readline()
         
        #Reading first and second lines
        tmp = np.fromfile(f,sep=' ',count=1,dtype='int')
        amform : AtmosphericProfileFormatEnum = AtmosphericProfileFormatEnum(int(tmp[0]))
        tmp = np.fromfile(f,sep=' ',count=1,dtype='int')

        #Reading third line
        tmp = f.readline().split()
        nplanet : PlanetEnum = PlanetEnum(int(tmp[0]))
        xlat = float(tmp[1])
        npro = int(tmp[2])
        ngas = int(tmp[3])
        if amform==AtmosphericProfileFormatEnum.MOLECULAR_WEIGHT_DEFINED:
            molwt = float(tmp[4])

        #Reading gases
        gasID = np.zeros(ngas,dtype='int')
        isoID = np.zeros(ngas,dtype='int')
        for i in range(ngas):
            tmp = np.fromfile(f,sep=' ',count=2,dtype='int')
            gasID[i] = int(tmp[0])
            isoID[i] = int(tmp[1])

        #Reading profiles
        height = np.zeros(npro)
        press = np.zeros(npro)
        temp = np.zeros(npro)
        vmr = np.zeros((npro,ngas))
        s = f.readline().split()
        for i in range(npro):
            tmp = np.fromfile(f,sep=' ',count=ngas+3,dtype='float')
            height[i] = float(tmp[0])
            press[i] = float(tmp[1])
            temp[i] = float(tmp[2])
            for j in range(ngas):
                vmr[i,j] = float(tmp[3+j])

        #Storing the results into the atmospheric class
        self.NP = npro
        self.NVMR = ngas
        self.ID = gasID
        self.ISO = isoID
        self.IPLANET : PlanetEnum = nplanet
        self.LATITUDE = xlat
        self.LONGITUDE = 0.0
        self.AMFORM : AtmosphericProfileFormatEnum = amform
        self.NLOCATIONS = 1
        self.edit_H(height*1.0e3)
        self.edit_P(press*101325.)
        self.edit_T(temp)
        self.edit_VMR(vmr)

        if self.AMFORM in (
                AtmosphericProfileFormatEnum.CALC_MOLECULAR_WEIGHT_SCALE_VMR_TO_ONE,
                AtmosphericProfileFormatEnum.CALC_MOLECULAR_WEIGHT_DO_NOT_SCALE_VMR
            ):
            self.calc_molwt()
        elif self.AMFORM == AtmosphericProfileFormatEnum.MOLECULAR_WEIGHT_DEFINED:
            molwt1 = np.zeros(npro)
            molwt1[:] = molwt
            self.MOLWT = molwt1 / 1000.   #kg/mole
        else:
            raise ValueError(f'Atmosphere_0.AMFORM must be one of {tuple(AtmosphericProfileFormatEnum)}')

        self.calc_grav()

    ##################################################################################

    def write_ref(self):
        """
        Write the current atmospheric profiles into the .ref file
        """

        if self.NLOCATIONS>1:
            raise ValueError('error :: write_ref only works if NLOCATIONS=1')

        fref = open(self.runname+'.ref','w')
        fref.write('\t %i \n' % (self.AMFORM))
        nlat = 1    #Would need to be updated to include more latitudes
        fref.write('\t %i \n' % (nlat))

        if self.AMFORM==AtmosphericProfileFormatEnum.MOLECULAR_WEIGHT_DEFINED:
            fref.write('\t %i \t %7.4f \t %i \t %i \t %7.4f \n' % (self.IPLANET,self.LATITUDE,self.NP,self.NVMR,self.MOLWT[0]*1.0e3))
        elif self.AMFORM in (
                AtmosphericProfileFormatEnum.CALC_MOLECULAR_WEIGHT_SCALE_VMR_TO_ONE,
                AtmosphericProfileFormatEnum.CALC_MOLECULAR_WEIGHT_DO_NOT_SCALE_VMR
            ):
            fref.write('\t %i \t %7.4f \t %i \t %i \n' % (self.IPLANET,self.LATITUDE,self.NP,self.NVMR))
        else:
            raise ValueError(f'Atmosphere_0.AMFORM must be one of {tuple(AtmosphericProfileFormatEnum)}')
            
        gasname = [''] * self.NVMR
        header = [''] * (3+self.NVMR)
        header[0] = 'height(km)'
        header[1] = 'press(atm)'
        header[2] = 'temp(K)  '
        str1 = header[0]+'\t'+header[1]+'\t'+header[2]
        for i in range(self.NVMR):
            fref.write('\t %i \t %i\n' % (self.ID[i],self.ISO[i]))
            strgas = 'GAS'+str(i+1)+'_vmr'
            str1 = str1+'\t'+strgas

        fref.write(str1+'\n')

        for i in range(self.NP):
            str1 = str('{0:7.3f}'.format(self.H[i]/1.0e3))+'\t'+str('{0:7.6e}'.format(self.P[i]/101325.))+'\t'+str('{0:7.4f}'.format(self.T[i]))
            for j in range(self.NVMR):
                str1 = str1+'\t'+str('{0:7.6e}'.format(self.VMR[i,j]))
            fref.write(str1+'\n')

        fref.close()

    ##################################################################################

    def read_aerosol(self):
        """
        Read the aerosol profiles from an aerosol.ref file
        
        Note: The units of the aerosol.ref file in NEMESIS are in particles per gram of atmosphere, while the units of the aerosols
              in the Atmosphere class are in particles per m3. Therefore, when reading the file an internal unit conversion is
              applied, but it requires the pressure and temperature profiles to be defined prior to reading the aerosol.ref file.
        """

        if self.NLOCATIONS!=1:
            raise ValueError('error :: read_aerosol only works if NLOCATIONS=1')
            
        #Checking if there are lines starting with #
        with open('aerosol.ref', 'r') as file:
            
            i0 = 0
            for line in file:
                # Skip lines starting with #
                if line.startswith('#'):
                    i0 = i0 + 1
                else:
                    break
                
        #Opening file
        f = open('aerosol.ref','r')

        #Reading header
        for i in range(i0):
            header = f.readline()

        #Reading first line
        tmp = np.fromfile(f,sep=' ',count=2,dtype='int')
        npro = tmp[0]
        naero = tmp[1]

        #Reading data
        height = np.zeros(npro)
        aerodens = np.zeros((npro,naero))
        for i in range(npro):
            tmp = np.fromfile(f,sep=' ',count=naero+1,dtype='float')
            height[i] = tmp[0]
            for j in range(naero):
                aerodens[i,j] = tmp[j+1]  #particle per gram of atm

        #Storing the results into the atmospheric class
        if self.NP==None:
            self.NP = npro
        else:
            if self.NP!=npro:
                raise ValueError('Number of altitude points in aerosol.ref must be equal to NP')

        #Filling the information into the class
        self.NP = npro
        self.NDUST = naero
        if self.H is None:
            self.edit_H(height*1.0e3)   #m

        self.DUST_UNITS_FLAG = -1*np.ones(self.NDUST)  #if reading from the NEMESIS files units are assumed to be particles per gram of atm
        self.edit_DUST(aerodens)    #particles per gram of atm

    ##################################################################################

    def write_aerosol(self):
        """
        Write current aerosol profile to a aerosol.ref file in Nemesis format.
        
        Note: The units of the aerosol.ref file in NEMESIS are in particles per gram of atmosphere, while the units of the aerosols
              in the Atmosphere class are in particles per m3. Therefore, when writing the file an internal unit conversion is
              applied, but it requires the pressure and temperature profiles to be defined prior to writing the aerosol.ref file.
        """

        if self.NLOCATIONS!=1:
            raise ValueError('error :: read_aerosol only works if NLOCATIONS=1')
            
        #Check if the density can be calculated
        if((self.T is not None) & (self.P is not None)):
            rho = self.calc_rho()  #kg/m3
            xscale = rho * 1000.
        else:
            xscale = 1.
            _lgr.warning(' :: reading aerosol.ref file but density is not define. Units of Atmosphere_0.DUST are in particles per gram of atmosphere')
            

        f = open('aerosol.ref','w')
        f.write('#aerosol.ref\n')
        f.write('{:<15} {:<15}'.format(self.NP, self.NDUST))
        for i in range(self.NP):
            f.write('\n{:<15.3f} '.format(self.H[i]*1e-3))
            if self.NDUST >= 1:
                for j in range(self.NDUST):
                    f.write('{:<15.3E} '.format(self.DUST[i][j]/xscale[i]))    #particles per m-3
            else:
                f.write('{:<15.3E}'.format(self.DUST[i]))
        f.close()

    ##################################################################################

    def read_parah2(self):
        """
        Reads in the para-h2 fraction profile from parah2.ref.
        """
        try:
            with open('parah2.ref', 'r') as file:
                self.PARAH2 = []
                first = False
                for line in file:
                    if line.startswith('#'):
                        continue
                    if not first:
                        first = True
                        continue
                    self.PARAH2.append(float(line.split()[-1]))          
            
        except FileNotFoundError as e:
            return
        self.PARAH2 = np.array(self.PARAH2)
        if len(self.PARAH2) != len(self.P):
            raise Exception("Incorrect number of entries in parah2.ref!")
        
    ##################################################################################

    def write_parah2(self):
        """
        Writes the para-h2 fraction profile to parah2.ref.
        """
        if self.PARAH2 is None:
            return
       
        if self.NLOCATIONS!=1:
            raise ValueError('error :: write_parah2 only works if NLOCATIONS=1')
       
        f = open('parah2.ref','w')
        f.write('\t %i \n' % (self.NP))
        for i in range(self.NP):
            f.write('\t %7.4f \t %7.4f \n'  % (self.H[i]/1.0e3,self.PARAH2[i]))
        f.close()
            
    ##################################################################################
    
    def read_vpf(self):
        """
        Reads in saturation vapour pressures from runname.vpf.
        """
        try:
            with open(self.runname + '.vpf', 'r') as file:
                lines = file.readlines()        
            for line in lines[1:]:
                gas_id, iso_id, vp, svpflag = line.split()
                gas_id = int(gas_id)
                iso_id = int(iso_id)
                vp = float(vp)
                svpflag = int(svpflag)
                self.SVP[(gas_id,iso_id)] = (vp,svpflag)
                
        except FileNotFoundError as e:
            return
    
    
    ##################################################################################
    
    def write_vpf(self):
        """
        Writes the saturation vapour pressures to runname.vpf.
        """
        try:
            with open(self.runname + '.vpf', 'w') as file:
                # Write the number of gases
                file.write(f"{len(self.SVP)}\n")
                # Write each gas data line
                for (gas_id, iso_id), (vp, svpflag) in self.SVP.items():
                    file.write(f"{gas_id} {iso_id} {vp:.6g} {svpflag}\n")
        except Exception as e:
            _lgr.info(f"Error writing to {self.runname}.vpf: {e}")

    
    ##################################################################################
    
    def calc_coldens(self):
        """
        Routine to integrate the density of each gas at all altitudes
        
        Ouputs
        -------
        
        coldens(NVMR,NLOCATIONS) :: Column density of each gas (m-2)
        
        """
        
        from scipy import integrate
        
        #Calculate the number density at each layer (m-3)
        numdens = self.calc_numdens()
        
        #Calculating the partial number density of each gas (m-3)
        if self.NLOCATIONS>1:
            par_numdens = self.VMR * numdens[:, np.newaxis, :]

            #Integrate the number density as a function of altitude (m-2)
            par_coldens = np.zeros((self.NVMR,self.NLOCATIONS))
            for iLOCATION in range(self.NLOCATIONS):
                par_coldens[:,iLOCATION] = integrate.simpson(par_numdens[:,:,iLOCATION],x=self.H[:,iLOCATION],axis=0)
                
        else:
            par_numdens = self.VMR * numdens[:, np.newaxis]
            
            #Integrate the number density as a function of altitude (m-2)
            par_coldens = integrate.simpson(par_numdens[:,:],x=self.H[:],axis=0)
            
        return par_coldens
        
    ##################################################################################

    def plot_Atm(self,SavePlot=None,ILOCATION=0):

        """
        Makes a summary plot of the current atmospheric profiles
        """
        
        from archnemesis.Data.gas_data import gas_info, const

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True,figsize=(10,4))

        if self.NLOCATIONS==1:
            p = self.P
            t = self.T
            h = self.H
            vmr = self.VMR
        elif self.NLOCATIONS>1:
            p = self.P[:,ILOCATION]
            t = self.T[:,ILOCATION]
            h = self.H[:,ILOCATION]
            vmr = self.VMR[:,:,ILOCATION]

        ax1.semilogx(p/101325.,h/1.0e3,c='black')
        ax2.plot(t,h/1.0e3,c='black')
        for i in range(self.NVMR):
            label1 = gas_info[str(self.ID[i])]['name']
            if self.ISO[i]!=0:
                label1 = label1+' ('+str(self.ISO[i])+')'
            ax3.semilogx(vmr[:,i],h/1.0e3, label=label1)
        ax1.set_xlabel('Pressure (atm)')
        ax1.set_ylabel('Altitude (km)')
        ax2.set_xlabel('Temperature (K)')
        ax3.set_xlabel('Volume mixing ratio')
        plt.subplots_adjust(left=0.08,bottom=0.12,right=0.88,top=0.96,wspace=0.16,hspace=0.20)
        legend = ax3.legend(bbox_to_anchor=(1.01, 1.02))
        ax1.grid()
        ax2.grid()
        ax3.grid()

        if SavePlot is not None:
            fig.savefig(SavePlot)
        else:
            plt.show()

    ##################################################################################

    def plot_Dust(self,SavePlot=None,ILOCATION=0):
        """
        Make a summary plot of the current dust profiles
        """

        if self.NDUST>0:

            if self.NLOCATIONS==1:
                h = self.H
                dust = self.DUST
            elif self.NLOCATIONS>1:
                h = self.H[:,ILOCATION]
                dust = self.DUST[:,:,ILOCATION]

            fig,ax1 = plt.subplots(1,1,figsize=(3,4))

            for i in range(self.NDUST):
                ax1.plot(dust[:,i],h/1.0e3, label=f'aerosol species {i}')
            ax1.grid()
            ax1.set_xlabel('Aerosol density (particles m$^{-3}$)')
            ax1.set_ylabel('Altitude (km)')
            legend = ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
            plt.tight_layout()
            if SavePlot is not None:
                fig.savefig(SavePlot)
            else:
                plt.show()

        else:

            _lgr.warning(' :: there are no aerosol populations defined in Atmosphere')
            
    ##################################################################################
            
    def plot_map(self,varplot,labelplot='Variable (unit)',subobs_lat=None,subobs_lon=None,cmap='viridis',vmin=None,vmax=None):
        """
        Function to plot a given variable over on a map
        
        Inputs
        -------
        
        varplot(NLOCATIONS) :: Variable to be plotted at each of the planet's locations
        """

        from mpl_toolkits.basemap import Basemap
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig,ax1 = plt.subplots(1,1,figsize=(4,4))

        #Plotting the geometry
        if((subobs_lat!=None) & (subobs_lon!=None)):
            map = Basemap(projection='ortho', resolution=None,
                lat_0=subobs_lat, lon_0=subobs_lon)
        else:
            map = Basemap(projection='ortho', resolution=None,
                lat_0=np.mean(self.LATITUDE), lon_0=np.mean(self.LONGITUDE))
            
        lats = map.drawparallels(np.linspace(-90, 90, 13))
        lons = map.drawmeridians(np.linspace(-180, 180, 13))

        im = map.scatter(self.LONGITUDE,self.LATITUDE,latlon=True,c=varplot,cmap=cmap,vmin=vmin,vmax=vmax)

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("bottom", size="5%", pad=0.15)
        cbar2 = plt.colorbar(im,cax=cax,orientation='horizontal')
        cbar2.set_label(labelplot)

        ax1.grid()
        plt.tight_layout()
        plt.show()