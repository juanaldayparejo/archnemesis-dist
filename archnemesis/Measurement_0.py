from archnemesis import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import os,sys
from numba import jit

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

###############################################################################################

"""
Created on Tue Mar 29 17:27:12 2021

@author: juanalday

State Vector Class.
"""

class Measurement_0:

    """Measurement class.

    This class includes all information required to model the specification of the measurement, such as the geometry of the observation
    or the instrument characteristics. 

    Attributes
    ----------
    runname : str
        Name of the Nemesis run
    NGEOM : int       
        Number of observing geometries
    FWHM : float
        Full-width at half-maximum of the instrument
    ISHAPE : int
        Instrument lineshape (only used if FWHM>0)
            (0) Square lineshape
            (1) Triangular
            (2) Gaussian
            (3) Hamming
            (4) Hanning
    ISPACE : int
        Spectral units
            (0) Wavenumber (cm-1)
            (1) Wavelength (um)
    IFORM : int
        Units of the spectra
            (0) Radiance - W cm-2 sr-1 (cm-1)-1 if ISPACE=0 ---- W cm-2 sr-1 μm-1 if ISPACE=1
            (1) F_planet/F_star - Dimensionsless
            (2) A_planet/A_star - 100.0 * A_planet/A_star (dimensionsless)
            (3) Integrated spectral power of planet - W (cm-1)-1 if ISPACE=0 ---- W um-1 if ISPACE=1
            (4) Atmospheric transmission multiplied by solar flux
    LATITUDE : float
        Planetocentric latitude at centre of the field of view
    LONGITUDE : float
        Planetocentric longitude at centre of the field of view
    V_DOPPLER : float
        Doppler velocity between the observed body and the observer (km/s)
        It is considered positive if source body is moving towards observer, and negative if it is moving away
    NCONV : 1D array, int (NGEOM)
        Number of convolution spectral points in each spectrum
    NAV : 1D array, int (NGEOM)
        For each geometry, number of individual geometries need to be calculated and averaged to reconstruct the field of view
    VCONV : 2D array, float (NCONV,NGEOM)
        Convolution spectral points (wavelengths/wavenumbers) in each spectrum
    MEAS : 2D array, float (NCONV,NGEOM)
        Measured spectrum for each geometry
    ERRMEAS : 2D array, float (NCONV,NGEOM)
        Noise in the measured spectrum for each geometry
    FLAT : 2D array, float (NGEOM,AV)
        Latitude of each averaging point needed to reconstruct the FOV (when NAV > 1)
    FLON : 2D array, float (NGEOM,NAV)
        Longitude of each averaging point needed to reconstruct the FOV (when NAV > 1)
    SOL_ANG : 2D array, float (NGEOM,NAV)
        Solar indicent angle of each averaging point needed to reconstruct the FOV (when NAV > 1)
    EMISS_ANG : 2D array, float (NGEOM,NAV)
        Emission angle of each averaging point needed to reconstruct the FOV (when NAV > 1)
    AZI_ANG : 2D array, float (NGEOM,NAV)
        Azimuth angle of each averaging point needed to reconstruct the FOV (when NAV > 1)
    TANHE : 2D array, float (NGEOM,NAV)
        Tangent height of each averaging point needed to reconstruct the FOV (when NAV > 1)
        (For limb or solar occultation observations)
    WGEOM : 2D array, float (NGEOM,NAV)
        Weights of each point for the averaging of the FOV (when NAV > 1)
    NWAVE : int
        Number of calculation wavelengths required to model the convolution wavelengths
    WAVE : 1D array (NWAVE)
        Calculation wavenumbers for one particular geometry
    NFIL : 1D array, int (NCONV)
        If FWHM<0.0, the ILS is expected to be defined separately for each convolution wavenumber.
        NFIL represents the number of spectral points to defined the ILS for each convolution wavenumber.
    VFIL : 2D array, int (NFIL,NCONV)
        If FWHM<0.0, the ILS is expected to be defined separately for each convolution wavenumber.
        VFIL represents the calculation wavenumbers at which the ILS is defined for each each convolution wavenumber.
    AFIL : 2D array, int (NFIL,NCONV)
        If FWHM<0.0, the ILS is expected to be defined separately for each convolution wavenumber.
        AFIL represents the value of the ILS at each VFIL for each convolution wavenumber.
    NY : int
        Number of points in the Measurement vector (sum of all NCONV)
    Y : 1D array, float (NY)
        Measurement vector (concatenation of all spectra in the class)
    SE : 2D array, float (NY,NY)
        Measurement uncertainty covariance matrix (assumed to be diagonal)
    SPECMOD : 2D array, float (NCONV,NGEOM)
        Modelled spectrum for each geometry
        
        
    Methods
    ----------
    
    Measurement_0.assess()
    Measurement_0.summary_info()
    
    Measurement_0.write_hdf5()
    Measurement_0.read_hdf5()
    Measurement_0.read_spx()
    Measurement_0.read_spx_SO()
    Measurement_0.write_spx()
    Measurement_0.write_spx_SO()
    Measurement_0.read_sha()
    Measurement_0.write_sha()
    Measurement_0.read_fil()
    Measurement_0.write_fil()
    
    Measurement_0.edit_VCONV()
    Measurement_0.edit_MEAS()
    Measurement_0.edit_ERRMEAS()
    Measurement_0.edit_SPECMOD()
    Measurement_0.edit_FLAT()
    Measurement_0.edit_FLON()
    Measurement_0.edit_SOL_ANG()
    Measurement_0.edit_EMISS_ANG()
    Measurement_0.edit_AZI_ANG()
    Measurement_0.edit_TANHE()
    Measurement_0.edit_WGEOM()
    
    Measurement_0.calc_MeasurementVector()
    
    Measurement_0.remove_geometry()
    Measurement_0.select_geometry()
    Measurement_0.select_geometries()
    Measurement_0.select_TANHE_SO()
    
    Measurement_0.wavesetc()
    Measurement_0.wavesetb()
    
    Measurement_0.calc_doppler_shift()
    Measurement_0.invert_doppler_shift()
    Measurement_0.correct_doppler_shift()
    
    Measurement_0.plot_SO()
    Measurement_0.plot_nadir()
    Measurement_0.plot_disc_averaging()
    
    """

    def __init__(self, runname='', NGEOM=1, FWHM=0.0, ISHAPE=2, IFORM=0, ISPACE=0, LATITUDE=0.0, LONGITUDE=0.0, V_DOPPLER=0.0, NCONV=[1], NAV=[1]):

        #Input parameters
        self.runname = runname
        self.NGEOM = NGEOM
        self.FWHM = FWHM
        self.ISPACE = ISPACE
        self.ISHAPE = ISHAPE
        self.IFORM = IFORM
        self.LATITUDE = LATITUDE        
        self.LONGITUDE = LONGITUDE
        self.V_DOPPLER = V_DOPPLER
        self.NAV = NAV       #np.zeros(NGEOM)
        self.NCONV = NCONV   #np.zeros(NGEOM)

        # Input the following profiles using the edit_ methods.
        self.VCONV = None # np.zeros(NCONV,NGEOM)
        self.MEAS =  None # np.zeros(NCONV,NGEOM)
        self.ERRMEAS = None # np.zeros(NCONV,NGEOM)
        self.FLAT = None # np.zeros(NGEOM,NAV)
        self.FLON = None # np.zeros(NGEOM,NAV)
        self.SOL_ANG = None # np.zeros(NGEOM,NAV)
        self.EMISS_ANG = None # np.zeros(NGEOM,NAV)
        self.AZI_ANG = None # np.zeros(NGEOM,NAV)
        self.TANHE = None # np.zeros(NGEOM,NAV)
        self.WGEOM = None # np.zeros(NGEOM,NAV)
        self.NY = None #np.sum(NCONV)
        self.Y = None #np.zeros(NY)
        self.SE = None #np.zeros(NY,NY)

        self.SPECMOD = None #np.zeros(NCONV,NGEOM)

        self.NFIL = None  #np.zeros(NCONV)
        self.VFIL = None  #np.zeros(NFIL,NCONV)
        self.AFIL = None  #np.zeros(NFIL,NCONV)

        self.NWAVE = None
        self.WAVE = None #np.zeros(NWAVE)


    #################################################################################################################

    def assess(self):
        """
        Assess whether the different variables have the correct dimensions and types
        """

        #Checking some common parameters to all cases
        assert np.issubdtype(type(self.NGEOM), np.integer) == True , \
            'NGEOM must be int'
        assert self.NGEOM > 0 , \
            'NGEOM must be >0'
        
        assert np.issubdtype(type(self.IFORM), np.integer) == True , \
            'IFORM must be int'
        assert self.IFORM >= 0 , \
            'IFORM must be >=0 and <=4'
        assert self.IFORM <= 4 , \
            'IFORM must be >=0 and <=4'

        assert np.issubdtype(type(self.ISPACE), np.integer) == True , \
            'ISPACE must be int'
        assert self.ISPACE >= 0 , \
            'ISPACE must be >=0 and <=1'
        assert self.ISPACE <= 1 , \
            'ISPACE must be >=0 and <=1'
        
        assert np.issubdtype(type(self.FWHM), float) == True , \
            'FWHM must be float'
            
        assert np.issubdtype(type(self.V_DOPPLER), float) == True , \
            'V_DOPPLER must be float'
        
        assert len(self.NCONV) == self.NGEOM , \
            'NCONV must have size (NGEOM)'
        
        assert self.VCONV.shape == (self.NCONV.max(),self.NGEOM) , \
            'VCONV must have size (NCONV,NGEOM)'
        
        assert self.MEAS.shape == (self.NCONV.max(),self.NGEOM) , \
            'MEAS must have size (NCONV,NGEOM)'
        
        assert self.ERRMEAS.shape == (self.NCONV.max(),self.NGEOM) , \
            'ERRMEAS must have size (NCONV,NGEOM)'
        
        assert len(self.NAV) == self.NGEOM , \
            'NAV must have size (NGEOM)'
        
        assert self.FLAT.shape == (self.NGEOM,self.NAV.max()) , \
            'FLAT must have size (NGEOM,NAV)'
        
        assert self.FLON.shape == (self.NGEOM,self.NAV.max()) , \
            'FLON must have size (NGEOM,NAV)'
        
        assert self.WGEOM.shape == (self.NGEOM,self.NAV.max()) , \
            'WGEOM must have size (NGEOM,NAV)'
        
        assert self.EMISS_ANG.shape == (self.NGEOM,self.NAV.max()) , \
            'EMISS_ANG must have size (NGEOM,NAV)'

        #Checking if there are any limb-viewing geometries
        if self.EMISS_ANG.min()<0.0:
            assert self.TANHE.shape == (self.NGEOM,self.NAV.max()) , \
                'TANHE must have size (NGEOM,NAV)'
            
        #Checking if there are any nadir-viewing / upward looking geometries
        if self.EMISS_ANG.max() >= 0.0:
            assert self.SOL_ANG.shape == (self.NGEOM,self.NAV.max()) , \
                'SOL_ANG must have size (NGEOM,NAV)'
            assert self.AZI_ANG.shape == (self.NGEOM,self.NAV.max()) , \
                'AZI_ANG must have size (NGEOM,NAV)'


        if self.FWHM > 0.0: #Analytical instrument lineshape

            assert np.issubdtype(type(self.ISHAPE), np.integer) == True , \
                'ISHAPE must be int'
            
        elif self.FWHM < 0.0: #Explicit instrument lineshape in each wavelength

            assert len(np.unique(self.NCONV)) == 1 , \
                'All geometries must have same number of spectral bins if FWHM<0'

            assert len(self.NFIL) == self.NCONV[0] , \
                'NFIL must have size (NCONV)'
            
            assert self.VFIL.shape == (self.NFIL.max(),self.NCONV[0]) , \
                'VFIL must have size (NFIL,NCONV)'
            
            assert self.AFIL.shape == (self.NFIL.max(),self.NCONV[0]) , \
                'AFIL must have size (NFIL.max,NCONV)'
            
    #################################################################################################################
            
    def summary_info(self):
        """
        Subroutine to print summary of information about the class
        """      

        #Defining spectral resolution
        if self.FWHM>0.0:
            print('Spectral resolution of the measurement (FWHM) :: ',self.FWHM)
        elif self.FWHM<0.0:
            print('Instrument line shape defined in .fil file')
        else:
            print('Spectral resolution of the measurement is account for in the k-tables')


        #Defining geometries
        print('Field-of-view centered at :: ','Latitude',self.LATITUDE,'- Longitude',self.LONGITUDE)
        print('There are ',self.NGEOM,'geometries in the measurement vector')
        for i in range(self.NGEOM):
            print('')
            print('GEOMETRY '+str(i+1))
            print('Minimum wavelength/wavenumber :: ',self.VCONV[0,i],' - Maximum wavelength/wavenumber :: ',self.VCONV[self.NCONV[i]-1,i])
            if self.NAV[i]>1:
                print(self.NAV[i],' averaging points')
                for j in range(self.NAV[i]):
                
                    if self.EMISS_ANG[i,j]<0.0:
                        if isinstance(self.TANHE,np.ndarray)==True:
                            print('Averaging point',j+1,' - Weighting factor ',self.WGEOM[i,j])
                            print('Limb-viewing or solar occultation measurement. Latitude :: ',self.FLAT[i,j],' - Longitude :: ',self.FLON[i,j],' - Tangent height :: ',self.TANHE[i,j])
                        else:
                            print('Averaging point',j+1,' - Weighting factor ',self.WGEOM[i,j])
                            print('Limb-viewing or solar occultation measurement. Latitude :: ',self.FLAT[i,j],' - Longitude :: ',self.FLON[i,j],' - Tangent height :: ',self.SOL_ANG[i,j])
                    
                    else:
                        print('Averaging point',j+1,' - Weighting factor ',self.WGEOM[i,j])
                        print('Nadir-viewing geometry. Latitude :: ',self.FLAT[i,j],' - Longitude :: ',self.FLON[i,j],' - Emission angle :: ',self.EMISS_ANG[i,j],' - Solar Zenith Angle :: ',self.SOL_ANG[i,j],' - Azimuth angle :: ',self.AZI_ANG[i,j])

            else:
                j = 0
                if self.EMISS_ANG[i,j]<0.0:
                    if isinstance(self.TANHE,np.ndarray)==True:
                        print('Limb-viewing or solar occultation measurement. Latitude :: ',self.FLAT[i,j],' - Longitude :: ',self.FLON[i,j],' - Tangent height :: ',self.TANHE[i,j])
                    else:
                        print('Limb-viewing or solar occultation measurement. Latitude :: ',self.FLAT[i,j],' - Longitude :: ',self.FLON[i,j],' - Tangent height :: ',self.SOL_ANG[i,j])
                else:
                    print('Nadir-viewing geometry. Latitude :: ',self.FLAT[i,j],' - Longitude :: ',self.FLON[i,j],' - Emission angle :: ',self.EMISS_ANG[i,j],' - Solar Zenith Angle :: ',self.SOL_ANG[i,j],' - Azimuth angle :: ',self.AZI_ANG[i,j])

            
    #################################################################################################################
            
    def write_hdf5(self,runname):
        """
        Write the Measurement parameters into an HDF5 file
        """

        import h5py

        #Assessing that all the parameters have the correct type and dimension
        self.assess()

        f = h5py.File(runname+'.h5','a')
        #Checking if Atmosphere already exists
        if ('/Measurement' in f)==True:
            del f['Measurement']   #Deleting the Measurement information that was previously written in the file

        grp = f.create_group("Measurement")

        #Writing the latitude/longitude at the centre of FOV
        dset = grp.create_dataset('LATITUDE',data=self.LATITUDE)
        dset.attrs['title'] = "Latitude at centre of FOV"
        dset.attrs['units'] = 'degrees'

        dset = grp.create_dataset('LONGITUDE',data=self.LONGITUDE)
        dset.attrs['title'] = "Longitude at centre of FOV"
        dset.attrs['units'] = 'degrees'
        
        #Writing the Doppler velocity
        dset = grp.create_dataset('V_DOPPLER',data=self.V_DOPPLER)
        dset.attrs['title'] = "Doppler velocity between the observed body and the observer"
        dset.attrs['units'] = 'km s-1'

        #Writing the spectral units
        dset = grp.create_dataset('ISPACE',data=self.ISPACE)
        dset.attrs['title'] = "Spectral units"
        if self.ISPACE==0:
            dset.attrs['units'] = 'Wavenumber / cm-1'
        elif self.ISPACE==1:
            dset.attrs['units'] = 'Wavelength / um'

        #Writing the measurement units
        dset = grp.create_dataset('IFORM',data=self.IFORM)
        dset.attrs['title'] = "Measurement units"
        
        if self.ISPACE==0:  #Wavenumber space
            if self.IFORM==0:
                lunit = 'Radiance / W cm-2 sr-1 (cm-1)-1'
            elif self.IFORM==1:
                lunit = 'Secondary transit depth (Fplanet/Fstar) / Dimensionless'
            elif self.IFORM==2:
                lunit = 'Primary transit depth (100*Aplanet/Astar) / Dimensionless'
            elif self.IFORM==3:
                lunit = 'Integrated spectral power of planet / W (cm-1)-1'
            elif self.IFORM==4:
                lunit = 'Atmospheric transmission multiplied by solar flux / W cm-2 (cm-1)-1'

        elif self.ISPACE==1:  #Wavelength space
            if self.IFORM==0:
                lunit = 'Radiance / W cm-2 sr-1 um-1'
            elif self.IFORM==1:
                lunit = 'Secondary transit depth (Fplanet/Fstar) / Dimensionless'
            elif self.IFORM==2:
                lunit = 'Primary transit depth (100*Aplanet/Astar) / Dimensionless'
            elif self.IFORM==3:
                lunit = 'Integrated spectral power of planet / W um-1'
            elif self.IFORM==4:
                lunit = 'Atmospheric transmission multiplied by solar flux / W cm-2 um-1'

        dset.attrs['units'] = lunit

        #Writing the number of geometries
        dset = grp.create_dataset('NGEOM',data=self.NGEOM)
        dset.attrs['title'] = "Number of measurement geometries"

        #Defining the averaging points required to reconstruct the field of view 
        dset = grp.create_dataset('NAV',data=self.NAV)
        dset.attrs['title'] = "Number of averaging points needed to reconstruct the field-of-view"

        dset = grp.create_dataset('FLAT',data=self.FLAT)
        dset.attrs['title'] = "Latitude of each averaging point needed to reconstruct the field-of-view"
        dset.attrs['unit'] = "Degrees"
 
        dset = grp.create_dataset('FLON',data=self.FLON)
        dset.attrs['title'] = "Longitude of each averaging point needed to reconstruct the field-of-view"
        dset.attrs['unit'] = "Degrees"

        dset = grp.create_dataset('WGEOM',data=self.WGEOM)
        dset.attrs['title'] = "Weight of each averaging point needed to reconstruct the field-of-view"
        dset.attrs['unit'] = ""

        dset = grp.create_dataset('EMISS_ANG',data=self.EMISS_ANG)
        dset.attrs['title'] = "Emission angle of each averaging point needed to reconstruct the field-of-view"
        dset.attrs['unit'] = "Degrees"

        #Checking if there are any limb-viewing geometries
        if np.nanmin(self.EMISS_ANG)<0.0:

            dset = grp.create_dataset('TANHE',data=self.TANHE)
            dset.attrs['title'] = "Tangent height of each averaging point needed to reconstruct the field-of-view"
            dset.attrs['unit'] = "km"

        #Checking if there are any nadir-viewing / upward looking geometries
        if np.nanmax(self.EMISS_ANG) >= 0.0:

            dset = grp.create_dataset('SOL_ANG',data=self.SOL_ANG)
            dset.attrs['title'] = "Solar zenith angle of each averaging point needed to reconstruct the field-of-view"
            dset.attrs['unit'] = "Degrees"

            dset = grp.create_dataset('AZI_ANG',data=self.AZI_ANG)
            dset.attrs['title'] = "Azimuth angle of each averaging point needed to reconstruct the field-of-view"
            dset.attrs['unit'] = "Degrees"

        dset = grp.create_dataset('NCONV',data=self.NCONV)
        dset.attrs['title'] = "Number of spectral bins in each geometry"

        dset = grp.create_dataset('VCONV',data=self.VCONV)
        dset.attrs['title'] = "Spectral bins"
        if self.ISPACE==0:
            dset.attrs['units'] = 'Wavenumber / cm-1'
        elif self.ISPACE==1:
            dset.attrs['units'] = 'Wavelength / um'

        dset = grp.create_dataset('MEAS',data=self.MEAS)
        dset.attrs['title'] = "Measured spectrum in each geometry"
        dset.attrs['units'] = lunit

        dset = grp.create_dataset('ERRMEAS',data=self.ERRMEAS)
        dset.attrs['title'] = "Uncertainty in the measured spectrum in each geometry"
        dset.attrs['units'] = lunit

        if self.FWHM>0.0:
            dset = grp.create_dataset('ISHAPE',data=self.ISHAPE)
            dset.attrs['title'] = "Instrument lineshape"
            if self.ISHAPE==0:
                lils = 'Square function'
            elif self.ISHAPE==1:
                lils = 'Triangular function'
            elif self.ISHAPE==2:
                lils = 'Gaussian function'
            elif self.ISHAPE==3:
                lils = 'Hamming function'
            elif self.ISHAPE==4:
                lils = 'Hanning function'
            dset.attrs['type'] = lils

        dset = grp.create_dataset('FWHM',data=self.FWHM)
        dset.attrs['title'] = "FWHM of instrument lineshape"
        if self.FWHM>0.0:
            if self.ISPACE==0:
                dset.attrs['units'] = 'Wavenumber / cm-1'
            elif self.ISPACE==1:
                dset.attrs['units'] = 'Wavelength / um'
            dset.attrs['type'] = 'Analytical lineshape ('+lils+')'
        elif self.FWHM==0:
            dset.attrs['type'] = 'Convolution already performed in k-tables'
        elif self.FWHM<0.0:
            dset.attrs['type'] = 'Explicit definition of instrument lineshape in each spectral bin'

        if self.FWHM<0.0:
            dset = grp.create_dataset('NFIL',data=self.NFIL)
            dset.attrs['title'] = "Number of points required to define the ILS in each spectral bin"

            if self.ISPACE==0:
                dset = grp.create_dataset('VFIL',data=self.VFIL)
                dset.attrs['title'] = "Wavenumber of the points required to define the ILS in each spectral bin"
                dset.attrs['unit'] = "Wavenumber / cm-1"
            elif self.ISPACE==1:
                dset = grp.create_dataset('VFIL',data=self.VFIL)
                dset.attrs['title'] = "Wavelength of the points required to define the ILS in each spectral bin"
                dset.attrs['unit'] = "Wavelength / um"

            dset = grp.create_dataset('AFIL',data=self.AFIL)
            dset.attrs['title'] = "ILS in each spectral bin"
            dset.attrs['unit'] = ""

        f.close()

    #################################################################################################################

    def read_hdf5(self,runname):
        """
        Read the Measurement properties from an HDF5 file
        """

        import h5py

        f = h5py.File(runname+'.h5','r')

        #Checking if Measurement exists
        e = "/Measurement" in f
        if e==False:
            sys.exit('error :: Measurement is not defined in HDF5 file')
        else:

            self.NGEOM = np.int32(f.get('Measurement/NGEOM'))
            self.ISPACE = np.int32(f.get('Measurement/ISPACE'))
            self.IFORM = np.int32(f.get('Measurement/IFORM'))
            self.LATITUDE = np.float64(f.get('Measurement/LATITUDE'))
            self.LONGITUDE = np.float64(f.get('Measurement/LONGITUDE'))
            self.NAV = np.array(f.get('Measurement/NAV'))
            self.FLAT = np.array(f.get('Measurement/FLAT'))
            self.FLON = np.array(f.get('Measurement/FLON'))
            self.WGEOM = np.array(f.get('Measurement/WGEOM'))
            self.EMISS_ANG = np.array(f.get('Measurement/EMISS_ANG'))
            
            #Reading Doppler shift if exists
            if 'Measurement/V_DOPPLER' in f:
                self.V_DOPPLER = np.float64(f.get('Measurement/V_DOPPLER'))

            #Checking if there are any limb-viewing geometries
            if np.nanmin(self.EMISS_ANG)<0.0:
                self.TANHE = np.array(f.get('Measurement/TANHE'))

            #Checking if there are any nadir-viewing / upward looking geometries
            if np.nanmax(self.EMISS_ANG) >= 0.0:
                self.SOL_ANG = np.array(f.get('Measurement/SOL_ANG'))
                self.AZI_ANG = np.array(f.get('Measurement/AZI_ANG'))


            self.NCONV = np.array(f.get('Measurement/NCONV'))
            self.VCONV = np.array(f.get('Measurement/VCONV'))
            self.MEAS = np.array(f.get('Measurement/MEAS'))
            self.ERRMEAS = np.array(f.get('Measurement/ERRMEAS'))

            self.FWHM = np.float64(f.get('Measurement/FWHM'))
            if self.FWHM>0.0:
                self.ISHAPE = np.int32(f.get('Measurement/ISHAPE'))
            elif self.FWHM<0.0:
                self.NFIL = np.array(f.get('Measurement/NFIL'))
                self.VFIL = np.array(f.get('Measurement/VFIL'))
                self.AFIL = np.array(f.get('Measurement/AFIL'))

        f.close()

        self.assess()

        #self.calc_MeasurementVector()
            
    #################################################################################################################
            
    def read_spx(self):
    
        """
        Read the .spx file and fill the attributes and parameters of the Measurement class.
        """

        #Opening file
        f = open(self.runname+'.spx','r')

        #Reading first line
        tmp = np.fromfile(f,sep=' ',count=4,dtype='float')
        inst_fwhm = float(tmp[0])
        xlat = float(tmp[1])
        xlon = float(tmp[2])
        ngeom = int(tmp[3])

        #Defining variables
        navmax = 100
        nconvmax = 15000
        nconv = np.zeros([ngeom],dtype='int')
        nav = np.zeros([ngeom],dtype='int')
        flattmp = np.zeros([ngeom,navmax])
        flontmp = np.zeros([ngeom,navmax])
        sol_angtmp = np.zeros([ngeom,navmax])
        emiss_angtmp = np.zeros([ngeom,navmax])
        azi_angtmp = np.zeros([ngeom,navmax])
        wgeomtmp = np.zeros([ngeom,navmax])
        wavetmp = np.zeros([nconvmax,ngeom])
        meastmp = np.zeros([nconvmax,ngeom])
        errmeastmp = np.zeros([nconvmax,ngeom])
        for i in range(ngeom):
            nconv[i] = int(f.readline().strip())
            nav[i] = int(f.readline().strip())
            for j in range(nav[i]):
                tmp = np.fromfile(f,sep=' ',count=6,dtype='float')
                flattmp[i,j] = float(tmp[0])
                flontmp[i,j] = float(tmp[1])
                sol_angtmp[i,j] = float(tmp[2])
                emiss_angtmp[i,j] = float(tmp[3])
                azi_angtmp[i,j] = float(tmp[4])
                wgeomtmp[i,j] = float(tmp[5])
            for iconv in range(nconv[i]):
                tmp = np.fromfile(f,sep=' ',count=3,dtype='float')
                wavetmp[iconv,i] = float(tmp[0])
                meastmp[iconv,i] = float(tmp[1])
                errmeastmp[iconv,i] = float(tmp[2])

        #Making final arrays for the measured spectra
        nconvmax2 = max(nconv)
        navmax2 = max(nav)
        wave = np.zeros([nconvmax2,ngeom])
        meas = np.zeros([nconvmax2,ngeom])
        errmeas = np.zeros([nconvmax2,ngeom])
        flat = np.zeros([ngeom,navmax2])
        flon = np.zeros([ngeom,navmax2])
        sol_ang = np.zeros([ngeom,navmax2])
        emiss_ang = np.zeros([ngeom,navmax2])
        azi_ang = np.zeros([ngeom,navmax2])
        wgeom = np.zeros([ngeom,navmax2])
        for i in range(ngeom):
            wave[0:nconv[i],i] = wavetmp[0:nconv[i],i]
            meas[0:nconv[i],i] = meastmp[0:nconv[i],i]
            errmeas[0:nconv[i],i] = errmeastmp[0:nconv[i],i]  
            flat[i,0:nav[i]] = flattmp[i,0:nav[i]]
            flon[i,0:nav[i]] = flontmp[i,0:nav[i]]
            sol_ang[i,0:nav[i]] = sol_angtmp[i,0:nav[i]]
            emiss_ang[i,0:nav[i]] = emiss_angtmp[i,0:nav[i]]
            azi_ang[i,0:nav[i]] = azi_angtmp[i,0:nav[i]]
            wgeom[i,0:nav[i]] = wgeomtmp[i,0:nav[i]]

        self.FWHM = inst_fwhm
        self.LATITUDE = xlat
        self.LONGITUDE = xlon
        self.NGEOM = ngeom
        self.NCONV = nconv
        self.NAV = nav
        self.edit_VCONV(wave)
        self.edit_MEAS(meas)
        self.edit_ERRMEAS(errmeas)
        self.edit_FLAT(flat)
        self.edit_FLON(flon)
        self.edit_WGEOM(wgeom)
        self.edit_SOL_ANG(sol_ang)
        self.edit_TANHE(sol_ang)
        self.edit_EMISS_ANG(emiss_ang)
        self.edit_AZI_ANG(azi_ang)

        self.calc_MeasurementVector()
            
    #################################################################################################################
            
    def read_spx_SO(self):
        """
        Read the .spx file and fill the attributes and parameters of the Measurement class.
        This routine is specific for solar occultation and limb observations
        """

        #Opening file
        f = open(self.runname+'.spx','r')
    
        #Reading first line
        tmp = np.fromfile(f,sep=' ',count=4,dtype='float')
        inst_fwhm = float(tmp[0])
        xlat = float(tmp[1])
        xlon = float(tmp[2])
        ngeom = int(tmp[3])
    
        #Defining variables
        nav = 1 #it needs to be generalized to read more than one NAV per observation geometry
        nconv = np.zeros([ngeom],dtype='int')
        flat = np.zeros([ngeom,nav])
        flon = np.zeros([ngeom,nav])
        tanhe = np.zeros([ngeom,nav])
        wgeom = np.zeros([ngeom,nav])
        nconvmax = 20000
        emiss_ang = np.zeros((ngeom,nav))
        wavetmp = np.zeros([nconvmax,ngeom])
        meastmp = np.zeros([nconvmax,ngeom])
        errmeastmp = np.zeros([nconvmax,ngeom])
        for i in range(ngeom):
            nconv[i] = int(f.readline().strip())
            for j in range(nav):
                navsel = int(f.readline().strip())
                tmp = np.fromfile(f,sep=' ',count=6,dtype='float')
                flat[i,j] = float(tmp[0])
                flon[i,j] = float(tmp[1])
                tanhe[i,j] = float(tmp[2])
                emiss_ang[i,j] = float(tmp[3])
                wgeom[i,j] = float(tmp[5])
            for iconv in range(nconv[i]):
                tmp = np.fromfile(f,sep=' ',count=3,dtype='float')
                wavetmp[iconv,i] = float(tmp[0])
                meastmp[iconv,i] = float(tmp[1])
                errmeastmp[iconv,i] = float(tmp[2])


        #Making final arrays for the measured spectra
        nconvmax2 = max(nconv)
        wave = np.zeros([nconvmax2,ngeom])
        meas = np.zeros([nconvmax2,ngeom])
        errmeas = np.zeros([nconvmax2,ngeom])
        for i in range(ngeom):
            wave[0:nconv[i],:] = wavetmp[0:nconv[i],:]
            meas[0:nconv[i],:] = meastmp[0:nconv[i],:]
            errmeas[0:nconv[i],:] = errmeastmp[0:nconv[i],:]

        self.NGEOM = ngeom
        self.FWHM = inst_fwhm
        self.LATITUDE = xlat
        self.LONGITUDE = xlon
        self.NCONV = nconv
        self.NAV = np.ones(ngeom,dtype='int32')

        self.edit_WGEOM(wgeom)
        self.edit_FLAT(flat)
        self.edit_FLON(flon)
        self.edit_VCONV(wave)
        self.edit_MEAS(meas)
        self.edit_ERRMEAS(errmeas)
        self.edit_TANHE(tanhe)
        self.edit_EMISS_ANG(emiss_ang)

        self.calc_MeasurementVector()

    #################################################################################################################

    def write_spx(self):
    
        """
        Write the .spx file based on the information on the Measurement class
        """

        fspx = open(self.runname+'.spx','w')
        fspx.write('%7.5f \t %7.5f \t %7.5f \t %i \n' % (self.FWHM,self.LATITUDE,self.LONGITUDE,self.NGEOM))

        for i in range(self.NGEOM):
            fspx.write('\t %i \n' % (self.NCONV[i]))
            fspx.write('\t %i \n' % (self.NAV[i]))
            for j in range(self.NAV[i]):
                fspx.write('\t %7.4f \t %7.4f \t %7.4f \t %7.4f \t %7.4f \t %7.4f \t \n' % (self.FLAT[i,j],self.FLON[i,j],self.SOL_ANG[i,j],self.EMISS_ANG[i,j],self.AZI_ANG[i,j],self.WGEOM[i,j]))
                for k in range(self.NCONV[i]):
                    fspx.write('\t %10.5f \t %20.7f \t %20.7f \n' % (self.VCONV[k,i],self.MEAS[k,i],self.ERRMEAS[k,i]))

        fspx.close()

    #################################################################################################################

    def write_spx_SO(self):
    
        """
        Write the .spx file for a solar occultation measurement
        """

        fspx = open(self.runname+'.spx','w')
        fspx.write('%7.5f \t %7.5f \t %7.5f \t %i \n' % (self.FWHM,self.LATITUDE,self.LONGITUDE,self.NGEOM))

        for i in range(self.NGEOM):
            fspx.write('\t %i \n' % (self.NCONV[i]))
            fspx.write('\t %i \n' % (1))
            dummy1 = -1.0
            dummy2 = 180.0
            dummy3 = 1.0
            fspx.write('\t %7.4f \t %7.4f \t %7.4f \t %7.4f \t %7.4f \t %7.4f \t \n' % (self.LATITUDE,self.LONGITUDE,self.TANHE[i,0],dummy1,dummy2,dummy3))
            for k in range(self.NCONV[i]):
                fspx.write('\t %10.5f \t %20.7f \t %20.7f \n' % (self.VCONV[k,i],self.MEAS[k,i],self.ERRMEAS[k,i]))

        fspx.close()

    #################################################################################################################

    def read_sha(self):
        """
        Read the .sha file to see what the Instrument Lineshape.
        This file is only read if FWHM>0.
            (0) Square lineshape
            (1) Triangular
            (2) Gaussian
            (3) Hamming
            (4) Hanning 
        """

        #Opening file
        f = open(self.runname+'.sha','r')
        s = f.readline().split()
        lineshape = int(s[0])

        self.ISHAPE = lineshape

    #################################################################################################################

    def write_sha(self):
    
        """
        Write the .sha file to define the shape of the Instrument function
        (Only valid if FWHM>0.0)
        """

        if self.FWHM<0.0:
            sys.exit('error in write_sha() :: The .sha file is only used if FWHM>0')

        f = open(self.runname+'.sha','w')
        f.write("%i \n" %  (self.ISHAPE))
        f.close()

    #################################################################################################################

    def read_fil(self,MakePlot=False):
    
        """
        Read the .fil file to see what the Instrument Lineshape for each convolution wavenumber 
        This file is only read if FWHM<0.
        """

        #Opening file
        f = open(self.runname+'.fil','r')
    
        #Reading first and second lines
        nconv = int(np.fromfile(f,sep=' ',count=1,dtype='int'))
        wave = np.zeros([nconv],dtype='d')
        nfil = np.zeros([nconv],dtype='int')
        nfilmax = 100000
        vfil1 = np.zeros([nfilmax,nconv],dtype='d')
        afil1 = np.zeros([nfilmax,nconv],dtype='d')
        for i in range(nconv):
            wave[i] = np.fromfile(f,sep=' ',count=1,dtype='d')
            nfil[i] = np.fromfile(f,sep=' ',count=1,dtype='int')
            for j in range(nfil[i]):
                tmp = np.fromfile(f,sep=' ',count=2,dtype='d')
                vfil1[j,i] = tmp[0]
                afil1[j,i] = tmp[1]

        nfil1 = nfil.max()
        vfil = np.zeros([nfil1,nconv],dtype='d')
        afil = np.zeros([nfil1,nconv],dtype='d')
        for i in range(nconv):
            vfil[0:nfil[i],i] = vfil1[0:nfil[i],i]
            afil[0:nfil[i],i] = afil1[0:nfil[i],i]
    
        if self.NCONV[0]!=nconv:
            sys.exit('error :: Number of convolution wavelengths in .fil and .spx files must be the same')

        self.NFIL = nfil
        self.VFIL = vfil
        self.AFIL = afil

        if MakePlot==True:
            fsize = 11
            axis_font = {'size':str(fsize)}
            fig, ([ax1,ax2,ax3]) = plt.subplots(1,3,figsize=(12,4))
        
            ix = 0  #First wavenumber
            ax1.plot(vfil[0:nfil[ix],ix],afil[0:nfil[ix],ix],linewidth=2.)
            ax1.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)',**axis_font)
            ax1.set_ylabel(r'f($\nu$)',**axis_font)
            ax1.set_xlim([vfil[0:nfil[ix],ix].min(),vfil[0:nfil[ix],ix].max()])
            ax1.ticklabel_format(useOffset=False)
            ax1.grid()
        
            ix = int(nconv/2)-1  #Centre wavenumber
            ax2.plot(vfil[0:nfil[ix],ix],afil[0:nfil[ix],ix],linewidth=2.)
            ax2.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)',**axis_font)
            ax2.set_ylabel(r'f($\nu$)',**axis_font)
            ax2.set_xlim([vfil[0:nfil[ix],ix].min(),vfil[0:nfil[ix],ix].max()])
            ax2.ticklabel_format(useOffset=False)
            ax2.grid()
        
            ix = nconv-1  #Last wavenumber
            ax3.plot(vfil[0:nfil[ix],ix],afil[0:nfil[ix],ix],linewidth=2.)
            ax3.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)',**axis_font)
            ax3.set_ylabel(r'f($\nu$)',**axis_font)
            ax3.set_xlim([vfil[0:nfil[ix],ix].min(),vfil[0:nfil[ix],ix].max()])
            ax3.ticklabel_format(useOffset=False)
            ax3.grid()
        
            plt.tight_layout()
            plt.show()

    #################################################################################################################

    def write_fil(self,IGEOM=0):
    
        """
        Write the .fil file to see what the Instrument Lineshape for each convolution wavenumber 
        (Only valid if FWHM<0.0)
        """

        f = open(self.runname+'.fil','w')
        f.write("%i \n" %  (self.NCONV[IGEOM]))

        #Running for each spectral point
        for i in range(self.NCONV[IGEOM]):
            f.write("%10.7f\n" % self.VCONV[i,IGEOM])

            f.write("%i \n" %  (self.NFIL[i]))
            for j in range(self.NFIL[i]):
                f.write("%10.10f %10.10e\n" % (self.VFIL[j,i], self.AFIL[j,i]) )
        f.close()

    #################################################################################################################
            
    def edit_VCONV(self, VCONV_array):
        """
        Edit the convolution wavelengths/wavenumbers array in each geometry

        Parameters
        ----------
        VCONV_array : 2D array, float (NCONV,NGEOM)
            Convolution wavelengths/wavenumbers of the spectrum in each geometry

        """
        VCONV_array = np.array(VCONV_array)
        try:
            assert VCONV_array.shape == (self.NCONV.max(), self.NGEOM),\
                'VCONV should be NCONV by NGEOM.'
        except:
            assert VCONV_array.shape == (self.NCONV[0]) and self.NGEOM==1,\
                'VCONV should be NCONV.'

        self.VCONV = VCONV_array

    #################################################################################################################

    def edit_MEAS(self, MEAS_array):
        """
        Edit the measured spectrum in each geometry in each geometry

        Parameters
        ----------
        MEAS_array : 2D array, float (NCONV,NGEOM)
            Measured spectrum in each geometry

        """
        MEAS_array = np.array(MEAS_array)
        try:
            assert MEAS_array.shape == (self.NCONV.max(), self.NGEOM),\
                'MEAS should be NCONV by NGEOM.'
        except:
            assert MEAS_array.shape == (self.NCONV,) and self.NGEOM==1,\
                'MEAS should be NCONV.'

        self.MEAS = MEAS_array

    #################################################################################################################

    def edit_ERRMEAS(self, ERRMEAS_array):
        """
        Edit the measured uncertainty of the spectrum in each geometry

        Parameters
        ----------
        ERRMEAS_array : 2D array, float (NCONV,NGEOM)
            Measured uncertainty of the spectrum in each geometry
        """
        ERRMEAS_array = np.array(ERRMEAS_array)
        try:
            assert ERRMEAS_array.shape == (self.NCONV.max(), self.NGEOM),\
                'ERRMEAS should be NCONV by NGEOM.'
        except:
            assert ERRMEAS_array.shape == (self.NCONV,) and self.NGEOM==1,\
                'ERRMEAS should be NCONV.'

        self.ERRMEAS = ERRMEAS_array

    #################################################################################################################

    def edit_SPECMOD(self, SPECMOD_array):
        """
        Edit the modelled spectrum in each geometry in each geometry

        Parameters
        ----------
        SPECMOD_array : 2D array, float (NCONV,NGEOM)
            Modelled spectrum in each geometry

        """
        SPECMOD_array = np.array(SPECMOD_array)
        try:
            assert SPECMOD_array.shape == (self.NCONV.max(), self.NGEOM),\
                'SPECMOD should be NCONV by NGEOM.'
        except:
            assert SPECMOD_array.shape == (self.NCONV,) and self.NGEOM==1,\
                'SPECMOD should be NCONV.'

        self.SPECMOD = SPECMOD_array

    #################################################################################################################

    def edit_FLAT(self, FLAT_array):
        """
        Edit the latitude of each sub-geometry needed to reconstruct the FOV (when NAV > 1)

        Parameters 
        ----------
        FLAT_array : 2D array, float (NAV,NGEOM)
            Latitude of each averaging point needed
            to reconstruct the FOV (when NAV > 1)
        """

        FLAT_array = np.array(FLAT_array)
        try:
            assert FLAT_array.shape == (self.NGEOM, self.NAV.max()),\
                'FLAT should be NGEOM by NAV.'
        except:
            assert FLAT_array.shape == (self.NGEOM,self.NAV) and self.NGEOM==1,\
                'FLAT should be NAV.'

        self.FLAT = FLAT_array

    #################################################################################################################

    def edit_FLON(self, FLON_array):
        """
        Edit the longitude of each sub-geometry needed to reconstruct the FOV (when NAV > 1)

        Parameters 
        ----------
        FLON_array : 2D array, float (NAV,NGEOM)
            Longitude of each averaging point needed
            to reconstruct the FOV (when NAV > 1)
        """

        FLON_array = np.array(FLON_array)

        assert FLON_array.shape == (self.NGEOM, self.NAV.max()),\
            'FLON should be NGEOM by NAV.'

        self.FLON = FLON_array

    #################################################################################################################

    def edit_SOL_ANG(self, SOL_ANG_array):
        """
        Edit the solar zenith angle of each sub-geometry needed to reconstruct the FOV (when NAV > 1)

        Parameters 
        ----------
        SOL_ANG_array : 2D array, float (NAV,NGEOM)
            Solar zenith angle of each averaging point needed
            to reconstruct the FOV (when NAV > 1)
        """

        SOL_ANG_array = np.array(SOL_ANG_array)
        
        assert SOL_ANG_array.shape == (self.NGEOM, self.NAV.max()),\
            'SOL_ANG should be NGEOM by NAV.'

        self.SOL_ANG = SOL_ANG_array

    #################################################################################################################

    def edit_EMISS_ANG(self, EMISS_ANG_array):
        """
        Edit the emission angle of each sub-geometry needed to reconstruct the FOV (when NAV > 1)

        Parameters 
        ----------
        EMISS_ANG_array : 2D array, float (NAV,NGEOM)
            Azimuth angle of each averaging point needed
            to reconstruct the FOV (when NAV > 1)
        """

        EMISS_ANG_array = np.array(EMISS_ANG_array)
        
        assert EMISS_ANG_array.shape == (self.NGEOM, self.NAV.max()),\
            'EMISS_ANG should be NGEOM by NAV.'

        self.EMISS_ANG = EMISS_ANG_array

    #################################################################################################################

    def edit_AZI_ANG(self, AZI_ANG_array):
        """
        Edit the azimuth angle of each sub-geometry needed to reconstruct the FOV (when NAV > 1)

        Parameters 
        ----------
        AZI_ANG_array : 2D array, float (NAV,NGEOM)
            Azimuth angle of each averaging point needed
            to reconstruct the FOV (when NAV > 1)
        """

        AZI_ANG_array = np.array(AZI_ANG_array)
        
        assert AZI_ANG_array.shape == (self.NGEOM, self.NAV.max()),\
            'AZI_ANG should be NGEOM by NAV.'

        self.AZI_ANG = AZI_ANG_array

    #################################################################################################################

    def edit_TANHE(self, TANHE_array):
        """
        Edit the tangent height of each sub-geometry needed to reconstruct the FOV (when NAV > 1)

        Parameters 
        ----------
        TANHE_array : 2D array, float (NAV,NGEOM)
            Tangent height of each averaging point needed
            to reconstruct the FOV (when NAV > 1)
        """
        TANHE_array = np.array(TANHE_array)
        
        assert TANHE_array.shape == (self.NGEOM, self.NAV.max()),\
            'TANHE should be NGEOM by NAV.'

        self.TANHE = TANHE_array

    #################################################################################################################

    def edit_WGEOM(self, WGEOM_array):
        """
        Edit the weight of each sub-geometry needed to reconstruct the FOV (when NAV > 1)

        Parameters 
        ----------
        WGEOM_array : 2D array, float (NAV,NGEOM)
            Weight of each averaging point needed
            to reconstruct the FOV (when NAV > 1)

        """
        WGEOM_array = np.array(WGEOM_array)
        
        assert WGEOM_array.shape == (self.NGEOM, self.NAV.max()),\
            'WGEOM should be NGEOM by NAV.'

        self.WGEOM = WGEOM_array

    #################################################################################################################

    def calc_MeasurementVector(self):
        """
        Calculate the measurement vector based on the other parameters
        defined in this class
        """

        self.NY = np.sum(self.NCONV)
        y1 = np.zeros(self.NY)
        se1 = np.zeros(self.NY)
        ix = 0
        for i in range(self.NGEOM):
            y1[ix:ix+self.NCONV[i]] = self.MEAS[0:self.NCONV[i],i]
            se1[ix:ix+self.NCONV[i]] = self.ERRMEAS[0:self.NCONV[i],i]
            ix = ix + self.NCONV[i]

        self.Y = y1
        se = np.zeros([self.NY,self.NY])
        for i in range(self.NY):
            se[i,i] = se1[i]**2.

        self.SE = se

    #################################################################################################################

    def remove_geometry(self,IGEOM):
        """
        Remove one spectrum (i.e., one geometry) from the Measurement class

        Parameters
        ----------
        IGEOM : int
            Integer indicating the geometry to be erased (from 0 to NGEOM-1)

        """

        if IGEOM>self.NGEOM-1:
            sys.exit('error in remove_geometry :: IGEOM must be between 0 and NGEOM')

        self.NGEOM = self.NGEOM - 1
        self.NCONV = np.delete(self.NCONV,IGEOM,axis=0)

        self.VCONV = np.delete(self.VCONV,IGEOM,axis=1)
        self.MEAS = np.delete(self.MEAS,IGEOM,axis=1)
        self.ERRMEAS = np.delete(self.ERRMEAS,IGEOM,axis=1)
        self.FLAT = np.delete(self.FLAT,IGEOM,axis=0)
        self.FLON = np.delete(self.FLON,IGEOM,axis=0)

        if isinstance(self.TANHE,np.ndarray)==True:
            self.TANHE = np.delete(self.TANHE,IGEOM,axis=0)
        if isinstance(self.SOL_ANG,np.ndarray)==True:
            self.SOL_ANG = np.delete(self.SOL_ANG,IGEOM,axis=0)
        if isinstance(self.EMISS_ANG,np.ndarray)==True:
            self.EMISS_ANG = np.delete(self.EMISS_ANG,IGEOM,axis=0)
        if isinstance(self.AZI_ANG,np.ndarray)==True:
            self.AZI_ANG = np.delete(self.AZI_ANG,IGEOM,axis=0)
        if isinstance(self.WGEOM,np.ndarray)==True:
            self.WGEOM = np.delete(self.WGEOM,IGEOM,axis=0)
            
        self.assess()
        
    #################################################################################################################
        
    def select_geometry(self,IGEOM):
        """
        Select only one spectrum (i.e., one geometry) from the Measurement class
        and delete the rest of them

        Parameters
        ----------
        IGEOM : int
            Integer indicating the geometry to be selected (from 0 to NGEOM-1)

        """

        if IGEOM>self.NGEOM-1:
            sys.exit('error in select_geometry :: IGEOM must be between 0 and NGEOM')

        self.NGEOM = 1
        NCONV = np.zeros(self.NGEOM,dtype='int32')
        NCONV[0] = self.NCONV[IGEOM]
        self.NCONV = NCONV
        
        VCONV = np.zeros((NCONV.max(),1))
        MEAS = np.zeros((NCONV.max(),1))
        ERRMEAS = np.zeros((NCONV.max(),1))
        VCONV[:,0] = self.VCONV[0:NCONV[0],IGEOM]
        MEAS[:,0] = self.MEAS[0:NCONV[0],IGEOM]
        ERRMEAS[:,0] = self.ERRMEAS[0:NCONV[0],IGEOM]
        
        self.edit_VCONV(VCONV)
        self.edit_MEAS(MEAS)
        self.edit_ERRMEAS(ERRMEAS)
        
        NAV = np.zeros(self.NGEOM,dtype='int32')
        NAV[0] = self.NAV[IGEOM]
        self.NAV = NAV
        
        FLAT = np.zeros((self.NGEOM,self.NAV.max()))
        FLON = np.zeros((self.NGEOM,self.NAV.max()))
        WGEOM = np.zeros((self.NGEOM,self.NAV.max()))
        FLAT[0,:] = self.FLAT[IGEOM,0:NAV[0]]
        FLON[0,:] = self.FLON[IGEOM,0:NAV[0]]
        WGEOM[0,:] = self.WGEOM[IGEOM,0:NAV[0]]
        self.edit_FLAT(FLAT)
        self.edit_FLON(FLON)
        self.edit_WGEOM(WGEOM)
        
        if isinstance(self.TANHE,np.ndarray)==True:
            TANHE = np.zeros((self.NGEOM,self.NAV.max()))
            TANHE[0,:] = self.TANHE[IGEOM,0:NAV[0]]
            self.edit_TANHE(TANHE)
            
        if isinstance(self.EMISS_ANG,np.ndarray)==True:
            EMISS_ANG = np.zeros((self.NGEOM,self.NAV.max()))
            EMISS_ANG[0,:] = self.EMISS_ANG[IGEOM,0:NAV[0]]
            self.edit_EMISS_ANG(EMISS_ANG)
            
        if isinstance(self.SOL_ANG,np.ndarray)==True:
            SOL_ANG = np.zeros((self.NGEOM,self.NAV.max()))
            SOL_ANG[0,:] = self.SOL_ANG[IGEOM,0:NAV[0]]
            self.edit_SOL_ANG(SOL_ANG)
            
        if isinstance(self.AZI_ANG,np.ndarray)==True:
            AZI_ANG = np.zeros((self.NGEOM,self.NAV.max()))
            AZI_ANG[0,:] = self.AZI_ANG[IGEOM,0:NAV[0]]
            self.edit_AZI_ANG(AZI_ANG)
            
        self.assess()
          
    #################################################################################################################
          
    def select_geometries(self,IGEOM):
        """
        Select only some spectra (i.e., some geometries) from the Measurement class
        and delete the rest of them

        Parameters
        ----------
        IGEOM : 1D array
            Array of integers indicating the geometry to be selected (from 0 to NGEOM-1)

        """

        NGEOMsel = len(IGEOM)
        if np.max(IGEOM)>self.NGEOM-1:
            sys.exit('error in select_geometries :: IGEOM must be between 0 and NGEOM')
        if np.min(IGEOM)<0:
            sys.exit('error in select_geometries :: IGEOM must be between 0 and NGEOM')

        self.NGEOM = NGEOMsel
        NCONV = np.zeros(self.NGEOM,dtype='int32')
        NCONV[:] = self.NCONV[IGEOM]
        self.NCONV = NCONV
        
        VCONV = np.zeros((NCONV.max(),self.NGEOM))
        MEAS = np.zeros((NCONV.max(),self.NGEOM))
        ERRMEAS = np.zeros((NCONV.max(),self.NGEOM))
        for i in range(self.NGEOM):
            VCONV[0:NCONV[i],i] = self.VCONV[0:NCONV[i],IGEOM[i]]
            MEAS[0:NCONV[i],i] = self.MEAS[0:NCONV[i],IGEOM[i]]
            ERRMEAS[0:NCONV[i],i] = self.ERRMEAS[0:NCONV[i],IGEOM[i]]
        
        self.edit_VCONV(VCONV)
        self.edit_MEAS(MEAS)
        self.edit_ERRMEAS(ERRMEAS)
        
        NAV = np.zeros(self.NGEOM,dtype='int32')
        NAV[:] = self.NAV[IGEOM]
        self.NAV = NAV
        
        FLAT = np.zeros((self.NGEOM,self.NAV.max()))
        FLON = np.zeros((self.NGEOM,self.NAV.max()))
        WGEOM = np.zeros((self.NGEOM,self.NAV.max()))
        for i in range(self.NGEOM):
            FLAT[i,0:NAV[i]] = self.FLAT[IGEOM[i],0:NAV[i]]
            FLON[i,0:NAV[i]] = self.FLON[IGEOM[i],0:NAV[i]]
            WGEOM[i,0:NAV[i]] = self.WGEOM[IGEOM[i],0:NAV[i]]
        self.edit_FLAT(FLAT)
        self.edit_FLON(FLON)
        self.edit_WGEOM(WGEOM)
        
        if isinstance(self.TANHE,np.ndarray)==True:
            TANHE = np.zeros((self.NGEOM,self.NAV.max()))
            for i in range(self.NGEOM):
                TANHE[i,0:NAV[i]] = self.TANHE[IGEOM[i],0:NAV[i]]
            self.edit_TANHE(TANHE)
            
        if isinstance(self.EMISS_ANG,np.ndarray)==True:
            EMISS_ANG = np.zeros((self.NGEOM,self.NAV.max()))
            for i in range(self.NGEOM):
                EMISS_ANG[i,0:NAV[i]] = self.EMISS_ANG[IGEOM[i],0:NAV[i]]
            self.edit_EMISS_ANG(EMISS_ANG)
            
        if isinstance(self.SOL_ANG,np.ndarray)==True:
            SOL_ANG = np.zeros((self.NGEOM,self.NAV.max()))
            for i in range(self.NGEOM):
                SOL_ANG[i,0:NAV[i]] = self.SOL_ANG[IGEOM[i],0:NAV[i]]
            self.edit_SOL_ANG(SOL_ANG)
            
        if isinstance(self.AZI_ANG,np.ndarray)==True:
            AZI_ANG = np.zeros((self.NGEOM,self.NAV.max()))
            for i in range(self.NGEOM):
                AZI_ANG[i,0:NAV[i]] = self.AZI_ANG[IGEOM[i],0:NAV[i]]
            self.edit_AZI_ANG(AZI_ANG)
            
        self.assess()
        
    #################################################################################################################
        
    def select_TANHE_SO(self,TANHE_min,TANHE_max):
    
        """
        Based on the information of the Measurement class, update it based on selected tangent heights
        (Applicable to Solar Occultation measurements)
        """

        #Selecting the tangent heights
        iTANHE = np.where( (self.TANHE[:,0]>=TANHE_min) & (self.TANHE[:,0]<=TANHE_max) )[0]

        #Defining arrays
        ngeom = len(iTANHE)
        nav = 1 #it needs to be generalized to read more than one NAV per observation geometry
        nconv = np.zeros([ngeom],dtype='int')
        flat = np.zeros([ngeom,nav])
        flon = np.zeros([ngeom,nav])
        tanhe = np.zeros([ngeom,nav])
        wgeom = np.zeros([ngeom,nav])
        emiss_ang = np.zeros((ngeom,nav))
        wavetmp = np.zeros([self.NCONV.max(),ngeom])
        meastmp = np.zeros([self.NCONV.max(),ngeom])
        errmeastmp = np.zeros([self.NCONV.max(),ngeom])

        #Filling arrays
        nconv[:] = self.NCONV[iTANHE]
        flat[:,:] = self.FLAT[iTANHE,:]
        flon[:,:] = self.FLON[iTANHE,:]
        tanhe[:,:] = self.TANHE[iTANHE,:]
        wgeom[:,:] = self.WGEOM[iTANHE,:]
        emiss_ang[:,:] = self.EMISS_ANG[iTANHE,:]
        wavetmp[:,:] = self.VCONV[:,iTANHE]
        meastmp[:,:] = self.MEAS[:,iTANHE]
        errmeastmp[:,:] = self.ERRMEAS[:,iTANHE]

        #Updating class
        self.NGEOM = ngeom
        self.NCONV = nconv
        self.edit_FLAT(flat)
        self.edit_FLON(flon)
        self.edit_TANHE(tanhe)
        self.edit_WGEOM(wgeom)
        self.edit_EMISS_ANG(emiss_ang)
        self.edit_VCONV(wavetmp)
        self.edit_MEAS(meastmp)
        self.edit_ERRMEAS(errmeastmp)

    #################################################################################################################

    def wavesetc(self,Spectroscopy,IGEOM=0):
        """
        Subroutine to calculate which 'calculation' wavelengths are needed to 
        cover the required 'convolution wavelengths' (In case of line-by-line calculation).

        Parameters
        ----------
        Spectroscopy : class 
            Spectroscopy class indicating the grid of calculation wavelengths
        IGEOM : int, optional
            Integer defining the geometry at which the calculation numbers will be computed
        """

        if Spectroscopy is not None:

            if self.FWHM>0.0:

                if self.ISHAPE==0:
                    dv = 0.5*self.FWHM
                elif self.ISHAPE==1:
                    dv = self.FWHM
                elif self.ISHAPE==2:
                    dv = 3.* 0.5 * self.FWHM / np.sqrt(np.log(2.0))
                else:
                    dv = 3.*self.FWHM

                wavemin = self.VCONV[0,IGEOM] - dv
                wavemax = self.VCONV[self.NCONV[IGEOM]-1,IGEOM] + dv

            elif self.FWHM<=0.0:

                wavemin = 1.0e10
                wavemax = 0.0
                for i in range(self.NCONV[IGEOM]):
                    vminx = self.VFIL[0,i]
                    vmaxx = self.VFIL[self.NFIL[i]-1,i]
                    if vminx<wavemin:
                        wavemin = vminx
                    if vmaxx>wavemax:
                        wavemax= vmaxx

            #Correcting the wavelengths for Doppler shift
            print('nemesis :: Correcting for Doppler shift of ',self.V_DOPPLER,'km/s')        
            wavemin = self.invert_doppler_shift(wavemin)
            wavemax = self.invert_doppler_shift(wavemax)
            
            #Sorting the wavenumbers if the ILS is flipped
            if wavemin>=wavemax:
                sys.exit('error in wavesetc :: the spectral points defining the instrument lineshape must be increasing')

            #Checking that the lbl-tables encompass this wavelength range
            err = 0.001
            if (wavemin<(1-err)*Spectroscopy.WAVE.min() or wavemax>(1+err)*Spectroscopy.WAVE.max()):
                print('Required wavelength range :: ',wavemin,wavemax)
                print('Wavelength range in lbl-tables :: ',Spectroscopy.WAVE.min(),Spectroscopy.WAVE.max())
                sys.exit('error from wavesetc :: Channel wavelengths not covered by lbl-tables')


            #Selecting the necessary wavenumbers
            iwave = np.where( (Spectroscopy.WAVE>=wavemin) & (Spectroscopy.WAVE<=wavemax) )
            iwave = iwave[0]
            
            #Adding two more points to avoid problems with edges
            iwavex = np.zeros(len(iwave)+2,dtype='int32')
            if iwave[0]>0:
                iwavex[0] = iwave[0] - 1
            if iwave[len(iwave)-1]<Spectroscopy.NWAVE-1:
                iwavex[len(iwave)+1] = iwave[len(iwave)-1] + 1
            else:
                iwavex[len(iwave)+1] = Spectroscopy.NWAVE-1
            iwavex[1:len(iwave)+1] = iwave[:]

            iwavex = np.unique(iwavex)
            
            self.WAVE = Spectroscopy.WAVE[iwave]
            self.NWAVE = len(self.WAVE)
            
        else:
            
            self.NWAVE = self.NCONV[IGEOM]
            self.WAVE = np.zeros(self.NWAVE)
            self.WAVE[:] = self.VCONV[0:self.NCONV[IGEOM],IGEOM]

    #################################################################################################################

    def wavesetb(self,Spectroscopy,IGEOM=0):
    
        """
        Subroutine to calculate which 'calculation' wavelengths are needed to 
        cover the required 'convolution wavelengths' (In case of correlated-k calculation).

        Parameters
        ----------
        Spectroscopy : class
            Spectroscopy class indicating the grid of calculation wavelengths
        IGEOM : int, optional
            Integer defining the geometry at which the calculation numbers will be computed
        """
        
        if Spectroscopy is not None:

            #if (vkstep < 0.0 or fwhm == 0.0):
            if self.FWHM==0:

                wave = np.zeros(self.NCONV[IGEOM])
                wave[:] = self.VCONV[0:self.NCONV[IGEOM],IGEOM]
                self.WAVE = wave
                self.NWAVE = self.NCONV[IGEOM]

            elif self.FWHM<0.0:

                wavemin = 1.0e10
                wavemax = 0.0
                for i in range(self.NCONV[IGEOM]):
                    vminx = self.VFIL[0,i]
                    vmaxx = self.VFIL[self.NFIL[i]-1,i]
                    if vminx<wavemin:
                        wavemin = vminx
                    if vmaxx>wavemax:
                        wavemax= vmaxx

                if (wavemin<Spectroscopy.WAVE.min() or wavemax>Spectroscopy.WAVE.max()):
                    sys.exit('error from wavesetc :: Channel wavelengths not covered by k-tables')

                #Selecting the necessary wavenumbers
                iwavemin = np.argmin(np.abs(Spectroscopy.WAVE,wavemin))
                wave0min = Spectroscopy.WAVE[iwavemin]
                iwavemax = np.argmin(np.abs(Spectroscopy.WAVE,wavemax))
                wave0max = Spectroscopy.WAVE[iwavemax]

                if wave0min>wavemin:
                    iwavemin = iwavemin - 1
                if wave0max<wavemax:
                    iwavemax = iwavemax + 1

                iwave = np.linspace(iwavemin,iwavemax,iwavemax-iwavemin+1,dtype='int32')

                self.WAVE = Spectroscopy.WAVE[iwave]
                self.NWAVE = len(self.WAVE)

            elif self.FWHM>0.0:

                dv = self.FWHM * 0.5
                wavemin = self.VCONV[0,IGEOM] - dv
                wavemax = self.VCONV[self.NCONV[IGEOM]-1,IGEOM] + dv

                if (wavemin<Spectroscopy.WAVE.min() or wavemax>Spectroscopy.WAVE.max()):
                    sys.exit('error from wavesetc :: Channel wavelengths not covered by k-tables')

                iwave = np.where( (Spectroscopy.WAVE>=wavemin) & (Spectroscopy.WAVE<=wavemax) )
                iwave = iwave[0]
                self.WAVE = Spectroscopy.WAVE[iwave]
                self.NWAVE = len(self.WAVE)

            else:
                sys.exit('error :: Measurement FWHM is not defined')

        else:
            
            wave = np.zeros(self.NCONV[IGEOM])
            wave[:] = self.VCONV[0:self.NCONV[IGEOM],IGEOM]
            self.WAVE = wave
            self.NWAVE = self.NCONV[IGEOM]

    #################################################################################################################

    def calc_doppler_shift(self,wave):
        """
        Subroutine to calculate the Doppler shift in wavenumber or wavelength units based on
        the Doppler velocity between the observed body and the observer. The formula is:
            
            shift = lambda - lambda_0 = lambda_0 * v / c
        
        V_DOPPLER is defined as positive if moving towards the observer and negative if moving away
        
        This function returns Delta_Wave, where:
            Delta_Wave = Wave * v / c
        """
        
        c = 299792458.0   #Speed of light (m/s)
        
        if self.ISPACE==0:
            #Wavenumber (cm-1)
            wave_shift = self.V_DOPPLER*1.0e3 / c * wave
        elif self.ISPACE==1:
            #Wavelength (um)
            wave_shift = -self.V_DOPPLER*1.0e3 / c * wave
        
        return wave_shift
    
    #################################################################################################################
    
    def invert_doppler_shift(self,wave):
        """
        Subroutine to calculate the Doppler shift in wavenumber or wavelength units based on
        the Doppler velocity between the observed body and the observer.
        
        Knowing the observed wavelength lambda, we want to calculate the non-shifted wavelength lambda_0.
        
         lambda - lambda_0 = lambda_0 * v /c -----> lambda_0 = lambda / (1 + v/c)
        
        V_DOPPLER is defined as positive if moving towards the observer and negative if moving away
        
        This function returns WAVE_0 (in walengths or wavenumbers, depending on ISPACE)
        """
        
        c = 299792458.0   #Speed of light (m/s)
        
        if self.ISPACE==0:
            #Wavenumber (cm-1)
            wave_0 = wave / (1.0-self.V_DOPPLER*1.0e3 / c)
        elif self.ISPACE==1:
            #Wavelength (um)
            wave_0 = wave / (1.0+self.V_DOPPLER*1.0e3 / c)
        
        return wave_0
    
    #################################################################################################################
    
    def correct_doppler_shift(self,wave_0):
        """
        Subroutine to calculate the Doppler shift in wavenumber or wavelength units based on
        the Doppler velocity between the observed body and the observer.
        
        Knowing the non-shifted wavelength lambda_0, we want to calculate the shifted wavelength lambda.
        
         lambda - lambda_0 = lambda_0 * v /c -----> lambda = lambda_0 * (1 + v/c)
        
        V_DOPPLER is defined as positive if moving towards the observer and negative if moving away
        
        This function returns WAVE_0 (in walengths or wavenumbers, depending on ISPACE)
        """
        
        c = 299792458.0   #Speed of light (m/s)
        
        if self.ISPACE==0:
            #Wavenumber (cm-1)
            wave = wave_0 * (1.0-self.V_DOPPLER*1.0e3 / c)
        elif self.ISPACE==1:
            #Wavelength (um)
            wave = wave_0 * (1.0+self.V_DOPPLER*1.0e3 / c)
        
        return wave
    
    #################################################################################################################
    
    def plot_SO(self):
        """
        Subroutine to make a summary plot of a solar occultation observation
        """

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig,ax1 = plt.subplots(1,1,figsize=(12,4))

        colormap = 'nipy_spectral'
        cmap = matplotlib.cm.get_cmap(colormap,100)
        cmin = self.TANHE[:,0].min()
        cmax = self.TANHE[:,0].max()

        for igeom in range(self.NGEOM):
            
            color = (self.TANHE[igeom,0]-cmin)/(cmax-cmin)
            
            #ax1.plot(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom],c=s_m.to_rgba([self.TANHE[igeom,0]]))
            im1 = ax1.plot(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom],color=cmap(color))

        if np.mean(self.VCONV)>30.:
            ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
        else:
            ax1.set_xlabel('Wavelength ($\mu$m)')
        ax1.set_ylabel('Transmission')
        ax1.set_title('Latitude = '+str(np.round(self.LATITUDE,1))+' - Longitude = '+str(np.round(self.LONGITUDE,1)))
        ax1.grid()
        
        # Create a ScalarMappable object
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])  # Set dummy array to create the colorbar

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar2 = plt.colorbar(sm,cax=cax,orientation='vertical')
        cbar2.set_label('Altitude (km)')
        
        # Update colorbar ticks based on TANHE values
        n = 10
        cbar_ticksx = np.linspace(0, 1, num=n)  # Adjust the number of ticks as needed
        cbar_ticks = np.linspace(cmin, cmax, num=n)  # Adjust the number of ticks as needed
        cbar2.set_ticks(cbar_ticksx)
        cbar2.set_ticklabels([f'{tick:.2f}' for tick in cbar_ticks])  # Adjust the formatting as needed

        
        plt.tight_layout()
        plt.show()
        
    #################################################################################################################

    def plot_nadir(self,subobs_lat=None,subobs_lon=None):
        """
        Subroutine to make a summary plot of a nadir-viewing observation

        Other Parameters
        ----------
        subobs_lat : float, optional
            Sub-observer latitude (degrees)
        subobs_lon : float, optional
            Sub-observer longitude (degrees)

        """

        from mpl_toolkits.basemap import Basemap
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        #Making a figure for each geometry
        for igeom in range(self.NGEOM):

            fig = plt.figure(figsize=(12,7))

            #Plotting the geometry
            ax1 = plt.subplot2grid((2,3),(0,0),rowspan=2,colspan=1)
            if((subobs_lat!=None) & (subobs_lon!=None)):
                map = Basemap(projection='ortho', resolution=None,
                    lat_0=subobs_lat, lon_0=subobs_lon)
            else:
                map = Basemap(projection='ortho', resolution=None,
                    lat_0=self.LATITUDE, lon_0=self.LONGITUDE)

            
            lats = map.drawparallels(np.linspace(-90, 90, 13))
            lons = map.drawmeridians(np.linspace(-180, 180, 13))

            if self.NAV[igeom]>1:
                im = map.scatter(self.FLON[igeom,:],self.FLAT[igeom,:],latlon=True,c=self.WGEOM[igeom,:])

                # create an axes on the right side of ax. The width of cax will be 5%
                # of ax and the padding between cax and ax will be fixed at 0.05 inch.
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes("bottom", size="5%", pad=0.15)
                cbar2 = plt.colorbar(im,cax=cax,orientation='horizontal')
                cbar2.set_label('Weight')
            else:
                im = map.scatter(self.FLON[igeom,:],self.FLAT[igeom,:],latlon=True)

            ax1.set_title('Geometry '+str(igeom+1))

            #Plotting the spectra in linear scale
            ax2 = plt.subplot2grid((2,3),(0,1),rowspan=1,colspan=2)
            ax2.fill_between(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom]-self.ERRMEAS[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom]+self.ERRMEAS[0:self.NCONV[igeom],igeom],alpha=0.3)
            ax2.plot(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom])

            ax2.grid()

            #Plotting the spectra in log scale
            ax3 = plt.subplot2grid((2,3),(1,1),rowspan=1,colspan=2,sharex=ax2)
            ax3.fill_between(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom]-self.ERRMEAS[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom]+self.ERRMEAS[0:self.NCONV[igeom],igeom],alpha=0.3)
            ax3.plot(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom])
            ax3.set_yscale('log')

            if np.mean(self.VCONV)>30.:
                ax3.set_xlabel('Wavenumber (cm$^{-1}$)')
                ax3.set_ylabel('Radiance (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)')
                ax2.set_ylabel('Radiance (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)')
            else:
                ax3.set_xlabel('Wavelength ($\mu$m)')
                ax3.set_ylabel('Radiance (W cm$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)')
                ax2.set_ylabel('Radiance (W cm$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)')

            ax3.grid()

            plt.tight_layout()
        plt.show()
        
    #################################################################################################################

    def plot_disc_averaging(self,subobs_lat=None,subobs_lon=None, colormap='cividis'):
        """
        Subroutine to make a summary plot of a disc averaging observation 

        Parameters
        ----------
        subobs_lat : float, optional
            Sub-observer latitude (degrees)
        subobs_lon : float, optional
            Sub-observer longitude (degrees)

        """

        from mpl_toolkits.basemap import Basemap
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        #Making a figure for each geometry
        for igeom in range(self.NGEOM):

            fig = plt.figure(figsize=(15,7))



            #Plotting the geometry
            ax1 = plt.subplot2grid((2,4),(0,0),rowspan=1,colspan=1)
            if((subobs_lat!=None) & (subobs_lon!=None)):
                map1 = Basemap(projection='ortho', resolution=None,
                    lat_0=subobs_lat, lon_0=subobs_lon)
            else:
                map1 = Basemap(projection='ortho', resolution=None,
                    lat_0=self.LATITUDE, lon_0=self.LONGITUDE)


            lats = map1.drawparallels(np.linspace(-90, 90, 13))
            lons = map1.drawmeridians(np.linspace(-180, 180, 13))
            im1 = map1.scatter(self.FLON[igeom,:],self.FLAT[igeom,:],latlon=True,c=self.WGEOM[igeom,:],cmap=colormap)

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("bottom", size="5%", pad=0.15)
            cbar1 = plt.colorbar(im1,cax=cax,orientation='horizontal')
            cbar1.set_label('Weight')






            ax2 = plt.subplot2grid((2,4),(0,1),rowspan=1,colspan=1)
            if((subobs_lat!=None) & (subobs_lon!=None)):
                map2 = Basemap(projection='ortho', resolution=None,
                    lat_0=subobs_lat, lon_0=subobs_lon)
            else:
                map2 = Basemap(projection='ortho', resolution=None,
                    lat_0=self.LATITUDE, lon_0=self.LONGITUDE)

            
            lats = map2.drawparallels(np.linspace(-90, 90, 13))
            lons = map2.drawmeridians(np.linspace(-180, 180, 13))
            im2 = map2.scatter(self.FLON[igeom,:],self.FLAT[igeom,:],latlon=True,c=self.EMISS_ANG[igeom,:],cmap=colormap)

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("bottom", size="5%", pad=0.15)
            cbar2 = plt.colorbar(im2,cax=cax,orientation='horizontal')
            cbar2.set_label('Emission angle')






            ax3 = plt.subplot2grid((2,4),(1,0),rowspan=1,colspan=1)
            if((subobs_lat!=None) & (subobs_lon!=None)):
                map3 = Basemap(projection='ortho', resolution=None,
                    lat_0=subobs_lat, lon_0=subobs_lon)
            else:
                map3 = Basemap(projection='ortho', resolution=None,
                    lat_0=self.LATITUDE, lon_0=self.LONGITUDE)

            
            lats = map3.drawparallels(np.linspace(-90, 90, 13))
            lons = map3.drawmeridians(np.linspace(-180, 180, 13))
            im3 = map3.scatter(self.FLON[igeom,:],self.FLAT[igeom,:],latlon=True,c=self.SOL_ANG[igeom,:],cmap=colormap)

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("bottom", size="5%", pad=0.15)
            cbar2 = plt.colorbar(im3,cax=cax,orientation='horizontal')
            cbar2.set_label('Solar Zenith angle')






            ax4 = plt.subplot2grid((2,4),(1,1),rowspan=1,colspan=1)
            if((subobs_lat!=None) & (subobs_lon!=None)):
                map4 = Basemap(projection='ortho', resolution=None,
                    lat_0=subobs_lat, lon_0=subobs_lon)
            else:
                map4 = Basemap(projection='ortho', resolution=None,
                    lat_0=self.LATITUDE, lon_0=self.LONGITUDE)

            
            lats = map4.drawparallels(np.linspace(-90, 90, 13))
            lons = map4.drawmeridians(np.linspace(-180, 180, 13))
            im4 = map4.scatter(self.FLON[igeom,:],self.FLAT[igeom,:],latlon=True,c=self.AZI_ANG[igeom,:],cmap=colormap)

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax4)
            cax = divider.append_axes("bottom", size="5%", pad=0.15)
            cbar2 = plt.colorbar(im4,cax=cax,orientation='horizontal')
            cbar2.set_label('Solar Zenith angle')
            
            #Plotting the spectra in linear scale
            ax5 = plt.subplot2grid((2,4),(0,2),rowspan=1,colspan=2)
            ax5.fill_between(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom]-self.ERRMEAS[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom]+self.ERRMEAS[0:self.NCONV[igeom],igeom],alpha=0.3)
            ax5.plot(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom])

            ax5.grid()

            #Plotting the spectra in log scale
            ax6 = plt.subplot2grid((2,4),(1,2),rowspan=1,colspan=2,sharex=ax5)
            ax6.fill_between(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom]-self.ERRMEAS[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom]+self.ERRMEAS[0:self.NCONV[igeom],igeom],alpha=0.3)
            ax6.plot(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom])
            ax6.set_yscale('log')

            if np.mean(self.VCONV)>30.:
                ax6.set_xlabel('Wavenumber (cm$^{-1}$)')
                ax6.set_ylabel('Radiance (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)')
                ax5.set_ylabel('Radiance (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)')
            else:
                ax6.set_xlabel('Wavelength ($\mu$m)')
                ax6.set_ylabel('Radiance (W cm$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)')
                ax5.set_ylabel('Radiance (W cm$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)')

            ax6.grid()
            ax5.set_title('Geometry '+str(igeom+1))

            plt.tight_layout()
        plt.show()
        
    #################################################################################################################