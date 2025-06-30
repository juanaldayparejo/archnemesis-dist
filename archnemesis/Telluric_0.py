#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Created on 25 Sep 2024

@author: juanaldayparejo

Telluric Class.
"""
from __future__ import annotations #  for 3.9 compatability
from archnemesis import *
import numpy as np

from archnemesis.enums import PlanetEnum, AtmosphericProfileFormatEnum, SpectraUnit, SpectralCalculationMode

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

class Telluric_0:
    """
    Class to define the telluric atmosphere and observing geometry from the Earth. 
    Relevant for the anaylsis and simulation of ground-based observations.
    """
    def __init__(self, DATE='01-01-2020', TIME='00:00:00', LATITUDE=19.82067, LONGITUDE=-155.46806, ALTITUDE=4207.3, EMISS_ANG=180.):
        """
        Inputs
        ------

        @param DATE: str
            UTC date of the observations (dd-mm-yyyy)
        @param TIME: str
            UTC time of the observations (hh:mm:ss)
        @param LATITUDE: real
            Planetocentric latitude of the observatory (degrees)
        @param LONGITUDE: real 
            Planetocentric longitude of the observatory (degrees)
        @param ALTITUDE: real,
            Altitude of the observatory (m)
        @param EMISS_ANG: real,
            Emission angle of the observation. Must be between 90 and 180
            i.e. 90 is looking towards the horizon
            i.e. 180 is looking upwards towards the zenith

        Attributes
        ----------
        @attribute Atmosphere: archNEMESIS class
            Atmosphere class of archNEMESIS to define the telluric atmosphere
        @attribute Spectroscopy: archNEMESIS class
            Spectroscopy class of archNEMESIS to define the telluric absorption
        

        Methods
        -------

        """
        self.DATE = DATE
        self.TIME = TIME
        self.LATITUDE = LATITUDE
        self.LONGITUDE = LONGITUDE
        self.ALTITUDE = ALTITUDE
        self.EMISS_ANG = EMISS_ANG

        self.Spectroscopy = None   #Spectroscopy class inside the Telluric class
        self.Atmosphere = None   #Atmosphere class inside the Telluric class

  
    ##################################################################################

    def write_hdf5(self,runname):
        """
        Function to write the HDF5 file for the Telluric class
        """
        
        import h5py
        from archnemesis.Data.gas_data import gas_info
        from archnemesis.Data.planet_data import planet_info

        #Assessing that all the parameters have the correct type and dimension
        #self.assess()

        f = h5py.File(runname+'.h5','a')
        #Checking if Atmosphere already exists
        if ('/Telluric' in f)==True:
            del f['Telluric']   #Deleting the Telluric information that was previously written in the file

        grp = f.create_group("Telluric")
        
        #Writing the parameters about the observation and observatory
        dt = h5py.special_dtype(vlen=str)
        dset = grp.create_dataset('DATE',data=self.DATE,dtype=dt)
        dset.attrs['title'] = "UTC date of the observation (DD-MM-YYYY)"
        
        dset = grp.create_dataset('TIME',data=self.TIME,dtype=dt)
        dset.attrs['title'] = "UTC time of the observation (HH:MM:SS)"
        
        dset = grp.create_dataset('LATITUDE',data=self.LATITUDE)
        dset.attrs['title'] = "Latitude of the observatory (degrees)"
        
        dset = grp.create_dataset('LONGITUDE',data=self.LONGITUDE)
        dset.attrs['title'] = "Longitude of the observatory (degrees)"
        
        dset = grp.create_dataset('ALTITUDE',data=self.ALTITUDE)
        dset.attrs['title'] = "Altitude of the observatory (metres)"
        
        dset = grp.create_dataset('EMISS_ANG',data=self.EMISS_ANG)
        dset.attrs['title'] = "Emission angle (degrees)"
        
        f.close()
        
        #Writing the Atmosphere
        if self.Atmosphere is None:
            raise ValueError('error in write_hdf5 :: Telluric.Atmosphere must be defined')   
        else:
            self.Atmosphere.write_hdf5(runname,inside_telluric=True)
                
        #Writing the spectroscopy
        if self.Spectroscopy is None:
            raise ValueError('error in write_hdf5 :: Telluric.Spectroscopy must be defined')   
        else:
            self.Spectroscopy.write_hdf5(runname,inside_telluric=True)
  
        
    ##################################################################################

    def read_hdf5(self,runname):
        """
        Function to read the HDF5 file and extract the information about the Telluric class
        """
        
        from archnemesis import Atmosphere_0, Spectroscopy_0
        
        f = h5py.File(runname+'.h5','r')

        #Checking if Telluric exists
        name = '/Telluric'
        e = name in f
        if e==False:
            f.close()
            raise ValueError('error :: Telluric is not defined in HDF5 file')
        else:

            grp = f[name]

            self.DATE = grp['DATE'][()].decode('ascii')
            self.TIME = grp['TIME'][()].decode('ascii')
            
            self.LATITUDE = np.float32(f.get(name+'/LATITUDE'))
            self.LONGITUDE = np.float32(f.get(name+'/LONGITUDE'))
            self.ALTITUDE = np.float32(f.get(name+'/ALTITUDE'))
            self.EMISS_ANG = np.float32(f.get(name+'/EMISS_ANG'))
            
            self.Atmosphere = Atmosphere_0()
            self.Atmosphere.read_hdf5(runname,inside_telluric=True)
            
            self.Spectroscopy = Spectroscopy_0()
            self.Spectroscopy.read_hdf5(runname,inside_telluric=True)
        
            f.close()

        
        
    ##################################################################################

    def extract_atmosphere_era5(self):
        """
        Function to extract information about the Earth's atmosphere at our desired location and time from the ERA5 reanalysis model
        
        
        Information
        ____________
        
        Information about this model can be read from: https://cds-beta.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview
        The parameters we read from the ERA5 model are: Pressure, temperature, water vapour mixing ratio, ozone mixing ratio
        
        Information about how to setup the connection to the Climate Data Store (CDS) where the ERA5 data is stored can be found in:
        https://cds.climate.copernicus.eu/api-how-to
        
        For the rest of the species included in this function (CO2,N2O,CO,CH4,O2), we use some reference profiles from CIRC (https://earth.gsfc.nasa.gov/climate/models/circ/cases)
        """
        
        from datetime import datetime, timedelta
        from archnemesis.Data.path_data import archnemesis_path 
        from archnemesis import Atmosphere_0
        from archnemesis.Data.gas_data import const
        import cdsapi,pygrib
        
        #Defining the inputs
        ##############################################################################################################
        
        date = self.DATE
        time = self.TIME
        latitude = self.LATITUDE
        longitude = self.LONGITUDE
        
        #Defining the arrays as they are defined in the ERA5 model
        ##############################################################################################################
        
        presslevels = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000])
        lats = np.arange(-90.,90.+0.25,0.25)
        lons = np.arange(-180.,180.+0.25,0.25)

        nlevels = len(presslevels)
        presslevels_str = []
        for i in range(nlevels):
            presslevels_str.append(str(presslevels[i]))
            
        #Reading the parameters of the desired location and defining the values on the grid to fetch
        ##############################################################################################################

        #Finding the index of the arrays that we need
        ilat = np.argmin(np.abs(latitude-lats))
        ilon = np.argmin(np.abs(longitude-lons))

        ilats = np.zeros(2,dtype='int32')
        ilons = np.zeros(2,dtype='int32')
        if lats[ilat]>latitude:
            ilats[0] = ilat - 1
            ilats[1] = ilat
        else:
            ilats[0] = ilat
            ilats[1] = ilat + 1

        if lons[ilon]>longitude:
            ilons[0] = ilon - 1
            ilons[1] = ilon
        else:
            ilons[0] = ilon
            ilons[1] = ilon + 1

        lats_sel = lats[ilats]
        lons_sel = lons[ilons]
        
        #Finding the closest hour at which to extract the data (resolution of ERA5 is 1 hour)
        ##############################################################################################################
        
        datetime_str = f'{date} {time}'
        
        # Define the format in which the date and time are provided
        datetime_format = '%d-%m-%Y %H:%M:%S'
        
        # Convert the string to a datetime object
        dt = datetime.strptime(datetime_str, datetime_format)
        
        # Check if we need to round up or down based on the minute
        if dt.minute >= 30:
            # Round up to the next hour
            dt = dt + timedelta(hours=1)
            
        # Set minutes and seconds to 0 to get the nearest hour (resolution of ERA5)
        dt = dt.replace(minute=0, second=0, microsecond=0)
        
        # Convert the rounded datetime back to a string
        rounded_datetime_str = dt.strftime(datetime_format)

        day = rounded_datetime_str[0:2] ; month = rounded_datetime_str[3:5] ; year = rounded_datetime_str[6:10]
        time = rounded_datetime_str[11:16]
            
        #Extracting the pressure-dependent data
        ##########################################################################################

        dataset = "reanalysis-era5-pressure-levels"
        request = {
            'product_type': ['reanalysis'],
            'variable': ['fraction_of_cloud_cover', 'ozone_mass_mixing_ratio', 'specific_cloud_liquid_water_content', 'temperature', 'specific_humidity'],
            'year': year,
            'month': month,
            'day': day,
            'time': time,
            'pressure_level': presslevels_str,
            'data_format': 'grib',
            'download_format': 'unarchived',
            'area': [lats_sel[0], lons_sel[0], lats_sel[1], lons_sel[1]]
        }
        
        client = cdsapi.Client()
        client.retrieve(dataset, request, "download.grib")
        isort = np.argsort(presslevels)[::-1]  #Array from max pressure to min pressure
        press = presslevels[isort] * 100.   #Pa
        temp = extract_grib_parameter('download.grib','Temperature',latitude,longitude)[isort]
        specific_humidity = extract_grib_parameter('download.grib','Specific humidity',latitude,longitude)[isort]
        ozone_mmr = extract_grib_parameter('download.grib','Ozone mass mixing ratio',latitude,longitude)[isort]
        os.remove('download.grib')

        #Calculating the hydrostatic altitudes
        ###########################################################################################

        #First estimation of the altitudes
        g0 = 9.80665 #m/s2
        mmol = 0.0289644 #kg/mol
        R = const["R"]
        
        sh = R * temp / (mmol * g0)
        h = -np.log(press/press[0]) * sh
        
        #Calculating the VMRs of H2O and O3
        ############################################################################################
        
        # Calculate water vapour mixing ratio (w = q / (1 - q))
        vmr_h2o = specific_humidity / (1 - specific_humidity)
    
        #Converting the mass mixing ratio of O3 to the volue mixing ratio
        vmr_o3 = ozone_mmr / 0.048 * mmol
        
        #Reading the VMRs for CH4,CO2,CO and N2O
        ############################################################################################
        
        Atmosphere_CIRC = Atmosphere_0(runname=archnemesis_path()+'archnemesis/Data/reference_profiles/earth_circ_case1')
        Atmosphere_CIRC.read_ref()
        
        ico2 = np.where(Atmosphere_CIRC.ID==2)[0][0]
        vmr_co2 = np.interp(press,Atmosphere_CIRC.P[::-1],Atmosphere_CIRC.VMR[:,ico2][::-1])
        
        in2o = np.where(Atmosphere_CIRC.ID==4)[0][0]
        vmr_n2o = np.interp(press,Atmosphere_CIRC.P[::-1],Atmosphere_CIRC.VMR[:,in2o][::-1])
        
        ico = np.where(Atmosphere_CIRC.ID==5)[0][0]
        vmr_co = np.interp(press,Atmosphere_CIRC.P[::-1],Atmosphere_CIRC.VMR[:,ico][::-1])
        
        ich4 = np.where(Atmosphere_CIRC.ID==6)[0][0]
        vmr_ch4 = np.interp(press,Atmosphere_CIRC.P[::-1],Atmosphere_CIRC.VMR[:,ich4][::-1])
        
        io2 = np.where(Atmosphere_CIRC.ID==7)[0][0]
        vmr_o2 = np.interp(press,Atmosphere_CIRC.P[::-1],Atmosphere_CIRC.VMR[:,io2][::-1])
        
        in2 = np.where(Atmosphere_CIRC.ID==22)[0][0]
        vmr_n2 = np.interp(press,Atmosphere_CIRC.P[::-1],Atmosphere_CIRC.VMR[:,in2][::-1])
        
        #Defining the Atmosphere class with the ERA5 profiles
        ##############################################################################################
        
        Atmosphere = Atmosphere_0()
        Atmosphere.IPLANET  : PlanetEnum = PlanetEnum.Earth
        Atmosphere.LATITUDE = latitude
        Atmosphere.LONGITUDE = longitude
        Atmosphere.AMFORM : AtmosphericProfileFormatEnum = AtmosphericProfileFormatEnum.MOLECULAR_WEIGHT_DEFINED
        Atmosphere.NVMR = 8
        Atmosphere.ID = np.array([1,2,3,4,5,6,7,22],dtype='int32')
        Atmosphere.ISO = np.zeros(Atmosphere.NVMR,dtype='int32')
        Atmosphere.NP = nlevels
        Atmosphere.edit_H(h)
        Atmosphere.edit_P(press) #Pa
        Atmosphere.edit_T(temp)
        Atmosphere.MOLWT = np.ones(nlevels) * mmol
        Atmosphere.edit_VMR(np.zeros((Atmosphere.NP,Atmosphere.NVMR)))
        Atmosphere.VMR[:,0] = vmr_h2o[:]
        Atmosphere.VMR[:,1] = vmr_co2[:]
        Atmosphere.VMR[:,2] = vmr_o3[:]
        Atmosphere.VMR[:,3] = vmr_n2o[:]
        Atmosphere.VMR[:,4] = vmr_co[:]
        Atmosphere.VMR[:,5] = vmr_ch4[:]
        Atmosphere.VMR[:,6] = vmr_o2[:]
        Atmosphere.VMR[:,7] = vmr_n2[:]
        
        Atmosphere.NDUST = 1
        Atmosphere.edit_DUST(np.zeros((Atmosphere.NP,Atmosphere.NDUST)))
     
        Atmosphere.calc_grav()
        Atmosphere.calc_radius()
        Atmosphere.adjust_hydrostatH()
        
        self.Atmosphere = Atmosphere
        
    ##################################################################################
        
    def extract_atmosphere_circ(self):
        """
        Function to extract information about the Earth's atmosphere at our desired location and time from the CIRC reference profiles.
        The reference atmosphere can be downloaded from: https://earth.gsfc.nasa.gov/climate/models/circ/cases
        """
    
        from archnemesis import Atmosphere_0
        from archnemesis.Data.path_data import archnemesis_path 
    
        #Reading the VMRs for CH4,CO2,CO and N2O
        ############################################################################################
        
        Atmosphere_CIRC = Atmosphere_0(runname=archnemesis_path()+'archnemesis/Data/reference_profiles/earth_circ_case1')
        Atmosphere_CIRC.read_ref()
        
        self.Atmosphere = Atmosphere_CIRC
    
    ##################################################################################
    
    def calc_transmission(self):
        """
        Function to calculate the line-of-sight densities for each gas based on an input telluric atmosphere
        """
    
        from archnemesis import Layer_0, ForwardModel_0, Scatter_0, Measurement_0
        from archnemesis import k_overlap
        
        #Adding zero dust in case it does not exist
        self.Atmosphere.NDUST = 1
        self.Atmosphere.edit_DUST(np.zeros((self.Atmosphere.NP,self.Atmosphere.NDUST)))
        
        #Calculating the Layering of the atmosphere
        Layer = Layer_0()
        Layer.RADIUS = self.Atmosphere.RADIUS
        Layer.LAYHT = self.ALTITUDE
        Layer.NLAY = 31
        Layer.LAYTYP=2
        Layer.LAYANG=0.
        Layer.calc_layering(H=self.Atmosphere.H,P=self.Atmosphere.P,T=self.Atmosphere.T, ID=self.Atmosphere.ID,VMR=self.Atmosphere.VMR, DUST=self.Atmosphere.DUST, PARAH2=self.Atmosphere.PARAH2)
    
        #Defining extra classes for geometry
        Scatter = Scatter_0()
        Scatter.ISCAT = 0   #No scattering
        
        assert self.EMISS_ANG > 90. , \
            'EMISS_ANG must be >90 and <=180 (90 is looking towards horizon and 180 is looking up towards zenith)'
        assert self.EMISS_ANG <= 180. , \
            'EMISS_ANG must be >90 and <=180 (90 is looking towards horizon and 180 is looking up towards zenith)'
        
        Scatter.EMISS_ANG = self.EMISS_ANG
        Scatter.SOL_ANG = 0.
        Scatter.AZI_ANG = 0.
        Measurement = Measurement_0()
        Measurement.IFORM = SpectraUnit.Radiance
        
        #Calculating the path
        FM = ForwardModel_0()
        FM.calc_path(Atmosphere=self.Atmosphere,Scatter=Scatter,Layer=Layer,Measurement=Measurement)
    
        #Calculating the line-of-sight column density for each gas
        amounts = (np.transpose(Layer.AMOUNT[FM.PathX.LAYINC[:,:],:],axes=(2,0,1)) * FM.PathX.SCALE[:,:])[:,:,0] #N_col density in each layer for each gas (NVMR,NLAY)
        tlay = Layer.TEMP[FM.PathX.LAYINC[:,:]][:,0]  #(NLAY)
        play = Layer.PRESS[FM.PathX.LAYINC[:,:]][:,0]  #(NLAY)
        
        #Calculating the optical depth along the line-of-sight
        ########################################################################################################
        if self.Spectroscopy.ILBL==SpectralCalculationMode.LINE_BY_LINE_TABLES:  #LBL-table

            #Calculating the cross sections for each gas in each layer
            k = self.Spectroscopy.calc_klbl(len(tlay),play/101325.,tlay,WAVECALC=self.Spectroscopy.WAVE)

            TAUGAS = np.zeros((self.Spectroscopy.NWAVE,self.Spectroscopy.NG,len(tlay),self.Spectroscopy.NGAS))  #Vertical opacity of each gas in each layer

            for i in range(self.Spectroscopy.NGAS):
                igas = np.where( (self.Atmosphere.ID==self.Spectroscopy.ID[i]) & (self.Atmosphere.ISO==self.Spectroscopy.ISO[i]) )[0][0]

                #Calculating vertical column density in each layer
                VLOSDENS = amounts[igas,:] * 1.0e-4 * 1.0e-20   #cm-2

                #Calculating vertical opacity for each gas in each layer
                TAUGAS[:,0,:,i] = k[:,:,i] * VLOSDENS

            #Combining the gaseous opacity in each layer
            TAUGAS = np.sum(TAUGAS,3) #(NWAVE,NG,NLAY)
            #Removing necessary data to save memory
            del k

        elif self.Spectroscopy.ILBL==SpectralCalculationMode.K_TABLES:    #K-table
            
            #Calculating the k-coefficients for each gas in each layer
            k_gas = self.Spectroscopy.calc_k(len(tlay),play/101325.,tlay,WAVECALC=self.Spectroscopy.WAVE) # (NWAVE,NG,NLAY,NGAS)
            
            f_gas = np.zeros((self.Spectroscopy.NGAS,len(tlay)))
            utotl = np.zeros(len(tlay))
            for i in range(self.Spectroscopy.NGAS):
                igas = np.where( (self.Atmosphere.ID==self.Spectroscopy.ID[i]) & (self.Atmosphere.ISO==self.Spectroscopy.ISO[i]) )[0][0]
                f_gas[i,:] = amounts[igas,:] * 1.0e-4 * 1.0e-20  #Vertical column density of the radiatively active gases in cm-2

            #Combining the k-distributions of the different gases in each layer
            k_layer = k_overlap(self.Spectroscopy.DELG,k_gas,f_gas)

            #Calculating the opacity of each layer
            TAUGAS = k_layer #(NWAVE,NG,NLAY)

            #Removing necessary data to save memory
            del k_gas
            del k_layer

        else:
            raise ValueError('error in CIRSrad :: ILBL must be either 0 or 2')

        #Calculating the atmospheric transmission
        ###################################################################
        
        TAUTOT = np.sum(TAUGAS,axis=2)  #(NWAVE,NG)
        TRANS = np.exp(-TAUTOT)
        
        #Integrating over the g-ordinates
        ###################################################################
        
        SPECOUT = np.tensordot(TRANS, self.Spectroscopy.DELG, axes=([1],[0])) #NWAVE

        return self.Spectroscopy.WAVE,SPECOUT
    
    
    ##################################################################################

    def edit_PLAY(self, array):
        """
        Edit the Pressure profile for each atmospheric layer.
        @param P_array: 1D
            Pressures of the vertical points in Pa
        """
        array = np.array(array)

        assert len(array) == self.NLAY, 'PLAY should have NLAY elements'

        self.PLAY = array

    ##################################################################################

    def edit_TLAY(self, array):
        """
        Edit the Temperature profile for each atmospheric layer.
        @param T_array: 1D
            Temperature of the vertical points in K
        """
        array = np.array(array)

        assert len(array) == self.NLAY, 'TLAY should have NLAY elements'

        self.TLAY = array

    ##################################################################################

    def edit_NCOL(self, array):
        """
        Edit the line-of-sight column density of each gas in each layer
        @param NCOL_array: 2D
            NVMR by NLAY array containing the line-of-sight column density of gases in m-2.
            NCOL_array[i,j] is the line-of-sight column density of gas i at layer j.
        """
        array = np.array(array)

        assert array.shape == (self.NVMR,self.NLAY), 'NCOL should have (NVMR,NLAY) elements'

        self.NCOL = array
    
    
############################################################################################
############################################################################################
############################################################################################


def extract_grib_parameter(filename,parameter,latitude,longitude):

    """
        FUNCTION NAME : extract_grib_parameter()
        
        DESCRIPTION :
        
            Function read a parameter from the .grib file generated from the ERA5 model
            and interpolate it to the correct location
        
        INPUTS :
        
            filename :: Name of the .grib file
            parameter_str :: Name of the parameter to extract from the file
            latitude :: Latitutde at which the parameter wants to be interpolated to (degrees)
            longitude :: Longitude at which the parameter wants to be interpolated to (degrees)
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            parameter_int :: Interpolated value of the parameter 
        
        CALLING SEQUENCE:
        
            parameter_int = extract_grib_parameter(filename,parameter,latitude,longitude)
        
        MODIFICATION HISTORY : Juan Alday (29/09/2024)
        
    """
    
    import pygrib

    # Open the GRIB file and read the information
    #########################################################################
    
    grbs = pygrib.open(filename)
    
    # List to store temperature data at different levels
    param_data_list = []
    latitudes = None
    longitudes = None
    
    # Loop through GRIB messages and extract parameter data for each pressure level
    for grb in grbs:
        #_lgr.info(grb.name)
        if grb.name == parameter:  # Ensure it's the parameter we want
            if latitudes is None and longitudes is None:
                # Get latitude and longitude grids (these will be the same for each level)
                latitudes, longitudes = grb.latlons()
            
            # Extract the temperature data for this level
            param_data = grb.values  # 2D array for this level
            param_data_list.append(param_data)  # Add to the list
    
    # Convert the list of 2D arrays into a 3D NumPy array (stack along the third axis)
    param_3d_array = np.stack(param_data_list, axis=-1)  # Shape: (lat, lon, levels)
    
    # Close the GRIB file
    grbs.close()


    #Interpolate the parameter to the desired latitude and longitude
    ############################################################################

    lat1 = latitudes[0,0] ; lat2 = latitudes[1,0]
    lon1 = longitudes[0,0] ; lon2 = longitudes[1,1]

    u = (latitude-lat1)/(lat2-lat1)
    v = (longitude-lon1)/(lon2-lon1)

    if u>1.:
        raise ValueError('error in the interpolation between latitudes')
    if v>1.:
        raise ValueError('error in the interpolation between longitudes')
    
    param_lat1_lon1 = param_3d_array[0,0,:]
    param_lat1_lon2 = param_3d_array[0,1,:]
    param_lat2_lon1 = param_3d_array[1,0,:]
    param_lat2_lon2 = param_3d_array[1,1,:]

    param_int = (1.0-v)*(1.0-u)*param_lat1_lon1[:] + \
               v*(1.0-u)*param_lat1_lon2[:] + \
               u*(1.0-v)*param_lat2_lon1[:] + \
               u*v*param_lat2_lon2[:]

    return param_int