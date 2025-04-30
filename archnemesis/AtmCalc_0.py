#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np

from archnemesis.enums import ZenithAngleOrigin, PathObserverPointing, PathCalc

"""
Object to calculate the atmospheric paths
"""


'''
class ZenithAngleOrigin(IntEnum):
    """
    Defines values for 'IPZEN'

    Used in 'AtmCalc_0.py'
    """
    BOTTOM = 0 # Zenith angle is defined at the bottom of the bottom layer
    ALTITUDE_ZERO = 1 # Zenith angle is defined at 0km atltitude
    TOP = 2 # Zenith angle is defined at the top of the top layer


class PathObserverPointing(IntEnum):
    """
    Defines location of the PathObserver, used when calculating radiative transfer.

    Used in 'AtmCalc_0.py'
    """
    LIMB = 0 # Limb path, path observer is looking at the limb of the planet
    NADIR = 1 # Nadir path, path observer is on the planet looking upwards
    DISK = 2 # Disk path, path observer is looking at the disk of the planet


class PathCalc(IntFlag):
    """
    Defines path calculation type used when calculating radiative transfer.
    
    Used as elements of 'IMOD' in 'AtmCalc_0.py', 'Path_0.py', 'ForwardModel_0.py'
    """
    WEIGHTING_FUNCTION = auto()
    NET_FLUX = auto()
    UPWARD_FLUX = auto()
    OUTWARD_FLUX = auto()
    DOWNWARD_FLUX = auto()
    CURTIS_GODSON = auto()
    THERMAL_EMISSION = auto()
    HEMISPHERE = auto()
    SCATTER = auto()
    NEAR_LIMB = auto()
    SINGLE_SCATTERING_PLANE_PARALLEL = auto()
    SINGLE_SCATTERING_SPHERICAL = auto()
    ABSORBTION = auto()
    PLANK_FUNCTION_AT_BIN_CENTRE = auto()
    BROADENING = auto()
'''


class AtmCalc_0:
    def __init__(
            self,
            Layer,
            path_observer_pointing=PathObserverPointing.DISK,
            #LIMB=False,
            #NADIR=False,
            BOTLAY=0,
            ANGLE=0.0,
            EMISS_ANG=0.0,
            SOL_ANG=0.0,
            AZI_ANG=0.0,
            
            IPZEN = ZenithAngleOrigin.BOTTOM,
            
            path_calc = PathCalc.PLANK_FUNCTION_AT_BIN_CENTRE,
            
            #WF=False,
            #NETFLUX=False,
            #OUTFLUX=False,
            #BOTFLUX=False,
            #UPFLUX=False,
            #CG=False,
            #THERM=False,
            #HEMISPHERE=False,
            #NEARLIMB=False,
            #SINGLE=False,
            #SPHSINGLE=False,
            #SCATTER=False,
            #BROAD=False,
            #ABSORB=False,
            #BINBB=True
        ):
        """
            After splitting the atmosphere in different layers and storing them in the Layer class,
            the atmospheric paths are calculated.
            Inputs
            ------
            @param Layer: class
                Python class including all required information about the different atmospheric layers
            @param LIMB: log
                Flag indicating whether it is a limb path. If True, then the attribute BOTLAY must be
                filled with the bottom layer of the path
            @param NADIR: log
                Flag indicating whether it is a nadir path. If True, then the attribute BOTLAY must be
                filled with the bottom layer to use (typically 0) and VIEW_ANG must be filled with the angle
                from the nadir 
            @param BOTLAY: real
                Bottom layer to use in the calculation of the path            
            @param ANGLE: real
                Observing angle from nadir (deg). Note that more than 90deg is looking upwards
            @param EMISS_ANG: real
                Observing angle from nadir (deg). Note that more than 90deg is looking upwards
            @param SOL_ANG: real
                Solar zenith angle (deg). Note that 0 is at zenith ang >90 is below the horizon
            @param AZI_ANG: real
                Azimuth angle (deg). Note that 0 is forward scattering
            @param IPZEN: int
                Flag defining where the zenith angle is defined. 
                0 = at bottom of bottom layer. 
                1 = at the 0km altitude level.
                2 = at the very top of the atmosphere.
            @param WF: log
                Flags indicating the type of calculation to be performed: Weighting function
            @param NETFLUX: log
                Flags indicating the type of calculation to be performed: Net flux calculation
            @param UPFLUX: log
                Flags indicating the type of calculation to be performed: Internal upward flux calculation
            @param OUTFLUX: log
                Flags indicating the type of calculation to be performed: Upward flux at top of topmost layer
            @param BOTFLUX: log
                Flags indicating the type of calculation to be performed: Downward flux at bottom of lowest layer
            @param CG: log
                Flags indicating the type of calculation to be performed: Curtis Godson
            @param THERM: log
                Flags indicating the type of calculation to be performed: Thermal emission
            @param HEMISPHERE: log
                Flags indicating the type of calculation to be performed: Integrate emission into hemisphere
            @param SCATTER: log
                Flags indicating the type of calculation to be performed: Full scattering calculation
            @param NEARLIMB: log
                Flags indicating the type of calculation to be performed: Near-limb scattering calculation
            @param SINGLE: log
                Flags indicating the type of calculation to be performed: Single scattering calculation (plane parallel)
            @param SPHSINGLE: log
                Flags indicating the type of calculation to be performed: Single scattering calculation (spherical atm.)
            @param ABSORB: log
                Flags indicating the type of calculation to be performed: calculate absorption not transmission
            @param BINBB: log
                Flags indicating the type of calculation to be performed: use planck function at bin centre in genlbl
            @param BROAD: log
                Flags indicating the type of calculation to be performed: calculate emission outside of genlbl
            
            Attributes
            -----------
            @attribute SURFACE: log
                Flag indicating whether the observer is position in space (False) and looks downwards or
                in the surface and looks upwards (True)
            @attribute NPATH: int
                Number of atmospheric paths required to perform the atmospheric calculation
            @attribute NLAYIN: 1D array
                For each path, number of layers involved in the calculation
            @attribute LAYINC: 2D array
                For each path, layers involved in the calculation
            @attribute EMTEMP: 1D array
                For each path, emission temperature of the layers involved in the calculation
            @attribute SCALE: 1D array
                For each path, scaling factor to calculate line-of-sight density in each layer with respect
                to the vertical line-of-sight density
            @attribute IMOD: 1D array
                For each path, calculation type       

            Methods
            -------

        """

        #parameters
        self.path_observer_pointing = path_observer_pointing
        self.path_observer_height = np.nan # height of the path observer above height = 0. Can be 0 or np.inf at the moment
        #self.LIMB = LIMB
        #self.NADIR = NADIR
        self.BOTLAY = BOTLAY
        self.ANGLE = ANGLE
        self.EMISS_ANG = EMISS_ANG
        self.SOL_ANG = SOL_ANG
        self.AZI_ANG = AZI_ANG
        self.IPZEN = IPZEN
        self.path_calc = path_calc
        #self.WF = WF
        #self.NETFLUX = NETFLUX
        #self.UPFLUX = UPFLUX
        #self.OUTFLUX = OUTFLUX
        #self.BOTFLUX = BOTFLUX
        #self.CG = CG
        #self.THERM = THERM
        #self.HEMISPHERE = HEMISPHERE
        #self.SCATTER = SCATTER
        #self.NEARLIMB = NEARLIMB
        #self.SINGLE = SINGLE
        #self.SPHSINGLE = SPHSINGLE
        #self.ABSORB = ABSORB
        #self.BINBB = BINBB
        #self.BROAD = BROAD

        #attributes
        #self.SURFACE = None   #Flag indicating whether the observer is on the surface (looking upwards)
        self.NPATH = None     #Number of paths needed for the atmospheric calculation
        self.NLAYIN = None    #np.array(NPATH) For each path, number of layers involved in the calculation
        self.LAYINC = None    #np.array(NLAYIN,NPATH) For each path, layers involved in the calculation
        self.EMTEMP = None    #np.array(NLAYIN,NPATH) For each path, emission temperature of each of the layers involved
        self.SCALE = None     #np.array(NLAYIN,NPATH) For each path, scaling factor 
                              #to calculate line-of-sight density in each layer
        self.IMOD = None      #np.array(NPATH) Calculation type
        self.ITYPE = None
        self.NINTP = None
        self.ICALD = None
        self.NREALP = None
        self.RCALD = None


        #Checking the geometry of the observation 
        ###########################################

        match self.path_observer_pointing:
            case PathObserverPointing.DISK:
                self.path_observer_height = np.inf
            case PathObserverPointing.LIMB:
                self.path_observer_height = np.inf
                self.ANGLE = 90.
            case PathObserverPointing.NADIR:
                if self.EMISS_ANG > 90.:
                    self.ANGLE = 180.0 - self.ANGLE
                    self.path_observer_height = 0.0
                else:
                    self.path_observer_height = np.inf
            case _:
                raise ValueError(f'error in AtmCalc_0 :: path observer pointing "{_}" not recognised')
        
        assert self.path_observer_height in (0.0, np.inf), f'error in AtmCalc_0 :: path observer height must be 0 or np.inf. Other values such as {self.path_observer_height} are not implemented yet'

        #Checking incompatible flags and resetting them 
        #################################################

        if (self.path_calc & PathCalc.THERMAL_EMISSION 
                and self.path_calc & PathCalc.ABSORBTION
            ):
            self.path_calc &= ~PathCalc.ABSORBTION
            print('warning in .pat file :: Cannot use absorption for thermal calcs - resetting')
        
        if (self.path_calc & PathCalc.SINGLE_SCATTERING_PLANE_PARALLEL 
                and self.path_calc & PathCalc.SCATTER
            ):
            self.path_calc &= ~PathCalc.SINGLE_SCATTERING_PLANE_PARALLEL
            print('warning in .pat file :: Cannot use SINGLE and SCATTER - resetting')
        
        if (self.path_calc & PathCalc.SINGLE_SCATTERING_SPHERICAL 
                and self.path_calc & PathCalc.SCATTER
            ):
            self.path_calc &= ~PathCalc.SINGLE_SCATTERING_SPHERICAL
            print('warning in .pat file :: Cannot use SPHSINGLE and SCATTER - resetting')
        
        if (self.path_calc & PathCalc.PLANK_FUNCTION_AT_BIN_CENTRE 
                and self.path_calc & PathCalc.BROADENING
            ):
            self.path_calc &= ~PathCalc.BROADENING
            print('warning in .pat file :: Cannot use BINBB and BROAD - resetting')
        
        if self.path_calc & PathCalc.THERMAL_EMISSION:
            self.path_calc &= ~PathCalc.BROADENING
            self.path_calc &= ~PathCalc.PLANK_FUNCTION_AT_BIN_CENTRE
            print('warning in .pat file :: Cannot use BINBB and BROAD with THERM - resetting')
        
        if (self.path_calc & (PathCalc.SINGLE_SCATTERING_PLANE_PARALLEL | PathCalc.SINGLE_SCATTERING_SPHERICAL | PathCalc.SCATTER)
                and self.path_calc & PathCalc.THERMAL_EMISSION
            ):
            self.path_calc &= ~PathCalc.THERMAL_EMISSION
            print('THERM not required. Scattering includes emission')
        
        if (self.path_calc & PathCalc.HEMISPHERE
                and self.path_calc & PathCalc.THERMAL_EMISSION
            ):
            print('HEMISPHERE assumes THERM')
        
        if (self.path_calc & (PathCalc.SCATTER | PathCalc.SINGLE_SCATTERING_PLANE_PARALLEL | PathCalc.SINGLE_SCATTERING_SPHERICAL)
                and self.path_calc & PathCalc.CURTIS_GODSON
            ):
            self.path_calc &= ~PathCalc.CURTIS_GODSON
            print('warning in .pat file :: Cannot use CG and SCATTER - resetting')
        
        if (self.path_calc & PathCalc.SCATTER | PathCalc.SINGLE_SCATTERING_PLANE_PARALLEL | PathCalc.SINGLE_SCATTERING_SPHERICAL):
            if self.path_observer_pointing == PathObserverPointing.LIMB:
                if self.path_calc & PathCalc.SINGLE_SCATTERING_PLANE_PARALLEL:
                    raise ValueError('error in .pat file :: SINGLE and LIMB not catered for')
                if self.path_calc & PathCalc.SINGLE_SCATTERING_SPHERICAL:
                    raise ValueError('error in .pat file :: SPHSINGLE and LIMB not catered for')  
            else:
                if self.ANGLE != 0.0:
                    print('warning in .pat file :: ANGLE must be 0.0 for scattering calculations - resetting')    
                    self.ANGLE = 0.0
        
        if self.path_calc & PathCalc.HEMISPHERE:
            if self.path_observer_pointing == PathObserverPointing.LIMB:
                raise ValueError('error in .pat file :: cannot do HEMISPHERE and LIMB')  
            else:
                if self.ANGLE != 0.0:
                    print('warning in .pat file :: ANGLE must be 0.0 for HEMISPHERE - resetting') 
                    self.ANGLE = 0.0
        
        
        #Translating ANGLE to be defined at bottom of bottom layer in case it
        #has been defined in another way in the .pat file
        #######################################################################

        if(self.IPZEN==ZenithAngleOrigin.BOTTOM):
            #Compute zenith angle of ray at bottom of bottom layer, assuming it
            #has been defined at the 0km level
            z0 = Layer.RADIUS + Layer.BASEH[self.BOTLAY]
            self.ANGLE = np.arcsin(Layer.RADIUS/z0 * np.sin(self.ANGLE/180.*np.pi)) / np.pi * 180.
        elif(self.IPZEN==ZenithAngleOrigin.TOP):
            #Compute zenith angle of ray at bottom of bottom layer, assuming it
            #has been defined at the top of the atmosphere
            z0 = Layer.RADIUS + Layer.BASEH[Layer.NLAY-1] + Layer.DELH[Layer.NLAY-1]

            #Calculate tangent altitude of ray at lowest point
            HTAN = z0*np.sin(self.ANGLE/180.*np.pi)-Layer.RADIUS

            if HTAN<=Layer.BASEH[self.BOTLAY]:
                #Calculate zenith angle at bottom of lowest layer
                self.ANGLE = np.arcsin(z0/(Layer.RADIUS + Layer.BASEH[self.BOTLAY]) * np.sin(self.ANGLE/180.*np.pi)) / np.pi * 180.
            else:
                #We need to model this ray as a tangent path.
                self.path_observer_pointing = PathObserverPointing.LIMB
                self.ANGLE = 90.

                #Find number of bottom layer. Snap to layer with nearest base height
                #to computed tangent height.
                for ILAY in range(Layer.NLAY):
                    if Layer.BASEH[ILAY]<HTAN:
                        self.BOTLAY = ILAY
                
                if self.BOTLAY<Layer.NLAY-1:
                    F = (HTAN-Layer.BASEH[self.BOTLAY])/(Layer.BASEH[self.BOTLAY+1]-Layer.BASEH[self.BOTLAY])
                    if F>0.5:
                        self.BOTLAY = self.BOTLAY + 1


        Z0 = Layer.RADIUS + Layer.BASEH[self.BOTLAY]
        SIN2A = np.sin(self.ANGLE/180.*np.pi)**2.
        COSA = np.cos(self.ANGLE/180.*np.pi)

        #Calculate which layers to use in the calculation
        #######################################################################

        #Limb path
        if self.path_observer_pointing == PathObserverPointing.LIMB:

            #Calculate the total number of layers to use
            NUSE = int(2*(Layer.NLAY-self.BOTLAY))

            #Locating the layers to be included
            USELAY = np.zeros(NUSE,dtype='int32')
            for IUSE in range(int(NUSE/2)):
                USELAY[IUSE] = Layer.NLAY - 1 - IUSE
                USELAY[int(NUSE/2)+IUSE] = self.BOTLAY + IUSE
                
            #Calculating the emission temperature for those layers
            EMITT = np.zeros(NUSE)  #Emission temperature
            for IUSE in range(NUSE):
                EMITT[IUSE] = Layer.TEMP[USELAY[IUSE]]

        #Nadir path
        if self.path_observer_pointing == PathObserverPointing.NADIR:

            #Calculate the total number of layers to use
            NUSE = Layer.NLAY-self.BOTLAY

            if self.path_observer_height == 0.0: # Observer on the surface looking upwards
                #Locating layers and calculating emission temperature
                USELAY = np.zeros(NUSE,dtype='int32')
                EMITT = np.zeros(NUSE)  #Emission temperature
                for IUSE in range(NUSE):
                    USELAY[IUSE] = IUSE 
                    EMITT[IUSE] = Layer.TEMP[USELAY[IUSE]]

            elif self.path_observer_height == np.inf: # Observer in space looking down

                #Locating layers and calculating emission temperature
                USELAY = np.zeros(NUSE,dtype='int32')
                EMITT = np.zeros(NUSE)  #Emission temperature
                for IUSE in range(NUSE):
                    USELAY[IUSE] = Layer.NLAY - 1 - IUSE 
                    EMITT[IUSE] = Layer.TEMP[USELAY[IUSE]]


        #Computing the scale factor for the path in each layer (with respect to vertical integration)
        #################################################################################################

        SF = np.zeros(NUSE)   #Scaling factor 
        for IUSE in range(NUSE):

            STMP = (Layer.RADIUS + Layer.BASEH[USELAY[IUSE]])**2. - SIN2A * Z0**2.
            #Sometimes, there are rounding errors here that cause the program
            #to try to take the square root of a _very_ small negative number.
            #This quietly fixes that and hopefully doesn't break anything else.

            if STMP<0.0:
                STMP = 0.0
            
            S0 = np.sqrt(STMP) - Z0 * COSA

            if USELAY[IUSE]<Layer.NLAY - 1:
                S1 = np.sqrt( (Layer.RADIUS+Layer.BASEH[USELAY[IUSE]+1])**2. - SIN2A * Z0**2. ) - Z0 * COSA
                SF[IUSE] = (S1-S0)/(Layer.BASEH[USELAY[IUSE]+1]-Layer.BASEH[USELAY[IUSE]])
            if USELAY[IUSE]==Layer.NLAY - 1:
                NPRO = len(Layer.H)
                S1 = np.sqrt( (Layer.RADIUS+Layer.H[NPRO-1])**2. - SIN2A * Z0**2. ) - Z0 * COSA
                SF[IUSE] = (S1-S0)/(Layer.H[NPRO-1]-Layer.BASEH[USELAY[IUSE]])


        #Calculating any Curtis-Godson paths if needed
        #####################################################

        if self.path_calc & PathCalc.CURTIS_GODSON:
            raise NotImplementedError('error in .pat file :: Curtis-Godson files are not implemented in the path calculation yet')

        #Calculating the calculation type to pass to RADTRANS
        ######################################################

        #Calculating the number of paths required for the calculation
        #For example if calculating a weighting function or if performing a thermal integration outside the main Radtrans routines
        
        self.NPATH = 1

        if self.path_calc & PathCalc.WEIGHTING_FUNCTION:
            self.NPATH = NUSE
        
        if (self.path_calc & PathCalc.THERMAL_EMISSION
                and self.path_calc & PathCalc.BROADENING
            ):
            self.NPATH = NUSE
        
        if self.path_calc & PathCalc.UPWARD_FLUX:
            self.NPATH = NUSE
        
        if self.path_calc & PathCalc.NET_FLUX:
            raise ValueError('error :: need to properly define the paths (should be 2*NLAYER for upward and downward flux)')
            self.NPATH = NUSE

        if (self.path_calc & PathCalc.UPWARD_FLUX
                and not self.path_calc & PathCalc.SINGLE_SCATTERING_PLANE_PARALLEL
            ):
            raise ValueError('error in .pat file :: cannot do upward flux calculation with scattering turned off')
        
        if (self.path_calc & PathCalc.OUTWARD_FLUX
                and not self.path_calc & PathCalc.SINGLE_SCATTERING_PLANE_PARALLEL
            ):
            raise ValueError('error in .pat file :: cannot do outward flux calculation with scattering turned off')
        
        if (self.path_calc & PathCalc.DOWNWARD_FLUX
                and not self.path_calc & PathCalc.SINGLE_SCATTERING_PLANE_PARALLEL
            ):
            raise ValueError('error in .pat file :: cannot do downward flux calculation with scattering turned off')
        
        
        

        NLAYIN = np.zeros(self.NPATH,dtype='int32')
        LAYINC = np.zeros([NUSE,self.NPATH],dtype='int32')
        IMOD = np.zeros(self.NPATH,dtype='int32')
        SCALE = np.zeros([NUSE,self.NPATH])
        EMTEMP = np.zeros([NUSE,self.NPATH])
        for j in range(self.NPATH):
            
            #Setting the correct type for the path
            IMOD[j] = self.path_calc


        self.IMOD = IMOD
        self.NLAYIN = NLAYIN
        self.LAYINC = LAYINC
        self.EMTEMP = EMTEMP
        self.SCALE = SCALE

        #Having calculated the atmospheric paths, now outputing the calculation
        #############################################################################

        ITYPE = self.path_calc
        
        NINTP = 3
        ICALD = np.zeros(NINTP,dtype='int32')
        ICALD[0] = 1
        ICALD[1] = self.NPATH
        ICALD[2] = self.BOTLAY
        NREALP = 2
        RCALD = np.zeros(NREALP,dtype='int32')
        RCALD[0] = ANGLE
        HT = 0.0        # Fix!!!!!!!!!!!!!!!
        RCALD[1] = HT

        self.ITYPE = ITYPE
        self.NINTP = NINTP
        self.ICALD = ICALD
        self.NREALP = NREALP
        self.RCALD = RCALD