import pytest  
import archnemesis as ans
import numpy as np
import os
curr = os.getcwd()


def test_makephase_imie0():  
    '''
    Test MakePhase program to calculate the extinction coefficient, single scattering albedo and phase function using the Mie theory
    
    We make this test following the refractive index of Mars dust and assuming the following particle size distributions:
    - Log-normal distribution with a = 1.0, b = 0.1
    - Standard gamma distribution with a = 2.0, b = 0.1
    - Single particle size with r = 1.5
    - MCS Modified standard gamma distribution with a = 2.0, b = 0.1, c = 6.
    
    
    In this case, IMIE = 0, which means that the phase function is then fit using a double Henyey-Greenstein function.
    '''
    
    # CALCULATING THE EXTINCTION COEFFICIENT, SINGLE SCATTERING ALBEDO AND PHASE FUNCTION USING THE MIE THEORY WITH ARCHNEMESIS
    ##################################################################################################################################
    
    
    #Declaring class
    Scatter = ans.Scatter_0()
    
    #Defining the characteristics of the class
    Scatter.ISPACE = 1   #Units of the calculations (0 - Wavenumber in cm-1 ; 1 - Wavelength in um)
    Scatter.IMIE = 0     #Double Henyey-Greenstein phase function fit

    NDUST = 4      #Number of aerosol populations that we want to include in our atmosphere
    wavel = np.arange(0.5,2.5+0.5,0.5)
    theta = np.array([0.,1.,2.,3.,4.,5.,7.5,10.,12.5,15.,17.5,20.,25.,30.,35.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,145.,150.,155.,160.,162.5,165.,167.5,170.,172.5,175.,176.,177.,178.,179.,180.])
    NWAVE = len(wavel)    #Number of spectral points
    NTHETA = len(theta)
    
    #Now we initialise the arrays that will be filled with the calculations
    Scatter.initialise_arrays(NDUST,NWAVE,NTHETA)
    Scatter.WAVE = wavel
    Scatter.THETA = theta
    
    #Defining the inputs for our calculations of this particle aerosol distribution
    ################################################################################################

    #Reading the refractive index from the dictionary (Mars dust)
    Scatter.read_refind(1)

    #Defining the inputs for our standard gamma particle size distribution
    iscat = 2  #Log-normal distribution
    pars = np.array([1.0,0.1])

    idust = 0    #The index of the aerosol populations in the class that this calculation corresponds to (from 0 to NDUST-1)
    Scatter.makephase(idust,iscat,pars)

    #Defining the inputs for our calculations of this particle aerosol distribution
    ################################################################################################

    #Reading the refractive index from the dictionary (Mars dust)
    Scatter.read_refind(1)

    #Defining the inputs for our standard gamma particle size distribution
    iscat = 1  #Standard gamma distribution
    a = 2.0 ; b = 0.1 ; alpha = (1-3.*b)/b
    pars = np.array([a,b,alpha])

    idust = 1    #The index of the aerosol populations in the class that this calculation corresponds to (from 0 to NDUST-1)
    Scatter.makephase(idust,iscat,pars)     



    #Defining the inputs for our calculations of this particle aerosol distribution
    ################################################################################################

    #Reading the refractive index from the dictionary (Mars dust)
    Scatter.read_refind(1)

    #Defining the inputs 
    iscat = 4  #Single particle size
    r0 = 1.5
    pars = np.array([r0])

    idust = 2    #The index of the aerosol populations in the class that this calculation corresponds to (from 0 to NDUST-1)
    Scatter.makephase(idust,iscat,pars)



    #Defining the inputs for our calculations of this particle aerosol distribution
    ################################################################################################

    #Reading the refractive index from the dictionary (Mars dust)
    Scatter.read_refind(1)

    #Defining the inputs for our standard gamma particle size distribution
    iscat = 3  #Modified gamma distribution
    a = 2. ; b = 0.1 ; c = 6.
    pars = np.array([a,b,c])

    idust = 3    #The index of the aerosol populations in the class that this calculation corresponds to (from 0 to NDUST-1)
    Scatter.makephase(idust,iscat,pars)
    
    Scatter.check_phase_norm()
 
    # READING THE RESULTS OF THE CALCULATIONS FROM NEMESIS
    ##################################################################################################################################
    
    ScatterN = ans.Scatter_0()
    ScatterN.IMIE = 0
    test_dir = os.path.join(ans.archnemesis_path(), 'tests/files/Makephase/')
    os.chdir(test_dir) #Changing directory to read files
    ScatterN.read_xsc('nemesis_mars_dust')
    ScatterN.read_hgphase()
    
    assert np.allclose(Scatter.KEXT, ScatterN.KEXT, rtol=1.0e-3)
    

  
def test_makephase_imie1():  
    '''
    Test MakePhase program to calculate the extinction coefficient, single scattering albedo and phase function using the Mie theory
    
    We make this test following the refractive index of Mars dust and assuming the following particle size distributions:
    - Log-normal distribution with a = 1.0, b = 0.1
    - Standard gamma distribution with a = 2.0, b = 0.1
    - Single particle size with r = 1.5
    - MCS Modified standard gamma distribution with a = 2.0, b = 0.1, c = 6.
    '''
    
    # CALCULATING THE EXTINCTION COEFFICIENT, SINGLE SCATTERING ALBEDO AND PHASE FUNCTION USING THE MIE THEORY WITH ARCHNEMESIS
    ##################################################################################################################################
    
    
    #Declaring class
    Scatter = ans.Scatter_0()
    
    #Defining the characteristics of the class
    Scatter.ISPACE = 1   #Units of the calculations (0 - Wavenumber in cm-1 ; 1 - Wavelength in um)
    Scatter.IMIE = 1     #Phase function defined explicitly

    NDUST = 4      #Number of aerosol populations that we want to include in our atmosphere
    wavel = np.arange(0.5,2.5+0.5,0.5)
    theta = np.array([0.,1.,2.,3.,4.,5.,7.5,10.,12.5,15.,17.5,20.,25.,30.,35.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,145.,150.,155.,160.,162.5,165.,167.5,170.,172.5,175.,176.,177.,178.,179.,180.])
    NWAVE = len(wavel)    #Number of spectral points
    NTHETA = len(theta)
    
    #Now we initialise the arrays that will be filled with the calculations
    Scatter.initialise_arrays(NDUST,NWAVE,NTHETA)
    Scatter.WAVE = wavel
    Scatter.THETA = theta
    
    #Defining the inputs for our calculations of this particle aerosol distribution
    ################################################################################################

    #Reading the refractive index from the dictionary (Mars dust)
    Scatter.read_refind(1)

    #Defining the inputs for our standard gamma particle size distribution
    iscat = 2  #Log-normal distribution
    pars = np.array([1.0,0.1])

    idust = 0    #The index of the aerosol populations in the class that this calculation corresponds to (from 0 to NDUST-1)
    Scatter.makephase(idust,iscat,pars)

    #Defining the inputs for our calculations of this particle aerosol distribution
    ################################################################################################

    #Reading the refractive index from the dictionary (Mars dust)
    Scatter.read_refind(1)

    #Defining the inputs for our standard gamma particle size distribution
    iscat = 1  #Standard gamma distribution
    a = 2.0 ; b = 0.1 ; alpha = (1-3.*b)/b
    pars = np.array([a,b,alpha])

    idust = 1    #The index of the aerosol populations in the class that this calculation corresponds to (from 0 to NDUST-1)
    Scatter.makephase(idust,iscat,pars)     



    #Defining the inputs for our calculations of this particle aerosol distribution
    ################################################################################################

    #Reading the refractive index from the dictionary (Mars dust)
    Scatter.read_refind(1)

    #Defining the inputs 
    iscat = 4  #Single particle size
    r0 = 1.5
    pars = np.array([r0])

    idust = 2    #The index of the aerosol populations in the class that this calculation corresponds to (from 0 to NDUST-1)
    Scatter.makephase(idust,iscat,pars)



    #Defining the inputs for our calculations of this particle aerosol distribution
    ################################################################################################

    #Reading the refractive index from the dictionary (Mars dust)
    Scatter.read_refind(1)

    #Defining the inputs for our standard gamma particle size distribution
    iscat = 3  #Modified gamma distribution
    a = 2. ; b = 0.1 ; c = 6.
    pars = np.array([a,b,c])

    idust = 3    #The index of the aerosol populations in the class that this calculation corresponds to (from 0 to NDUST-1)
    Scatter.makephase(idust,iscat,pars)
    
    Scatter.check_phase_norm()
 
    # READING THE RESULTS OF THE CALCULATIONS FROM NEMESIS
    ##################################################################################################################################
    
    ScatterN = ans.Scatter_0()
    ScatterN.IMIE = 1
    test_dir = os.path.join(ans.archnemesis_path(), 'tests/files/Makephase/')
    os.chdir(test_dir) #Changing directory to read files
    ScatterN.read_xsc('nemesis_mars_dust')
    ScatterN.read_phase()
    
    assert np.allclose(Scatter.KEXT, ScatterN.KEXT, rtol=1.0e-3)
    