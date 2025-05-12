import pytest  
import archnemesis as ans
import numpy as np

from archnemesis.enums import PlanetEnum, AtmosphericProfileFormatEnum
  
def test_gravity():  
    '''
    Test calculation of gravity
    '''
    
    #Declaring class
    Atmosphere = ans.Atmosphere_0()
    Atmosphere.IPLANET = PlanetEnum.Earth   #Earth - The test can fail if the parameters for Earth in the dictionary are changed
    Atmosphere.NP = 3
    Atmosphere.H = np.linspace(0.,30000.,Atmosphere.NP)
    Atmosphere.LATITUDE=0.
    Atmosphere.calc_grav()
    
    expected = np.array([9.78027361, 9.7341945, 9.68843803])
    # Use a NumPy comparison for arrays
    assert np.allclose(Atmosphere.GRAV, expected, atol=1e-6)
 
  
def test_molwt():
    '''
    Test calculation of mean molecular weight
    '''
    
    #Declaring class
    Atmosphere = ans.Atmosphere_0()

    Atmosphere.NP = 2
    Atmosphere.NVMR = 5
    Atmosphere.ID = np.array([1,2,3,4,5])
    Atmosphere.ISO = np.array([0,0,0,0,0])
    Atmosphere.AMFORM = AtmosphericProfileFormatEnum.CALC_MOLECULAR_WEIGHT_DO_NOT_SCALE_VMR   #Calculate molecular weight automatically
    
    vmr = np.zeros((Atmosphere.NP,Atmosphere.NVMR))
    vmr[0,:] = np.array([0.1,0.5,0.1,0.1,0.2])
    vmr[1,:] = np.array([0.5,0.1,0.1,0.2,0.1])
    Atmosphere.edit_VMR(vmr)
    
    Atmosphere.calc_molwt()
    
    expected = np.array([0.0386093, 0.02981164])
    # Use a NumPy comparison for arrays
    assert np.allclose(Atmosphere.MOLWT, expected, atol=1e-6)
    
    
def test_hydrostath():
    '''
    Test calculation of hydrostatic calculations
    '''
    
    Atmosphere = ans.Atmosphere_0()

    Atmosphere.IPLANET = PlanetEnum.Earth
    Atmosphere.AMFORM = AtmosphericProfileFormatEnum.CALC_MOLECULAR_WEIGHT_DO_NOT_SCALE_VMR
    Atmosphere.NP = 3
    Atmosphere.NVMR = 8
    Atmosphere.LATITUDE = 30.
    
    #Defining gases 
    Atmosphere.ID = np.array([1,2,3,4,5,6,7,22])
    Atmosphere.ISO = np.array([0,0,0,0,0,0,0,0])

    #Defining vertical profiles
    Atmosphere.edit_H(np.array([0.,2902.,7417.]))
    Atmosphere.edit_P(np.array([9.710141e-01,6.825167e-01,3.811004e-01])/101325.)
    Atmosphere.edit_T(np.array([288.9900,279.1200,249.3800]))
    VMR = np.zeros((Atmosphere.NP,Atmosphere.NVMR))
    VMR[0,:] = np.array([6.637074e-03,3.599889e-04,6.859128e-08,3.199949e-07,1.482969e-07,1.700002e-06,2.089960e-01,7.840047e-01])
    VMR[1,:] = np.array([1.402168e-03,3.600041e-04,5.794829e-08,3.200007e-07,1.338216e-07,1.700007e-06,2.090029e-01,7.892327e-01])
    VMR[2,:] = np.array([7.306020e-05,3.599975e-04,5.972404e-08,3.200221e-07,1.202784e-07,1.697634e-06,2.089991e-01,7.905656e-01])
    Atmosphere.edit_VMR(VMR)

    #Making some calculations
    Atmosphere.calc_grav()
    Atmosphere.calc_molwt()
    
    Atmosphere.adjust_hydrostatH()

    expected_h = np.array([0.,2950.78396658,7487.94634149])

    assert np.allclose(Atmosphere.H, expected_h, atol=1e-6)
    
    
def test_hydrostatp():
    '''
    Test calculation of hydrostatic calculations
    '''
    
    Atmosphere = ans.Atmosphere_0()

    Atmosphere.IPLANET = PlanetEnum.Earth
    Atmosphere.AMFORM = AtmosphericProfileFormatEnum.CALC_MOLECULAR_WEIGHT_DO_NOT_SCALE_VMR
    Atmosphere.NP = 3
    Atmosphere.NVMR = 8
    Atmosphere.LATITUDE = 30.
    
    #Defining gases 
    Atmosphere.ID = np.array([1,2,3,4,5,6,7,22])
    Atmosphere.ISO = np.array([0,0,0,0,0,0,0,0])

    #Defining vertical profiles
    Atmosphere.edit_H(np.array([0.,2902.,7417.]))
    Atmosphere.edit_P(np.array([9.710141e-01,6.825167e-01,3.811004e-01])/101325.)
    Atmosphere.edit_T(np.array([288.9900,279.1200,249.3800]))
    VMR = np.zeros((Atmosphere.NP,Atmosphere.NVMR))
    VMR[0,:] = np.array([6.637074e-03,3.599889e-04,6.859128e-08,3.199949e-07,1.482969e-07,1.700002e-06,2.089960e-01,7.840047e-01])
    VMR[1,:] = np.array([1.402168e-03,3.600041e-04,5.794829e-08,3.200007e-07,1.338216e-07,1.700007e-06,2.090029e-01,7.892327e-01])
    VMR[2,:] = np.array([7.306020e-05,3.599975e-04,5.972404e-08,3.200221e-07,1.202784e-07,1.697634e-06,2.089991e-01,7.905656e-01])
    Atmosphere.edit_VMR(VMR)

    #Making some calculations
    Atmosphere.calc_grav()
    Atmosphere.calc_molwt()

    ptan = 5.73591611e-6 ; htan = 2902.
    Atmosphere.adjust_hydrostatP(htan,ptan)

    expected_p = np.array([8.11304173e-06,5.73591611e-06,3.21192271e-06])

    assert np.allclose(Atmosphere.P, expected_p, atol=1e-6)
    
    
