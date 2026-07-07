import pytest  
import numpy as np
import os

import archnemesis as ans
import archnemesis.Data.path_data

curr = os.getcwd()

if False:
    print(pytest) # Removes the "unused import" error


THERMAL_EMISSION_CIRS_EXPECTED_RESULT = np.load(
    ans.Data.path_data.archnemesis_resolve_path("ARCHNEMESIS_PATH/tests/data/thermal_emission_cirs_expected_result.npy"), 
    allow_pickle=False
)

MULTIPLE_SCATTERING_CIRS_EXPECTED_RESULT = np.load(
    ans.Data.path_data.archnemesis_resolve_path("ARCHNEMESIS_PATH/tests/data/multiple_scattering_cirs_expected_result.npy"), 
    allow_pickle=False
)


def test_thermal_emission_cirs():  
    '''
    Jupiter thermal emission test against NEMESIS
    '''
    test_dir = os.path.join(ans.archnemesis_path(), 'tests/files/Jupiter_CIRS_nadir_thermal_emission/')
    runname = 'cirstest'
    expected_result = THERMAL_EMISSION_CIRS_EXPECTED_RESULT
    
    os.chdir(test_dir) #Changing directory to read files
    try:
        #Reading the input files
        Atmosphere,Measurement,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Variables,Retrieval = ans.Files.read_input_files(runname)
        
        #Calculating forward model with CIRSrad
        ForwardModel = ans.ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,Measurement=Measurement,Spectroscopy=Spectroscopy,Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=Variables)
        SPECONV_cirsrad = ForwardModel.nemesisfm()
        calculation_cirsrad = SPECONV_cirsrad[:,0]
        
        assert calculation_cirsrad.size == expected_result.size, f"Must have expected number of points {expected_result.size=} {calculation_cirsrad.size=}"
        assert not np.any(np.isnan(calculation_cirsrad)), f"No computed values must be NAN, calculation_cirsrad has {np.count_nonzero(np.isnan(calculation_cirsrad))} NANs"
        
        #Calculating forward model with CIRSradg
        SPECONV_cirsradg,_ = ForwardModel.nemesisfmg()
        calculation_cirsradg = SPECONV_cirsradg[:,0]
        
        assert calculation_cirsradg.size == expected_result.size, f"Must have expected number of points {expected_result.size=} {calculation_cirsradg.size=}"
        assert not np.any(np.isnan(calculation_cirsradg)), f"No computed values must be NAN, calculation_cirsradg has {np.count_nonzero(np.isnan(calculation_cirsradg))} NANs"
    finally:
        os.chdir(curr) #Changing directory back to the original
    
    atol=np.quantile(expected_result, 0.5) * 1E-2
    rtol = 5E-2
    
    if False: # Extra diagnostic information if required
        import matplotlib.pyplot as plt
        
        wave = ForwardModel.MeasurementX.VCONV[:ForwardModel.MeasurementX.NCONV[0]][:,0]
        
        tol = atol + rtol*np.abs(expected_result)
        
        plt.figure()
        plt.title('Expected vs Calculated result')
        plt.plot(wave, calculation_cirsrad, marker='none', ls='-', alpha=0.6, label='calculated spectra')
        plt.plot(wave, calculation_cirsradg, marker='none', ls='-', alpha=0.6, label='calculated gradient')
        lines = plt.plot(wave, expected_result, marker='none', ls='-', color='tab:red', alpha=0.6, label='expected result')
        plt.fill_between(wave, expected_result - tol, expected_result+tol, alpha=0.1, color=lines[0].get_color(), label='expected region')
        plt.xlabel('Wave ($cm^{-1}$)')
        plt.ylabel('Radiance ($W cm^{-2} sr^{-1} (cm^{-1})^{-1})$)')
        plt.legend()
        
        plt.figure()
        plt.title('Residual (calculated - expected)')
        plt.plot(wave, calculation_cirsrad - expected_result, marker='none', ls='--', alpha=0.6, label='residual spectra')
        plt.plot(wave, calculation_cirsradg - expected_result, marker='none', ls='--', alpha=0.6, label='residual gradient')
        plt.fill_between(wave, -1*tol, tol, alpha=0.1, color=lines[0].get_color(), label='expected region')
        plt.xlabel('Wave ($cm^{-1}$)')
        plt.ylabel('Radiance ($W cm^{-2} sr^{-1} (cm^{-1})^{-1})$)')
        plt.legend()
        
        plt.show()
    
    # Use a NumPy comparison for arrays
    np.testing.assert_allclose(calculation_cirsrad, expected_result, atol=atol, rtol=rtol)
    np.testing.assert_allclose(calculation_cirsradg, expected_result, atol=atol, rtol=rtol)

def test_thermal_emission_realtime_line_by_line():
    '''
    Jupiter thermal emission test against NEMESIS using REALTIME absorption coefficient calculation
    '''
    test_dir = os.path.join(ans.archnemesis_path(), 'tests/files/Jupiter_CIRS_nadir_thermal_emission_runtime_line_by_line/')
    runname = 'cirstest'
    expected_result = THERMAL_EMISSION_CIRS_EXPECTED_RESULT
    
    os.chdir(test_dir) #Changing directory to read files
    try:
        #Reading the input files
        Atmosphere,Measurement,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Variables,Retrieval = ans.Files.read_input_files(runname)
        
        #Calculating forward model with CIRSrad
        ForwardModel = ans.ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,Measurement=Measurement,Spectroscopy=Spectroscopy,Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=Variables)
        SPECONV_cirsrad = ForwardModel.nemesisfm()
        calculation_cirsrad = SPECONV_cirsrad[:,0]
        
        assert calculation_cirsrad.size == expected_result.size, f"Must have expected number of points {expected_result.size=} {calculation_cirsrad.size=}"
        assert not np.any(np.isnan(calculation_cirsrad)), f"No computed values must be NAN, calculation_cirsrad has {np.count_nonzero(np.isnan(calculation_cirsrad))} NANs"

        #Calculating forward model with CIRSradg
        SPECONV_cirsradg,_ = ForwardModel.nemesisfmg()
        calculation_cirsradg = SPECONV_cirsradg[:,0]
        
        assert calculation_cirsradg.size == expected_result.size, f"Must have expected number of points {expected_result.size=} {calculation_cirsradg.size=}"
        assert not np.any(np.isnan(calculation_cirsradg)), f"No computed values must be NAN, calculation_cirsradg has {np.count_nonzero(np.isnan(calculation_cirsradg))} NANs"
    finally:
        os.chdir(curr) #Changing directory back to the original
    
    
    
    # As `expected_result` is calculated from k-tables we need to relax the
    # range of allowed values to avoid over-specifying the pass conditions.
    atol = np.quantile(expected_result, 0.5) * 2E-1
    rtol = 2E-1
    required_frac_within_tolerance = 0.95
    
    tol = atol + rtol*np.abs(expected_result)
    
    cirsrad_residual = calculation_cirsrad - expected_result
    cirsradg_residual = calculation_cirsradg - expected_result
    
    cirsrad_out_of_tolerance_mask = np.abs(cirsrad_residual) > tol
    cirsradg_out_of_tolerance_mask = np.abs(cirsradg_residual) > tol
    
    cirsrad_frac_out_of_tolerance = np.count_nonzero(cirsrad_out_of_tolerance_mask) / calculation_cirsrad.size
    cirsradg_frac_out_of_tolerance = np.count_nonzero(cirsradg_out_of_tolerance_mask) / calculation_cirsradg.size

    if False: # Extra diagnostic information if required
        import matplotlib.pyplot as plt

        wave = ForwardModel.MeasurementX.VCONV[:ForwardModel.MeasurementX.NCONV[0]][:,0]

        msg_frac_within_tol = f'frac within tolerance: cirsrad={1-cirsrad_frac_out_of_tolerance:5.3} cirsradg={1-cirsradg_frac_out_of_tolerance:5.3}'

        plt.figure()
        plt.title(f'Expected vs Calculated result\n{msg_frac_within_tol}')
        plt.plot(wave, calculation_cirsrad, marker='+', ls='-', alpha=0.6, label='calculated spectra')
        plt.plot(wave, calculation_cirsradg, marker='x', ls='-', alpha=0.6, label='calculated gradient')
        lines = plt.plot(wave, expected_result, marker='none', ls='-', color='tab:red', alpha=0.6, label='expected result')
        plt.fill_between(wave, expected_result - tol, expected_result+tol, alpha=0.1, color=lines[0].get_color(), label='expected region')
        
        if cirsrad_frac_out_of_tolerance > 0:
            plt.plot(wave[cirsrad_out_of_tolerance_mask], calculation_cirsrad[cirsrad_out_of_tolerance_mask], marker='s', markersize=10, markerfacecolor='none', ls='none', color='tab:red', alpha=0.6, label='calculated values out of tolerance')
        if cirsradg_frac_out_of_tolerance > 0:
            plt.plot(wave[cirsradg_out_of_tolerance_mask], calculation_cirsradg[cirsradg_out_of_tolerance_mask], marker='^', markersize=10, markerfacecolor='none', ls='none', color='tab:red', alpha=0.6, label='gradient calculated values out of tolerance')
        
        plt.xlabel('Wave ($cm^{-1}$)')
        plt.ylabel('Radiance ($W cm^{-2} sr^{-1} (cm^{-1})^{-1})$)')
        plt.legend()
        
        plt.figure()
        plt.title(f'Residual (calculated - expected)\n{msg_frac_within_tol}')
        plt.plot(wave, cirsrad_residual, marker='+', ls='--', alpha=0.6, label='residual spectra')
        plt.plot(wave, cirsradg_residual, marker='x', ls='--', alpha=0.6, label='residual gradient')
        plt.fill_between(wave, -1*tol, tol, alpha=0.1, color=lines[0].get_color(), label='expected region')
        
        if cirsrad_frac_out_of_tolerance > 0:
            plt.plot(wave[cirsrad_out_of_tolerance_mask], cirsrad_residual[cirsrad_out_of_tolerance_mask], marker='s', markersize=10, markerfacecolor='none', ls='none', color='tab:red', alpha=0.6, label='calculated values out of tolerance')
        if cirsradg_frac_out_of_tolerance > 0:
            plt.plot(wave[cirsradg_out_of_tolerance_mask], cirsradg_residual[cirsradg_out_of_tolerance_mask], marker='^', markersize=10, markerfacecolor='none', ls='none', color='tab:red', alpha=0.6, label='gradient calculated values out of tolerance')
        
        plt.xlabel('Wave ($cm^{-1}$)')
        plt.ylabel('Radiance ($W cm^{-2} sr^{-1} (cm^{-1})^{-1})$)')
        plt.legend()
        
        plt.show()

    # Perform comparison
    assert (1-cirsrad_frac_out_of_tolerance) >= required_frac_within_tolerance, (
        f'## Failed Comparison ##\n'
        f'  Parameters:\n'
        f'    {atol = :8.3G}\n'
        f'    {rtol = :8.3G}\n'
        f'    {required_frac_within_tolerance = }\n'
        f'  Comparison:\n'
        f'    calculated frac within tolerance = {1-cirsrad_frac_out_of_tolerance:0.3f}\n'
        f'    expected   = {np.array2string(expected_result, max_line_width=80, prefix="    expected   = ", threshold=16, edgeitems=8)}\n'
        f'    calculated = {np.array2string(calculation_cirsrad, max_line_width=80, prefix="    calculated = ", threshold=16, edgeitems=8)}\n'
        f'##-------------------##'
    )
    
    assert (1-cirsradg_frac_out_of_tolerance) >= required_frac_within_tolerance, (
        f'## Failed Comparison ##\n'
        f'  Parameters:\n'
        f'    {atol = :8.3G}\n'
        f'    {rtol = :8.3G}\n'
        f'    {required_frac_within_tolerance = }\n'
        f'  Comparison:\n'
        f'    calculated frac within tolerance = {1-cirsradg_frac_out_of_tolerance:0.3f}\n'
        f'    expected   = {np.array2string(expected_result, max_line_width=80, prefix="    expected   = ", threshold=16, edgeitems=8)}\n'
        f'    calculated = {np.array2string(calculation_cirsradg, max_line_width=80, prefix="    calculated = ", threshold=16, edgeitems=8)}\n'
        f'##-------------------##'
    )


def test_multiple_scattering_cirs():  
    '''
    Jupiter multiple scattering test against NEMESIS
    '''
    test_dir = os.path.join(ans.archnemesis_path(), 'tests/files/Jupiter_CIRS_angled_thermal_emission_scattering/')
    os.chdir(test_dir) #Changing directory to read files
    runname = 'cirstest'
    
    #Reading the input files
    Atmosphere,Measurement,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Variables,Retrieval = ans.Files.read_input_files(runname)
    
    #Calculating forward model with CIRSrad
    ForwardModel = ans.ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,Measurement=Measurement,Spectroscopy=Spectroscopy,Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=Variables)
    SPECONV_cirsrad = ForwardModel.nemesisfm()
    calculation_cirsrad = SPECONV_cirsrad[:,0]
    os.chdir(curr)
    
    assert np.allclose(calculation_cirsrad, MULTIPLE_SCATTERING_CIRS_EXPECTED_RESULT, rtol=5.0e-2), "Calculated values must be close to expected values"

def test_solar_occultation_mars():  
    '''
    Mars solar occultation test
    
    NOTE: Huge memory usage for this test.
    '''
    test_dir = os.path.join(ans.archnemesis_path(), 'tests/files/Mars_solar_occultation/')
    os.chdir(test_dir) #Changing directory to read files
    runname = 'mars_solocc'
    
    #Reading the input files
    (
        Atmosphere,
        Measurement,
        Spectroscopy,
        Scatter,
        Stellar,
        Surface,
        CIA,
        Layer,
        Variables,
        Retrieval,
        Telluric
    ) = ans.Files.read_input_files_hdf5(runname)
    
    #Calculating forward model with CIRSrad
    ForwardModel = ans.ForwardModel_0(
        runname=runname, 
        Atmosphere=Atmosphere,
        Surface=Surface,
        Measurement=Measurement,
        Spectroscopy=Spectroscopy,
        Stellar=Stellar,
        Scatter=Scatter,
        CIA=CIA,
        Layer=Layer,
        Variables=Variables
    )
    SPECONV = ForwardModel.nemesisSOfm()
    SPECONVg,dSPECONV = ForwardModel.nemesisSOfmg()
    
    #Reading the reference results
    expected_speconv = np.load('reference_result.npy')
    
    if False: # More diagnostics for testing. To enable change `if False:` to `if True:`
        import matplotlib.pyplot as plt
        
        print(f'{SPECONV.shape=} {SPECONVg.shape=} {dSPECONV.shape=}')
        print(f'{expected_speconv.shape=} ')
    
        plt.figure()
        plt.title('SPECONV')
        plt.imshow(SPECONV, cmap = 'viridis')
        
        plt.figure()
        plt.title('SPECONVg')
        plt.imshow(SPECONVg, cmap = 'viridis')
        
        plt.figure()
        plt.title('expected_speconv')
        plt.imshow(expected_speconv, cmap = 'viridis')
        
        plt.show()
        
    np.testing.assert_allclose(SPECONV, expected_speconv, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(SPECONVg, expected_speconv, rtol=1e-5, atol=1e-8)
    

def test_solar_occultation_mars_runtime():  
    '''
    Mars solar occultation test
    
    NOTE: Huge memory usage for this test.
    '''
    test_dir = os.path.join(ans.archnemesis_path(), 'tests/files/Mars_solar_occultation/')
    os.chdir(test_dir) #Changing directory to read files
    runname = 'mars_solocc_runtime'
    runname2 = 'mars_solocc'
     
    #Reading the input files
    (
        Atmosphere,
        Measurement,
        Spectroscopy,
        Scatter,
        Stellar,
        Surface,
        CIA,
        Layer,
        Variables,
        Retrieval,
        Telluric
    ) = ans.Files.read_input_files_hdf5(runname)
    
    #Calculating forward model with CIRSrad
    ForwardModel = ans.ForwardModel_0(
        runname=runname, 
        Atmosphere=Atmosphere,
        Surface=Surface,
        Measurement=Measurement,
        Spectroscopy=Spectroscopy,
        Stellar=Stellar,
        Scatter=Scatter,
        CIA=CIA,
        Layer=Layer,
        Variables=Variables
    )
    SPECONV = ForwardModel.nemesisSOfm()
    

    #Reading the input files
    (
        Atmosphere,
        Measurement,
        Spectroscopy,
        Scatter,
        Stellar,
        Surface,
        CIA,
        Layer,
        Variables,
        Retrieval,
        Telluric
    ) = ans.Files.read_input_files_hdf5(runname2)
    
    #Using only first gas in the Spectroscopy class
    Spectroscopy.NGAS = 1

    #Calculating forward model with CIRSrad
    ForwardModel = ans.ForwardModel_0(
        runname=runname, 
        Atmosphere=Atmosphere,
        Surface=Surface,
        Measurement=Measurement,
        Spectroscopy=Spectroscopy,
        Stellar=Stellar,
        Scatter=Scatter,
        CIA=CIA,
        Layer=Layer,
        Variables=Variables
    )
    SPECONV2 = ForwardModel.nemesisSOfm()

    if False: # More diagnostics for testing. To enable change `if False:` to `if True:`
        import matplotlib.pyplot as plt
        
        print(f'{SPECONV.shape=} {SPECONV2.shape=}')
    
        fig,ax1 = plt.subplots(1,1,figsize=(12,3))
        igeom = -1
        ax1.plot(Measurement.VCONV[:,igeom],SPECONV[:,igeom],label="Runtime",linewidth=0.5)
        ax1.plot(Measurement.VCONV[:,igeom],SPECONV2[:,igeom],label="Look-up tables",linewidth=0.5)
        ax1.legend()
        plt.tight_layout()
        fig.savefig("mars_solocc_runtime.png",dpi=300)
        plt.show()
        
        
    np.testing.assert_allclose(SPECONV, SPECONV2, rtol=0.1)
    


def test_titan_avefov():  
    '''
    Titan test where several geometry across the planet are averaged
    '''
    test_dir = os.path.join(ans.archnemesis_path(), 'tests/files/Titan_aveFOV/')
    os.chdir(test_dir) #Changing directory to read files
    runname = 'ch3cn'
    
    #Reading the input files
    Atmosphere,Measurement,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Variables,Retrieval = ans.Files.read_input_files(runname)
    
    #Calculating forward model with CIRSrad
    ForwardModel = ans.ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,Measurement=Measurement,Spectroscopy=Spectroscopy,Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=Variables)
    SPECONV = ForwardModel.nemesisfm()
    SPECONVg,dSPECONV = ForwardModel.nemesisfmg()
    
    #Reading the file from NEMESIS
    lat,lon,ngeom,ny,wave1,expected_speconv,specmeas1,specerrmeas,nx,Var,aprprof,aprerr,retprof,reterr = ans.read_mre(runname)
    expected_speconv *= 1.0e-9
    
    #Comparing the forward models
    np.testing.assert_allclose(SPECONV, expected_speconv, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(SPECONVg, expected_speconv, rtol=1e-5, atol=1e-8)
    