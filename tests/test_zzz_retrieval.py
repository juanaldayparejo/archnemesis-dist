from pathlib import Path
import contextlib

import pytest  
import numpy as np

import archnemesis as ans
import archnemesis.Data.path_data
from archnemesis.Retrieval import Retrieval


THERMAL_EMISSION_CIRS_EXPECTED_RESULT = np.load(
    ans.Data.path_data.archnemesis_resolve_path("ARCHNEMESIS_PATH/tests/data/thermal_emission_cirs_expected_result.npy"), 
    allow_pickle=False
)


@pytest.mark.parametrize(
    'test_ans_directory, test_runname, expected_result', 
    [
        ('tests/files/Jupiter_CIRS_nadir_thermal_emission/', 'cirstest', THERMAL_EMISSION_CIRS_EXPECTED_RESULT), 
    ]
)
def test_thermal_emission_cirs(
        test_ans_directory : Path | str, 
        test_runname : str, 
        expected_result : np.ndarray,
):
    '''
    Jupiter thermal emission test against NEMESIS
    '''
    unit_conversion_factor : float = np.nan
    test_dir = Path(ans.archnemesis_path()) / test_ans_directory

    with contextlib.chdir(test_dir):
        prev_dir_contents = tuple(test_dir.iterdir())
        try:
            retrieval = Retrieval.from_legacy(test_runname)
            retrieval.run_optimal_estimation()
            
            lat,lon,ngeom,ny,wave,specret,specmeas,specerrmeas,nx,Var,aprprof,aprerr,retprof,reterr = ans.read_mre(test_runname)
        
            # ensure that the units are what we expect
            with open(f'{test_runname}.mre') as f:
                if "Radiances expressed as nW cm-2 sr-1 (cm-1)-1" in f.read():
                    unit_conversion_factor = 1E-9
        finally:
            # Remove any created files
            for item in test_dir.iterdir():
                if item not in prev_dir_contents:
                    print(f'Removing file created by test at {item}')
                    if item.is_file():
                        item.unlink()
                    else:
                        raise RuntimeWarning(f'Subdirectory was created at {item}. Will not be removed after test is completed')
    
    assert specret.shape[1] == (1 if expected_result.ndim==1 else expected_result.shape[1]), "Calculated spectra must have same numbers of geometry as expected data"
    
    if np.isnan(unit_conversion_factor):
        raise RuntimeError(f'Unknown units in {test_runname}.mre, cannot find unit conversion factor.')
    
    if expected_result.ndim == 1:
        calculated_spectra = specret[:,0] * unit_conversion_factor
        wave = wave[:,0]
    else:
        calculated_spectra = specret * unit_conversion_factor
        
    print(f'{expected_result.shape=} {calculated_spectra.shape=} {wave.shape=}')
    
    atol=np.quantile(expected_result, 0.5) * 1E-2
    rtol = 5E-2
    
    if False: # Extra diagnostic information if required
        import matplotlib.pyplot as plt
        
        tol = atol + rtol*np.abs(expected_result)
        
        plt.figure()
        plt.title('Expected vs Calculated result')
        lines = plt.plot(wave, expected_result, marker='none', ls='-', alpha=0.6, label='expected_result')
        plt.plot(wave, calculated_spectra, marker='none', ls='-', alpha=0.6, label='calculated_spectra')
        plt.fill_between(wave, expected_result - tol, expected_result+tol, alpha=0.1, color=lines[0].get_color(), label='expected region')
        plt.xlabel('Wave ($cm^{-1}$)')
        plt.ylabel('Radiance ($W cm^{-2} sr^{-1} (cm^{-1})^{-1})$)')
        plt.legend()
        
        plt.figure()
        plt.title('Residual (calculated - expected)')
        plt.plot(wave, calculated_spectra - expected_result, marker='none', ls='-', color='tab:red', alpha=0.6, label='residual')
        plt.fill_between(wave, -1*tol, tol, alpha=0.1, color=lines[0].get_color(), label='expected region')
        plt.xlabel('Wave ($cm^{-1}$)')
        plt.ylabel('Radiance ($W cm^{-2} sr^{-1} (cm^{-1})^{-1})$)')
        plt.legend()
        
        plt.show()
    
    
    # Use a NumPy comparison for arrays
    np.testing.assert_allclose(calculated_spectra, expected_result, atol=atol, rtol=rtol)
