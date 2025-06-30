import os
from collections.abc import Sequence, Iterable
import shutil

import pytest
import numpy as np

import archnemesis as ans

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)

def ensure_equal_type(a, b):
    # 'a' and 'b' are not neccesarily the same types at this point
    type_a = type(a)
    type_b = type(b)
    _lgr.info(f'\t{type_a} vs {type_b}')
    are_types_same = type_a is type_b
    casting_rules_compared = False
    is_fwd_safe_casting_possible = False
    is_bkwd_safe_casting_possible = False
    is_fwd_samekind_casting_possible = False
    is_bkwd_samekind_casting_possible = False
    
    if (type_a == np.ndarray) and (type_b == np.ndarray):
        type_a = a.dtype
        type_b = b.dtype
        _lgr.info(f'\tnp.ndarray[{type_a}] vs np.ndarray[{type_b}]')
        are_types_same = type_a is type_b
    
    if (
            (
                np.issubdtype(type_a, np.number)
                or type_a in (int,float,complex)
            ) and (
                np.issubdtype(type_b, np.number)
                or type_b in (int,float,complex)
            )
        ):
        _lgr.info(f'\tTypes are number-like, so we can compare casting rules...')
        casting_rules_compared = True
        is_fwd_safe_casting_possible = np.can_cast(type_a, type_b, "safe")
        is_bkwd_safe_casting_possible = np.can_cast(type_b, type_a, "safe")
        is_fwd_samekind_casting_possible = np.can_cast(type_a, type_b, "same_kind")
        is_bkwd_samekind_casting_possible = np.can_cast(type_b, type_a, "same_kind")
    
    _lgr.info(f'\tTypes are identical: {are_types_same}')
    if casting_rules_compared:
        _lgr.info(f'\tSafe casting possible: fwd {is_fwd_safe_casting_possible}, bkwd {is_bkwd_safe_casting_possible}') 
        _lgr.info(f'\tSame-kind casting possible: fwd {is_fwd_samekind_casting_possible}, bkwd {is_bkwd_samekind_casting_possible}')

    return are_types_same, casting_rules_compared, is_fwd_safe_casting_possible, is_bkwd_safe_casting_possible, is_fwd_samekind_casting_possible, is_bkwd_samekind_casting_possible


def ensure_equal_value(a, b):
    # Assume that 'a' and 'b' are the same type at this point
    if issubclass(type(a), np.ndarray):
        is_same = True
        is_same = is_same and len(a.shape) == len(b.shape)
        is_same = is_same and all(s1==s2 for s1,s2 in zip(a.shape,b.shape))
        is_same = is_same and np.all(a == b)
        return is_same
    elif issubclass(type(a),Iterable) and (type(a) is not str):
        return all(ensure_equal_value(x,y) for x,y in zip(a,b))
    else:
        return a == b


def test_input_file_legacy_to_hdf5_conversion_does_not_alter_paramters():
    """
    Test that input data is consistent when we read in LEGACY format, write as HDF5, then read in the new HDF5.
    
    There should be FEW and DOCUMENTED exceptions to the assumption that input file format makes no difference.
    
    Assumptions:
        1) All attributes in ALL_CAPS are either Attributes or Parameters that we want to be consistent for each instance.
        2) Numbers (and arrays of numbers) can have different EXACT types as long as they have the same KIND of type. For
           example: int32 vs in64 is ok, but int64 vs float64 is not.
        3) Arrays of numbers must have the same shape.
    
    Exceptions:
        1) ans.Files.read_input_files(...) does not output a 'Telluric_0' instance, whereas ans.Files.read_input_files_hdf5(...) does.
        2) 'Atmosphere_0.DUST_UNITS_FLAG' and 'Layer_0.DUST_UNITS_FLAG' are always different in HDF5 format vs LEGACY. 
           HDF5 uses number density, whereas LEGACY uses mass density.
        3) 'Spectroscopy_0.RUNNAME' is not used in HDF5 format (and is barely used in LEGACY format).
        4) 'CIA_0.CIATABLE' will point to '.h5' files when using HDF5 format, and '.tab' files when using LEGACY format.
           Support code has been written to convert from '.tab' to '.h5' ciatable files.
    
    """
    starting_dir = os.getcwd()
    example_dir = os.path.join(ans.archnemesis_path(), './docs/examples')
    
    examples_to_ignore = {
        'mars_solocc': 'ktable files not available',
        'retrieval_Jupiter_Tprofile' : 'ktable files not availble'
    }
    
    for example_path in os.listdir(example_dir):
        if example_path in examples_to_ignore:
            _lgr.info(f'Ignoring example "{example_path}". Reason: {examples_to_ignore[example_path]}')
            continue
            
        current_example_dir = os.path.join(example_dir, example_path)
        _lgr.info(f'Looking for example with LEGACY input files at {current_example_dir}')
        if not os.path.isdir(current_example_dir): # skip files, only enter directories
            continue
        
        # Check we have LEGACY input files to do our test with
        
        runname = None
        for fpath in os.listdir(current_example_dir):
            if fpath.endswith('.inp'):
                runname = fpath[:-4]
                break
        
        # Change directory to example directory if we have a runname, otherwise continue searching
        if runname is not None:
            os.chdir(current_example_dir)
        else:
            continue
        
        
        # We are going to overwrite any h5_file that is already
        # present, therefore ensure we back up and restore any
        # h5 files that already exist
        h5_filename = runname+'.h5'
        backup_h5_filename = h5_filename+'.bak'
        if os.path.exists(runname+'.h5'):
            delete_h5_at_end = False
            restore_backed_up_h5_at_end = True
            shutil.move(h5_filename, backup_h5_filename)
            
        else:
            delete_h5_at_end=True
            restore_backed_up_h5_at_end = False
        
        
        # NOTE: We have put the test in a "try, finally" block
        # this is because we want to make sure we delete any
        # *.h5 files that were created so we don't pollute the file system.
        
        try:
            ###### PERFORM TEST ######
            _lgr.info('#'*40)
            _lgr.info(f'In directory "{current_example_dir}", performing input file format consistency test...')
        
            # Read LEGACY format
            Atmosphere,Measurement,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Variables,Retrieval = ans.Files.read_input_files(runname)
            
            # Write to HDF5 format
            for item in (Atmosphere,Measurement,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Retrieval):#,Variables):
                if hasattr(item, 'write_hdf5'):
                    item.write_hdf5(runname)
                elif hasattr(item, 'write_input_hdf5'):
                    item.write_input_hdf5(runname)
                else:
                    raise RuntimeError(f'For example at "{current_example_dir}", class "{type(item)}" does not have an attribute like "write{{_input,_}}hdf5", cannot continue writing new style file.')

            # Read HDF5 format
            Atmosphere_,Measurement_,Spectroscopy_,Scatter_,Stellar_,Surface_,CIA_,Layer_,Variables_,Retrieval_,Telluric_ = ans.Files.read_input_files_hdf5(runname)

            # Define the pairs of instances we want to compare
            pairs = (
                (Atmosphere, Atmosphere_),
                (Measurement, Measurement_),
                (Spectroscopy, Spectroscopy_),
                (Scatter, Scatter_),
                (Stellar, Stellar_),
                (Surface, Surface_),
                (CIA, CIA_),
                (Layer, Layer_),
                (Variables, Variables_),
                (Retrieval, Retrieval_),
                #(Telluric, Telluric_)
            )

            # Loop over instances and compare their ALL_CAPS attributes
            for old_format, new_format in pairs:
                _lgr.info(f'\nTesting integrity of {old_format.__class__.__name__}')
                # assume that everything in all caps should be identical
                of_attrs = tuple(item for item in filter(lambda x: x.isupper(), old_format.__dict__.keys()))
                nf_attrs = tuple(item for item in filter(lambda x: x.isupper(), new_format.__dict__.keys()))

                assert len(of_attrs) == len(nf_attrs), f'For example at "{current_example_dir}", class must have same number of attributes regardless of input format'
                
                _lgr.info(f'{of_attrs=}')
                _lgr.info(f'{nf_attrs=}')
                assert all(x==y for x,y in zip(of_attrs, nf_attrs)), f"For example at '{current_example_dir}', class must have same attribute names regardless of input format"

                for k in of_attrs:
                    _lgr.info(f'Testing attribute {old_format.__class__.__name__}.{k}:')
                    o_attr = getattr(old_format, k)
                    n_attr = getattr(new_format, k)

                    # Special cases
                    if f'{old_format.__class__.__name__}.{k}' == 'Atmosphere_0.DUST_UNITS_FLAG':
                        # HDF5 always uses number density (m^-3), indicated by 'None'
                        # Legacy always uses mass density (g cm^-3) indicated by a numpy array of -1 values.
                        assert type(o_attr) is np.ndarray and np.all(o_attr == -1) and n_attr is None, f"For example at '{current_example_dir}' attribute {old_format.__class__.__name__}.{k}, special case for {old_format.__class__.__name__}.{k} should be respected"
                        continue
                    if f'{old_format.__class__.__name__}.{k}' == 'Layer_0.DUST_UNITS_FLAG':
                        # HDF5 always uses number density (m^-3), indicated by 'None'
                        # Legacy always uses mass density (g cm^-3) indicated by a numpy array of -1 values.
                        assert type(o_attr) is np.ndarray and np.all(o_attr == -1) and n_attr is None, f"For example at '{current_example_dir}' attribute {old_format.__class__.__name__}.{k}, special case for {old_format.__class__.__name__}.{k} should be respected"
                        continue
                    if f'{old_format.__class__.__name__}.{k}' == 'Spectroscopy_0.RUNNAME':
                        # HDF5 files do not use this parameter
                        continue
                    if f'{old_format.__class__.__name__}.{k}' == 'CIA_0.CIATABLE':
                        if (o_attr != n_attr):
                            if o_attr.endswith('.tab') and n_attr.endswith('.h5'):
                                assert o_attr[:-4] == n_attr[:-3], f"For example at '{current_example_dir}' attribute {old_format.__class__.__name__}.{k}, different CIATABLE values must only be due to altering format from '.tab' to '.h5'"
                                continue
                        assert False, f"For example at '{current_example_dir}' attribute {old_format.__class__.__name__}.{k}, mismatched CIATABLE values '{o_attr}' vs '{n_attr}' not allowed except for format change"
                    
                    (
                        are_types_same, 
                        casting_rules_compared,
                        is_fwd_safe_casting_possible, 
                        is_bkwd_safe_casting_possible, 
                        is_fwd_samekind_casting_possible, 
                        is_bkwd_samekind_casting_possible
                    ) = ensure_equal_type(o_attr, n_attr)

                    if are_types_same:
                        pass # this is fine
                    elif (not are_types_same) and casting_rules_compared:
                        # Need to know how casting rules fail
                        if is_fwd_safe_casting_possible and is_bkwd_safe_casting_possible:
                            pass
                        elif is_fwd_samekind_casting_possible and is_bkwd_samekind_casting_possible:
                            msg = f"Types change when using different input format. They are the same kind, but safe casting is not always possible {type(o_attr)} and {type(n_attr)}"
                            _lgr.info(f'WARNING: {msg}')
                            #raise RuntimeWarning(msg)
                        else:
                            assert False, f"For example at '{current_example_dir}' attribute {old_format.__class__.__name__}.{k}, class must have attributes that are at least the same kind of type regardless of input format"
                    else:
                        # types are not the same and we didn't compare casting rules
                        assert False, f"For example at '{current_example_dir}' attribute {old_format.__class__.__name__}.{k}, class attributes must have identical types if they are not numerical regardless of input format"
                    
                    #_lgr.info(f'\t{o_attr} == {n_attr}')            
                    assert ensure_equal_value(o_attr, n_attr), f"For example at '{current_example_dir}' attribute {old_format.__class__.__name__}.{k}, class must have same values regardless of input format: {o_attr} != {n_attr}"

            ###### END TEST ######
        finally:
            # Delete any created '*.h5' file
            if delete_h5_at_end:
                os.remove(h5_filename)
            if restore_backed_up_h5_at_end:
                os.remove(h5_filename)
                shutil.move(backup_h5_filename, h5_filename)
            
            # Change back to starting directory after test has completed
            os.chdir(starting_dir)

