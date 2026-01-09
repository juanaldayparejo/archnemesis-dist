
# Mapping between RADTRAN and EXOMOL isotopologue identifiers
# Format: (radtran_molecule_id, radtran_isotopologue_id) : (exomol_molecule_id, exomol_isotopologue_id)
radtran_to_exomol : dict[tuple[int,int], tuple[int,int]] = {

}

exomol_to_radtran : dict[tuple[int,int], tuple[int,int]] = {v: k for k, v in radtran_to_exomol.items()}