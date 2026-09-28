



from typing import Annotated

from archnemesis.database.data_layouts.structured_qualifiers import IDNumber, Quantity

from archnemesis.database.data_layouts.record_layout import RecordLayout


# NOTE: A pseudo-continuum is always defined in terms of:
#         * An applicable temperature range
#         * A temperature `t_cont` at which the continuum was initially calculated (usually
#           the same as the highest applicable temperature.
#         * A line strength ceiling `s_max` above which a line is not included in the
#           pseudo-continuum and should be handled as a discrete line.
#
# NOTE: The definition data is held "above" the record level (as all records in a table share
#       the same definition data).

class PseudoContinuumDataRecordLayout(RecordLayout):
	mol_id                                         : Annotated[int,   IDNumber(
		r'RADTRAN ID of molecule'
	)]
	local_iso_id                                   : Annotated[int,   IDNumber(
		r'Isotope ID in context of parent molecule'
	)]
	wn_bin_center                                  : Annotated[float, Quantity(
		r"Wavenumber of pseudo-continuum bin center", 
		r"cm^{-1}"
	)]
	wn_bin_width                                   : Annotated[float, Quantity(
		r"Width of pseudo-continuum bin", 
		r"cm^{-1}"
	)]
	line_strength_sum                              : Annotated[float, Quantity(
		r"Sum of line strengths in this pseudo-continuum bin at reference temperature (see containing group) [\sum{S_{i}}]", 
		r'cm^{-1}/(molec.cm^{-2})'
	)]
	line_strength_weighted_mean_lower_energy_state : Annotated[float, Quantity(
		r"Mean lower energy state of lines in this pseudo-continuum bin weighted by line strength at reference temperature (see containing group) [\frac{\sum{{S_{i} E_{\textrm{lower}}}}{\sum{S_{i}}}]", 
		r"cm^{-1}"
	)]
	line_strength_weighted_gamma_self              : Annotated[float, Quantity(
		r"Mean lorentzian HWHM of lines in this pseudo-continuum bin weighted by line strength at reference temperature and pressure (see containing group) [\frac{\sum{{S_{i} \gamma_{\textrm{self}}}}{\sum{S_{i}}}]", 
		r'cm{^-1} atm^{-1}'
	)]
	line_strength_weighted_n_self                  : Annotated[float, Quantity(
		r"Mean temperature exponent for lorentzian HWHM of lines in this pseudo-continuum bin weighted by line strength at reference temperature (see containing group) [\frac{\sum{{S_{i} n_{\textrm{self}}}}{\sum{S_{i}}}]", 
		r"NUMBER"
	)]
	

class PseudoContinuumBroadenerRecordLayout(RecordLayout):
	line_strength_weighted_gamma_amb               : Annotated[float, Quantity(
		r"Mean ambient gas lorentzian HWHM of lines in this pseudo-continuum bin weighted by line strength at reference temperature and pressure [\frac{\sum{{S_{i} \gamma_{\textrm{self}}}}{\sum{S_{i}}}]", 
		r'cm{^-1} atm^{-1}'
	)]
	line_strength_weighted_n_amb                   : Annotated[float, Quantity(
		r"Mean ambient gas temperature exponent for lorentzian HWHM of lines in this pseudo-continuum bin weighted by line strength at reference temperature [\frac{\sum{{S_{i} n_{\textrm{self}}}}{\sum{S_{i}}}]", 
		r"NUMBER"
	)]
	