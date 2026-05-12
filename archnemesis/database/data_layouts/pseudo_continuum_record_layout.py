



from typing import Annotated

from archnemesis.database.data_layouts.structured_qualifiers import IDNumber, Quantity

from archnemesis.database.data_layouts.record_layout import RecordLayout


# NOTE: A pseudo-continuum is always defined in terms of an applicable temperature range and
#       a temperature (t_cont) at which the continuum was calculated (usually the same as the
#       highest applicable temperature.
#
# NOTE: The temperature data is held "above" the record level (as all records in a table share
#       the same temperature data).

class PseudoContinuumDataRecordLayout(RecordLayout):
	mol_id                                         : Annotated[int,   IDNumber(
		'RADTRAN ID of molecule'
	)]
	local_iso_id                                   : Annotated[int,   IDNumber(
		'Isotope ID in context of parent molecule'
	)]
	wn_bin_center                                  : Annotated[float, Quantity(
		"Wavenumber of pseudo-continuum bin center", 
		"cm^{-1}"
	)]
	wn_bin_width                                   : Annotated[float, Quantity(
		"Width of pseudo-continuum bin", 
		"cm^{-1}"
	)]
	line_strength_sum                              : Annotated[float, Quantity(
		"Sum of line strengths in this pseudo-continuum bin [\sum{S_{i}}]", 
		'cm^{-1}/(molec.cm^{-2})'
	)]
	line_strength_weighted_mean_lower_energy_state : Annotated[float, Quantity(
		"Mean lower energy state of lines in this pseudo-continuum bin weighted by line strength [\frac{\sum{{S_{i} E_{\textrm{lower}}}}{\sum{S_{i}}}]", 
		"cm^{-1}"
	)]
	line_strength_weighted_gamma_self              : Annotated[float, Quantity(
		"Mean lorentzian HWHM of lines in this pseudo-continuum bin weighted by line strength [\frac{\sum{{S_{i} \gamma_{\textrm{self}}}}{\sum{S_{i}}}]", 
		'cm{^-1} atm^{-1}'
	)]
	line_strength_weighted_n_self                  : Annotated[float, Quantity(
		"Mean temperature exponent for lorentzian HWHM of lines in this pseudo-continuum bin weighted by line strength [\frac{\sum{{S_{i} n_{\textrm{self}}}}{\sum{S_{i}}}]", 
		"NUMBER"
	)]
	

class PseudoContinuumBroadenerRecordLayout(RecordLayout):
	line_strength_weighted_gamma_amb               : Annotated[float, Quantity(
		"Mean ambient gas lorentzian HWHM of lines in this pseudo-continuum bin weighted by line strength [\frac{\sum{{S_{i} \gamma_{\textrm{self}}}}{\sum{S_{i}}}]", 
		'cm{^-1} atm^{-1}'
	)]
	line_strength_weighted_n_amb                   : Annotated[float, Quantity(
		"Mean ambient gas temperature exponent for lorentzian HWHM of lines in this pseudo-continuum bin weighted by line strength [\frac{\sum{{S_{i} n_{\textrm{self}}}}{\sum{S_{i}}}]", 
		"NUMBER"
	)]
	