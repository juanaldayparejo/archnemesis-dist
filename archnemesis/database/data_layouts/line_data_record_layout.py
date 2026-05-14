

from typing import Annotated

from archnemesis.database.data_layouts.structured_qualifiers import IDNumber, Quantity

from archnemesis.database.data_layouts.record_layout import RecordLayout


class LineDataRecordLayout(RecordLayout):
	mol_id       : Annotated[int,   IDNumber('RADTRAN ID of molecule')]
	local_iso_id : Annotated[int,   IDNumber('Isotope ID in context of parent molecule')]
	nu           : Annotated[float, Quantity('Wavenumber of line', 'cm^{-1}')]
	sw           : Annotated[float, Quantity('Line intensity at reference temperature (see containing group)', 'cm^{-1}/(molec.cm^{-2})')]
	a            : Annotated[float, Quantity('Einstein A-coefficient', 's^{-1}')]
	elower       : Annotated[float, Quantity('Lower-state energy', 'cm^{-1}')]
	gamma_self   : Annotated[float, Quantity('Self-broadened HWHM at reference temperature and pressure (see containing group)', 'cm{^-1} atm^{-1}')]
	n_self       : Annotated[float, Quantity('Temperature exponent for the self-broadened HWHM', 'NUMBER')]