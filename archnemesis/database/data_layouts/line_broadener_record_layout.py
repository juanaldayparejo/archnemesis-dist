

from typing import Annotated

from archnemesis.database.data_layouts.structured_qualifiers import Quantity

from archnemesis.database.data_layouts.record_layout import RecordLayout



class LineBroadenerRecordLayout(RecordLayout):
	gamma_amb    : Annotated[float, Quantity('Ambient gas broadened Lorentzian half-width at half-maximum at reference temperature and pressure (see containing group)', 'cm{^-1} atm^{-1}')]
	n_amb        : Annotated[float, Quantity('Temperature exponent for the ambient gas broadened HWHM', 'NUMBER')]
	delta_amb    : Annotated[float, Quantity('Pressure shift induced by ambient gas at reference pressure (see containing group)', 'cm^{-1} atm^{-1}')]
