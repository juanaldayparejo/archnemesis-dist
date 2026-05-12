

from archnemesis.database.data_layouts.record_layout import RecordLayout
from archnemesis.database.data_layouts.pseudo_continuum_record_layout import PseudoContinuumBroadenerRecordLayout
from archnemesis.database.data_layouts.pseudo_continuum_record_layout import PseudoContinuumDataRecordLayout

from .base import BaseTableWriter


class PseudoContinuumBroadenerTableWriter(BaseTableWriter):
	record_format : type[RecordLayout] = PseudoContinuumBroadenerRecordLayout

class PseudoContinuumDataTableWriter(BaseTableWriter):
	record_format : type[RecordLayout] = PseudoContinuumDataRecordLayout