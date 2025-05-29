"""
Configures logging for the package
"""
import sys
import logging

pkg_lgr = logging.getLogger(__name__.split('.',1)[0])
pkg_lgr.propagate = False

pkg_lgr.setLevel(logging.DEBUG)

pkg_stream_hdlr = logging.StreamHandler()
pkg_stream_hdlr.setLevel(logging.DEBUG)

pkg_stream_hdlr_formatter = logging.Formatter('%(levelname)s :: %(funcName)s :: %(filename)s-%(lineno)d :: %(message)s')
pkg_stream_hdlr.setFormatter(pkg_stream_hdlr_formatter)


pkg_lgr.addHandler(pkg_stream_hdlr)

def set_packagewide_level(log_level : int, mode : str = 'exact', _lgr : logging.Logger = pkg_lgr):
    """
    Sets the logging level for the whole package at once.
    
    ## ARGUMENTS ##
        log_level : int
            Level (e.g. logging.DEBUG) that all loggers in the package should be set to
        
        mode : str{'exact', 'min', 'max'} = 'exact'
            How the level should be set. 
                'exact' - set all loggers to the passed `log_level`
                'min' - set all loggers to have a most the passed `log_level`
                'max' - set all loggers to have at least the passed `log_level`
        
        _lgr : logging.Logger = pkg_lgr
            The highest logger in the logging hierarchy that will be affected.
    
    ## RETURNS ##
        None
    """
    if mode == 'exact':
        _lgr.setLevel(log_level)
    elif mode == 'max':
        if _lgr.level > log_level:
            _lgr.setLevel(log_level)
    elif mode == 'min':
        if _lgr.level < log_level:
            _lgr.setLevel(log_level)
    else:
        raise ValueError(f'{__name__}.set_packagewide_level(...): Unknown mode "{mode}", should be one of ("exact", "min", "max")')
    
    if sys.version_info >= (3,12):
        for child_lgr in _lgr.getChildren():
            set_packagewide_level(log_level, mode, child_lgr)
    else:
        for name, child_lgr in ((name, l) for name, l in logging.root.manager.loggerDict.items() if ((not isinstance(l, logging.PlaceHolder)) and name.startswith(_lgr.name) and (len(name[len(_lgr.name):].split('.'))==2))) :
            set_packagewide_level(log_level, mode, child_lgr)
