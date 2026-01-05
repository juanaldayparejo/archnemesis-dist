"""
Configures logging for the package
"""
import logging

pkg_lgr = logging.getLogger(__name__.split('.',1)[0])
pkg_lgr.propagate = False

pkg_lgr.setLevel(logging.INFO)

pkg_stream_hdlr = logging.StreamHandler()
pkg_stream_hdlr.setLevel(logging.INFO)

pkg_stream_hdlr_formatter = logging.Formatter('%(levelname)s :: %(funcName)s :: %(filename)s-%(lineno)d :: %(message)s')
pkg_stream_hdlr.setFormatter(pkg_stream_hdlr_formatter)


pkg_lgr.addHandler(pkg_stream_hdlr)

_logger_levels : dict[str,list[int]] = dict()


def get_all_logger_decendents(_lgr):
    """
    Generator that iterates over all decendents of `_lgr` including `_lgr` itself.
    """
    yield _lgr
    
    for name, child_lgr in ((name, l) for name, l in logging.root.manager.loggerDict.items() if ((not isinstance(l, logging.PlaceHolder)) and name.startswith(_lgr.name) and (len(name[len(_lgr.name):].split('.'))>=2))) :
        yield child_lgr

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
    for decendent_lgr in get_all_logger_decendents(_lgr):
    
        if mode == 'exact':
            #print(f'Setting logger "{decendent_lgr.name}" to {log_level}')
            decendent_lgr.setLevel(log_level)
        elif mode == 'max':
            if decendent_lgr.level > log_level:
                #print(f'Setting logger "{decendent_lgr.name}" to {log_level}')
                decendent_lgr.setLevel(log_level)
        elif mode == 'min':
            if decendent_lgr.level < log_level:
                #print(f'Setting logger "{decendent_lgr.name}" to {log_level}')
                decendent_lgr.setLevel(log_level)
        else:
            raise ValueError(f'{__name__}.set_packagewide_level(...): Unknown mode "{mode}", should be one of ("exact", "min", "max")')


def push_packagewide_level(log_level : int, mode : str = 'exact', _lgr : logging.Logger = pkg_lgr):
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
    #global _logger_levels
    
    for decendent_lgr in get_all_logger_decendents(_lgr):
        level_stack = _logger_levels.get(decendent_lgr.name, list())
        level_stack.append(decendent_lgr.level)
        _logger_levels[decendent_lgr.name] = level_stack
        
        if mode == 'exact':
            #print(f'Setting logger "{decendent_lgr.name}" to {log_level}')
            decendent_lgr.setLevel(log_level)
        elif mode == 'max':
            if decendent_lgr.level > log_level:
                #print(f'Setting logger "{decendent_lgr.name}" to {log_level}')
                decendent_lgr.setLevel(log_level)
        elif mode == 'min':
            if decendent_lgr.level < log_level:
                #print(f'Setting logger "{decendent_lgr.name}" to {log_level}')
                decendent_lgr.setLevel(log_level)
        else:
            raise ValueError(f'{__name__}.set_packagewide_level(...): Unknown mode "{mode}", should be one of ("exact", "min", "max")')


def pop_packagewide_level(_lgr : logging.Logger = pkg_lgr) -> None:
    #global _logger_levels
    
    for decendent_lgr in get_all_logger_decendents(_lgr):
    
        level_stack = _logger_levels.get(decendent_lgr.name, list())
        if len(level_stack) != 0:
            decendent_lgr.setLevel(level_stack.pop(-1))
            _logger_levels[decendent_lgr.name] = level_stack