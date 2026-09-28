"""
Configures logging for the package
"""
import sys
import logging

NOTSET = logging.NOTSET
DEBUG = logging.DEBUG
INFO = logging.INFO
WARN = logging.WARN
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

pkg_lgr = logging.getLogger("archnemesis")
pkg_lgr.setLevel(logging.DEBUG)
pkg_lgr.propagate = False
pkg_formatter = logging.Formatter('%(levelname)s :: %(funcName)s :: %(filename)s-%(lineno)d :: %(message)s')

progress_lgr = logging.getLogger("archnemesis.progress")
progress_lgr.setLevel(logging.DEBUG)
progress_lgr.propagate = False
progress_formatter = logging.Formatter('%(levelname)s :: %(funcName)s :: PROGRESS :: %(message)s')

class LogLevelFilter():
    def __init__(self, level : int, low_pass : bool = True):
        self.level = level
        self.low_pass = low_pass
    
    def filter(self, record):
        #print(f'LogLevelFilter::filter {self.level=} {self.low_pass=} {record=}')
        if self.low_pass:
            return record.levelno <= self.level
        else:
            return record.levelno > self.level

filter_pass_warning_and_below = LogLevelFilter(logging.WARN, low_pass=True)
filter_pass_above_warning = LogLevelFilter(logging.WARN, low_pass=False)

log_stdout_hdlr = logging.StreamHandler(sys.stdout)
log_stderr_hdlr = logging.StreamHandler(sys.stderr)

log_stdout_hdlr.setFormatter(pkg_formatter)
log_stderr_hdlr.setFormatter(pkg_formatter)

log_stdout_hdlr.addFilter(filter_pass_warning_and_below)
log_stderr_hdlr.addFilter(filter_pass_above_warning)

pkg_lgr.addHandler(log_stdout_hdlr)
pkg_lgr.addHandler(log_stderr_hdlr)


progress_stdout_hdlr = logging.StreamHandler(sys.stdout)
progress_stderr_hdlr = logging.StreamHandler(sys.stderr)

progress_stdout_hdlr.setFormatter(progress_formatter)
progress_stderr_hdlr.setFormatter(progress_formatter)

progress_stdout_hdlr.addFilter(filter_pass_warning_and_below)
progress_stderr_hdlr.addFilter(filter_pass_above_warning)

progress_stdout_hdlr.terminator = '\r' if sys.stdout.isatty() else '\n'
progress_stderr_hdlr.terminator = '\r' if sys.stderr.isatty() else '\n'

progress_lgr.addHandler(progress_stdout_hdlr)
progress_lgr.addHandler(progress_stderr_hdlr)


class LogAdaptorFilter():
    def __init__(self):
        #print( 'LogAdaptorFilter::__init__')
        self.adaptor_level_map = dict()
    
    def checkLevel(self, adaptor_name : str, level : int) -> bool:
        #print(f'LogAdaptorFilter::checkLevel {adaptor_name=} {level=} {self.getLevel(adaptor_name)=}')
        #print(f'{self.getLevel(adaptor_name) <= level=}')
        return self.getLevel(adaptor_name) <= level
    
    def filter(self, record) -> bool:
        #print(f'LogAdaptorFilter::filter {record=}')
        return self.checkLevel(record.name, record.levelno)
    
    def setLevel(self, adaptor_name : str, level : int):
        #print(f'LogAdaptorFilter::setLevel {adaptor_name=} {level=}')
        self.adaptor_level_map[adaptor_name] = level
    
    def getLevel(self, adaptor_name : str):
        #print(f'LogAdaptorFilter::getLevel {adaptor_name=}')
        return self.adaptor_level_map.get(adaptor_name, logging.NOTSET)
    
    

log_adaptor_filter = LogAdaptorFilter()

log_stdout_hdlr.addFilter(log_adaptor_filter)
log_stderr_hdlr.addFilter(log_adaptor_filter)

progress_stdout_hdlr.addFilter(log_adaptor_filter)
progress_stderr_hdlr.addFilter(log_adaptor_filter)

class ArchnemesisLoggerAdaptor(logging.LoggerAdapter):
    def __init__(self, name, logger):
        super().__init__(logger, extra={"adaptor_name": name})
        self.adaptor_name = name
        self.level = logging.NOTSET
        
    def setLevel(self, level):
        #print(f'ArchnemesisLoggerAdaptor::setLevel() {level=}')
        self.level = level
        log_adaptor_filter.setLevel(self.adaptor_name, level)
    
    def isEnabledFor(self, level):
        #print(f'ArchnemesisLoggerAdaptor::isEnabledFor() {level=}')
        return log_adaptor_filter.checkLevel(self.adaptor_name, level)
    
    def getEffectiveLevel(self):
        #print( 'ArchnemesisLoggerAdaptor::getEffectiveLevel()')
        return log_adaptor_filter.getLevel(self.adaptor_name)
    
    def process(self, msg, kwargs):
        #print(f'ArchnemesisLoggerAdaptor::process() {msg=} {kwargs=}')
        return msg, kwargs

Logger = ArchnemesisLoggerAdaptor


def getLogger(name : str, progress : bool = False):
    #print(f'getLogger() {name=} {progress=}')
    if name.startswith('archnemesis.'):
        name = name[len('archnemesis.'):]
    name = f"{progress_lgr.name if progress else pkg_lgr.name}.{name}"
    #print(f'{name=}')
    return ArchnemesisLoggerAdaptor(
        name,
        progress_lgr if progress else pkg_lgr,
    )

def setLogFormat(fmt_str : str, progress : bool = False):
    formatter = logging.Formatter(fmt_str)
    if progress:
        progress_stdout_hdlr.setFormatter(formatter)
        progress_stderr_hdlr.setFormatter(formatter)
    else:
        log_stdout_hdlr.setFormatter(formatter)
        log_stderr_hdlr.setFormatter(formatter)

def resetLogFormat(progress : bool = False):
    if progress:
        progress_stdout_hdlr.setFormatter(progress_formatter)
        progress_stderr_hdlr.setFormatter(progress_formatter)
    else:
        log_stdout_hdlr.setFormatter(pkg_formatter)
        log_stderr_hdlr.setFormatter(pkg_formatter)

def __getattr__(name):
    return getattr(logging, name)