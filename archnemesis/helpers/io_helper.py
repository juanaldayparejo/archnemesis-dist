from __future__ import annotations #  for 3.9 compatability

import sys, os
from typing import IO, Callable, Self
import time
import datetime as dt


import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

_default_out_width = 80
_set_out_width = []

def time_sec_to_hms(t):
        th = int(t//3600)
        t -= 3600*th
        tm = int(t//60)
        t -= 60*tm
        return f'{th:02}:{tm:02}:{t:06.3f}'

class OutWidth:
    _default_out_width = 80
    _set_out_width = []

    @classmethod
    def get(cls, f : IO = sys.stdout) -> int:
        """
        Get the widest string an output file descriptor can cope with.
        
        `f` is "stdout" by default.
        """
        if f.isatty():
            tty_cols = os.get_terminal_size().columns
            if len(cls._set_out_width) != 0 and cls._set_out_width[-1] < tty_cols:
                return cls._set_out_width[-1]
            else:
                return tty_cols
        else:
            return cls._default_out_width if len(cls._set_out_width) == 0  else cls._set_out_width[-1]

    @classmethod
    def set(cls, width=None):
        
        if width == None:
            cls._set_out_width = []
        else:
            if len(cls._set_out_width) != 0:
                cls._set_out_width[-1] = width
            else:
                cls._set_out_width = [width]
    
    @classmethod
    def push(cls,width):
        cls._set_out_width.append(width)

    @classmethod
    def pop(cls):
        if len(cls._set_out_width) != 0:
            cls._set_out_width = cls._set_out_width[:-1]


class SimpleProgressTracker:
    """
    Class that displays the progress of a task, knows how many tasks are being tracked at once
    so nested trackers can be displayed or not based upon their `depth`.
    
    """
    ##  Class variables ##
    
    _class_depth = 0
    _class_max_display_depth = 1
    
    
    ## Static methods ##
    
    @staticmethod
    def getTimeNow() -> float:
        """
        Get the time in seconds the system has been running and not suspended, want to know
        how long the tracker will take to complete so cannot use wall-clock time as
        that will not account for time the machine is suspended.
        """
        return time.clock_gettime(time.CLOCK_MONOTONIC)
    
    @staticmethod
    def getDatetimeNow() -> dt.datetime:
        return dt.datetime.now()
    
    
    ## Class Methods ##
    
    @classmethod
    def inc_depth(cls) -> None:
        cls._class_depth += 1
    
    @classmethod
    def dec_depth(cls) -> None:
        cls._class_depth -= 1
    
    @classmethod
    def set_max_display_depth(cls, value : int) -> None:
        cls._class_max_display_depth = value
    
    @classmethod
    def get_max_display_depth(cls) -> int:
        return cls._class_max_display_depth
    
    
    ## Properties ##
    
    @property
    def output_target(self) -> IO | logging.Logger:
        """
        Return the stored `self._output_target`
        """
        return self._output_target

    @output_target.setter
    def output_target(self, value : str | Callable[[SimpleProgressTracker],str]):
        """
        Set the `self._output_target` attribute to something that can accept a string.
        """
        self._output_target = value
        
        if isinstance(self._output_target, logging.Logger): # Pass the message through to the logger
            self._output_fn = lambda s: self._output_target.info(s, stacklevel=4)
        elif callable(self._output_target): # `self._output_target` is already a callable and should accept a string
            self._output_fn = self._output_target
        elif isinstance(self._output_target, IO): # `self._output_target` is a file-like object, so write to it
            self._output_fn = lambda s: print(s, file=self._output_target)
        else:
            raise ValueError(f'Unknown type {type(self._output_target)} for attribute `output_target`')
        
        return
    
    @property
    def message(self) -> str:
        """
        Create the message to display by formatting or calling `self.msg_format`
        """
        if callable(self.msg_format):
            return self.msg_format(self)
        else:
            return self.msg_format.format(
                name = self.name,
                percent_complete = self.percent_complete,
                percent_remaining = self.percent_remaining,
                ratio_complete = self.ratio_complete,
                ratio_remaining = self.ratio_remaining,
                time_elapsed = self.time_elapsed,
                time_remaining = self.time_remaining,
                datetime_complete = self.datetime_complete
            )
    
    
    ## Output format properties ##
    
    @property
    def percent_complete(self) -> str:
        # Property to be used as an output format field
        return f'{100.0*self.frac_complete: 6.2f} %' if self.frac_complete is not None else 'UNKNOWN PERCENT COMPLETE'
    
    @property
    def percent_remaining(self) -> str:
        # Property to be used as an output format field
        return f'{100.0*(1-self.frac_complete): 6.2f} %' if self.frac_complete is not None else 'UNKNOWN PERCENT REMAINING'
    
    @property
    def ratio_complete(self) -> str:
        # Property to be used as an output format field
        return f'{self.n} / {self.n_max}' if self.n_max is not None else 'UNKNOWN RATIO COMPLETE'

    @property
    def ratio_remaining(self) -> str:
        # Property to be used as an output format field
        return f'{(self.n_max - self.n)} / {self.n_max}' if self.n_max is not None else 'UNKNOWN RATIO REMAINING'

    @property
    def time_elapsed(self) -> str:
        # Property to be used as an output format field
        return time_sec_to_hms(self.t_delta)
    
    @property
    def time_remaining(self) -> str:
        # Property to be used as an output format field
        return time_sec_to_hms(self.t_delta*((1.0/self.frac_complete) - 1)) if self.frac_complete is not None else "UNKNOWN TIME REMAINING"

    @property
    def datetime_complete(self) -> str:
        # Property to be used as an output format field
        return (dt.datetime.now() + dt.timedelta(seconds = self.t_delta*((1.0/self.frac_complete)-1))).isoformat() if self.frac_complete is not None else "UNKNOWN DATETIME OF COMPLETION"
    
    
    ## Instance Methods ##
    
    def __init__(self, 
            name : str, # Name of tracker (required)
            n_max : None | int = None, # Maximum number of iterations the tracker will run for, or `None` if that is unknown
            *, # end positional arguments
            msg_format : str | Callable[[SimpleProgressTracker],str] = None, # Format string or callable that takes a `SimpleProgressTracker` object and returns a string to be displayed
            display_interval_sec : float = 5, # How often (in seconds) to display the tracker progress
            display_interval_n : int = 1, # How often (in number of iterations) to calculate and display tracker progress.
            output_target : IO | logging.Logger = sys.stdout # tracker output will be sent to this object
    ):
        
        self._output_fn : Callable[[str],None] = None
        
        # Assign arguments
        self.name : str = name
        self.n_max : None | int = n_max
        
        if msg_format is None:
            if self.n_max is None:
                self.msg_format = "{name} Iteration {n} Time/Iteration {hms_per_iteration}"
            else:
                self.msg_format = "{name} Progress: {ratio_complete} [{percent_complete}] Time: Elapsed {time_elapsed} Est. Remaining {time_remaining} Est. Completion {datetime_complete}"
        else:
            self.msg_format = msg_format
        
        self.display_interval_sec : float = display_interval_sec
        self.display_interval_n : int = display_interval_n
        self.output_target : IO | logging.Logger = output_target
        
        # Assign internal attributes
        self._creation_depth : int = self._class_depth
        self.depth : int = self._creation_depth
        self.n : int = 0# number of iterations performed (usually out of self.n_max)
        self.t_start : float = None
        self.t_now : None | float = None
        self.t_delta : float = None
        self.frac_complete : None | float = None
        self.t_last_display : float = 0
        
        return
    
    def check_depth(self) -> bool:
        # Should progress tracker display based on depth
        return self.depth < self.get_max_display_depth()
    
    def check_n_interval(self) -> bool:
        # Should progress tracker display based on iteration interval
        return (self.n % self.display_interval_n) == 0
    
    def check_t_interval(self) -> bool:
        # Should progress tracker display based on time interval
        return (self.t_last_display + self.display_interval_sec) < self.t_now
    
    def output(self, msg : str) -> Self:
        """
        Output `msg` to `self.output_target`, performing all output checks
        """
        if (
            self.check_depth()
            and self.check_n_interval()
            and self.check_t_interval()
        ):
            self._output_fn(msg)
            self.t_last_display = self.t_now
            return self
    
    def display(self) -> None:
        """
        Display the status of the progress tracker and update
        """
        if self.n > 0:
            self.update()
            self.output(self.message)
        self.increment()
    
    
    def __repr__(self):
        """
        Get a string representation of the tracker
        """
        return f'{self.__class__.__name__}(name={self.name})'
    
    def __enter__(self) -> Self:
        """
        Method is run upon entering a context
        """
        self.depth = self._class_depth
        self.inc_depth()
        self.begin()
        return self
    
    def __exit__(self, typ, value, traceback):
        """
        Method is run upon exiting a context
        """
        self.dec_depth()
        self.depth = self._creation_depth
        self.end()
        return
    
    def update(self) -> Self:
        """
        Update all dependent internal attributes
        """
        self.t_now = self.getTimeNow()
        self.t_delta = (self.t_now - self.t_start) 
        self.frac_complete : None | float = None if self.n_max is None else (self.n / self.n_max)
        return self
    
    def begin(self) -> Self:
        """
        Set independent internal attributes to be consistent with the beginning of progress tracking
        """
        self.n = 0
        self.t_start = self.getTimeNow()
        self.t_last_display = 0
        if self.check_depth():
            self._output_fn(f'{self.name} STARTING at {dt.datetime.now().isoformat()}')
    
    def end(self) -> Self:
        """
        Set independent internal attributes to be consistent with the end of progress tracking
        """
        self.display()
        if self.check_depth():
            self._output_fn(f'{self.name} FINISHED at {dt.datetime.now().isoformat()}')
    
    def increment(self) -> Self:
        """
        Increment independent internal attributes
        """
        self.n += 1
        return self
    
    def set_progress(self, value : int) -> Self:
        """
        Set values of independent internal attributes
        """
        self.n = value
        self.update()
        return self
    
        




