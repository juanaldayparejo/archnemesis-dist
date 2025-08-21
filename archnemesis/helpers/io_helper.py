from __future__ import annotations #  for 3.9 compatability

import sys, os
from typing import IO, Any, Callable
import time


import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

_default_out_width = 80
_set_out_width = []

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


def time_sec_to_hms(t):
    th = int(t//3600)
    t -= 3600*th
    tm = int(t//60)
    t -= 60*tm
    return f'{th:02}:{tm:02}:{t:06.3f}'

class SimpleProgressTracker:
    def __init__(self, n_max, msg_prefix : str = "", display_interval_sec = 5, target_logger=None):
        self.n_max = n_max
        self.display_interval_sec = display_interval_sec
        self.msg_prefix = msg_prefix if msg_prefix[-1] in (' ', '\n') else (msg_prefix+' ')
        self.target_logger = target_logger
        self.reset()
    
    @property
    def n(self):
        return self._n
    
    @n.setter
    def n(self, value):
        self.t_now = time.time()
        self._n = value
    
    def reset(self):
        self._n=0
        self.t_start = time.time()
        self.t_now = None
        self.t_last_display = 0
    
    def increment(self):
        self.n = self.n+1
    
    def get_message(self):
        if self.t_now is None:
            return f'{self.msg_prefix}Progress: {self.n} / {self.n_max} [{100.0*self.n/self.n_max: 6.2f} %] Time: UNKNOWN'
        delta_t = self.t_now - self.t_start
        remaining_t = ((self.n_max/self.n - 1)*delta_t) if (self.n>0) else None
        return f'{self.msg_prefix}Progress: {self.n} / {self.n_max} [{100.0*self.n/self.n_max: 6.2f} %] Time: Elapsed {time_sec_to_hms(delta_t)} Est. Remaining {"UNKNOWN" if remaining_t is None else time_sec_to_hms(remaining_t)}'
    
    def get_message_and_increment(self):
        msg = self.get_message()
        self.increment()
        return msg
    
    def display(self, f : IO = sys.stdout, end : str = '\n'):
        t = time.time()
        dt_last_display = t - self.t_last_display
        if dt_last_display > self.display_interval_sec:
            f.write(self.get_message()+end)
            self.t_last_display = t
    
    def display_and_increment(self, f : IO = sys.stdout, end : str = '\n'):
        self.display(f, end)
        self.increment()
    
    def send_to(self, target : Callable[[str,...],Any] = lambda x: print(x,end=None), end : str = '\n'):
        t = time.time()
        dt_last_display = t - self.t_last_display
        if dt_last_display > self.display_interval_sec:
            target(self.get_message()+end)
            self.t_last_display = t
    
    def send_to_and_increment(self, target : Callable[[str,...],Any] = lambda x: print(x,end=None), end : str = '\n'):
        self.send_to(target, end)
        self.increment()
    
    def log_at(self, level, rate_limit=True, stacklevel=2, **kwargs):
        t = time.time()
        dt_last_display = t - self.t_last_display
        if (not rate_limit) or (dt_last_display > self.display_interval_sec):
            if self.target_logger is None:
                self.display()
            else:
                self.target_logger.log(level, self.get_message(), stacklevel=stacklevel, **kwargs)
            self.t_last_display = t
        return
    
    def log_at_and_increment(self, level, rate_limit=True, stacklevel=2, **kwargs):
        self.log_at(level, rate_limit=rate_limit, stacklevel=stacklevel+1, **kwargs)
        self.increment()
        return
    
    def __repr__(self):
            return f'{self.__class__.__name__}(n_max={self.n_max}, n={self.n}, t_start={self.t_start}, t_now={self.t_now}, display_interval_sec={self.display_interval_sec}, msg_prefix={self.msg_prefix})'
        




