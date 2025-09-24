"""
Functions and classes etc. that fetch resources from the web.
"""
import urllib
import ssl
from typing import Generator

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)


PROGRESS_INTERVAL_Kb : None | float = 1024
# Interval (in kilobytes) on amount of data fetched to report progress (at log level `INFO`). If `None` will not report progress.

def file_in_chunks(
        url : str, 
        *, # All following arguments are keyword only
        chunk_size : None | int = (1024*1024), 
        encoding : str = 'ascii', 
        proxy : None | dict[str,str]
) -> Generator[bytes | str]:
    """
    Fetch a file from the web and download it in chunks of `chunk_size`
    ## ARGUMENTS ##
        url : str
            Universal Resource Location to fetch file from, written as a string.
        chunk_size : None | str = None
            Size of chunks (in bytes) to iterate through the data in the file at `url`, if
            `None` will iterate through the file in lines.
        encoding : None | str = None
            Name of file encoding ('ascii', 'utf-8', ...). If `None` will return bytes
        proxy : None | dict[str,str] = None
            `None` or a dictionary detailing proxy mappings

    ## RETURNS ##
        data_generator : Generator[bytes | str]
            A generator for the data in the file at `url`. If `encoding` is `None`
            will generate `bytes` else will generate `str`.
    """
    req = urllib.request.Request(url)
    _lgr.info(f'{url=}')
    _lgr.debug(f'{chunk_size=} {encoding=}')
    
    
    handlers = []
    
    if req.type == 'https':
        #context = ssl._create_unverified_context()
        context = ssl.create_default_context(
            ssl.Purpose.SERVER_AUTH, # SERVER_AUTH is 'we want the server to be able to authenticate us', so it is used by clients connecting to servers.
        )
        handlers.append(urllib.request.HTTPSHandler(context=context))
    elif req.type == 'http':
        handlers.append(urllib.request.HTTPHandler())
    else:
        raise urllib.error.UrlError('Unknown request type "{req.type}", cannot assign handler')
            
    if proxy is not None:
        _lgr.info('Using the following proxies:')
        for k, v in proxy.items():
            _lgr.info(f'\t{k} : {v}')
        handlers.append(urllib.request.ProxyHandler(proxy))
    
    opener = urllib.request.build_opener(*handlers)
    
    response = opener.open(url)
    
    if chunk_size is None:
        get_chunk = lambda response: response.readline()
    else:
        get_chunk = lambda response: response.read(chunk_size)
        
    if encoding is None:
        do_decode = lambda x: x
    else:
        do_decode = lambda x: x.decode(encoding)
    
    
    last_reported_size = -1E30 # very negative number so we report the first size value
    accumulated_size = 0
    i = 0
    while (size_of_current_chunk := len(chunk := get_chunk(response))) > 0:
        
        if PROGRESS_INTERVAL_Kb is not None and ((accumulated_size - last_reported_size) >= (PROGRESS_INTERVAL_Kb*1024)):
            _lgr.info(f'Fetching chunk {i}. Chunk is {size_of_current_chunk/1024} Kb. Fetched {accumulated_size/1024} Kb so far...')
            last_reported_size = accumulated_size
        
        yield do_decode(chunk)
        accumulated_size += size_of_current_chunk
        i += 1
    
    return

def file(
        url : str, 
        *, # All following arguments are keyword only
        to_fpath : None | str = None, 
        encoding : None | str = None, 
        proxy : None | dict[str,str] = None
) -> None | bytes | str:
    """
    ## ARGUMENTS ##
        url : str
            Universal Resource Location to fetch file from, written as a string.
        to_fpath : None | str = None
            filepath to save file to. If present will save data to the file and
            return `None`, otherwise will not save file and will return
            the data instead.
        encoding : None | str = None
            Name of file encoding ('ascii', 'utf-8', ...). If `None` will return bytes
        proxy : None | dict[str,str] = None
            `None` or a dictionary detailing proxy mappings

    ## RETURNS ##
        data : None | bytes | str
            If `to_fpath` is not `None` will return data from the file at the `url`.
            Otherwise will write the data to a file at `to_fpath` and return `None`.
    """
    file_chunk_generator = file_in_chunks(url, encoding=encoding, proxy=proxy)
    
    
    if to_fpath is not None:
        _lgr.info(f"Downloading from {url} and saving to path '{to_fpath}'")
        
        write_mode = 'wb' if encoding is None else 'w'
        
        with open(to_fpath, write_mode) as f:
            for chunk in file_chunk_generator:
                f.write(chunk)
        return
    else:
        join_str = b'' if encoding is None else ''
        return join_str.join(file_chunk_generator)
   
