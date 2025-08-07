from __future__ import annotations #  for 3.9 compatability

import json
import numpy as np
import numpy.ma
import re

import hapi.hapi as hapi



def storage2cache_MONKEYPATCH(TableName,cast=True,ext=None,nlines=None,pos=None):
    """ edited by NHL
    TableName: name of the HAPI table to read in
    ext: file extension
    nlines: number of line in the block; if None, read all line at once 
    pos: file position to seek
    """
    #print 'storage2cache:'
    #print('TableName',TableName)
    if nlines is not None:
        print('WARNING: storage2cache is reading the block of maximum %d lines'%nlines)
    fullpath_data,fullpath_header = hapi.getFullTableAndHeaderName(TableName,ext)
    if TableName in hapi.LOCAL_TABLE_CACHE and \
       'filehandler' in hapi.LOCAL_TABLE_CACHE[TableName] and \
       hapi.LOCAL_TABLE_CACHE[TableName]['filehandler'] is not None:
        InfileData = hapi.LOCAL_TABLE_CACHE[TableName]['filehandler']
    else:
        InfileData = hapi.open_(fullpath_data,'r')            
    InfileHeader = open(fullpath_header,'r')
    #try:
    header_text = InfileHeader.read()
    try:
        Header = json.loads(header_text)
    except:
        print('HEADER:')
        print(header_text)
        raise Exception('Invalid header')
    #print 'Header:'+str(Header)
    hapi.LOCAL_TABLE_CACHE[TableName] = {}
    hapi.LOCAL_TABLE_CACHE[TableName]['header'] = Header
    hapi.LOCAL_TABLE_CACHE[TableName]['data'] = hapi.CaselessDict()
    hapi.LOCAL_TABLE_CACHE[TableName]['filehandler'] = InfileData
    # Check if Header['order'] and Header['extra'] contain
    #  parameters with same names, raise exception if true.
    #intersct = set(Header['order']).intersection(set(Header.get('extra',[])))
    intersct = set(Header.get('order',[])).intersection(set(Header.get('extra',[])))
    if intersct:
        raise Exception('Parameters with the same names: {}'.format(intersct))
    # initialize empty data to avoid problems
    glob_order = []; glob_format = {}; glob_default = {}
    if "order" in hapi.LOCAL_TABLE_CACHE[TableName]['header'].keys():
        glob_order += hapi.LOCAL_TABLE_CACHE[TableName]['header']['order']
        glob_format.update(hapi.LOCAL_TABLE_CACHE[TableName]['header']['format'])
        glob_default.update(hapi.LOCAL_TABLE_CACHE[TableName]['header']['default'])
        for par_name in hapi.LOCAL_TABLE_CACHE[TableName]['header']['order']:
            hapi.LOCAL_TABLE_CACHE[TableName]['data'][par_name] = []
    if "extra" in hapi.LOCAL_TABLE_CACHE[TableName]['header'].keys():
        glob_order += hapi.LOCAL_TABLE_CACHE[TableName]['header']['extra']
        glob_format.update(hapi.LOCAL_TABLE_CACHE[TableName]['header']['extra_format'])
        for par_name in hapi.LOCAL_TABLE_CACHE[TableName]['header']['extra']:
            glob_default[par_name] = hapi.PARAMETER_META.get(par_name,hapi.PMETA_DEFAULT)['default_fmt']
            hapi.LOCAL_TABLE_CACHE[TableName]['data'][par_name] = []
    
    header = hapi.LOCAL_TABLE_CACHE[TableName]['header']
    if 'extra' in header and header['extra']:
        line_count = 0
        flag_EOF = False
        #line_number = hapi.LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows']
        #for line in InfileData:
        while True:
            #print '%d line from %d' % (line_count,line_number)
            #print 'line: '+line #
            if nlines is not None and line_count>=nlines: break
            line = InfileData.readline()
            if line=='': # end of file is represented by an empty string
                flag_EOF = True
                break 
            try:
                RowObject = hapi.getRowObjectFromString(line,TableName)
                line_count += 1
            except:
                continue
            #print 'RowObject: '+str(RowObject)
            hapi.addRowObject(RowObject,TableName)
        #except:
        #    raise Exception('TABLE FETCHING ERROR')
        hapi.LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows'] = line_count
    else:
        quantities = header['order']
        formats = [header['format'][qnt].lower() for qnt in quantities]
        types = {'d':int, 'f':float, 'e':float, 's':str}
        converters = []
        end = 0
        for qnt, fmt in zip(quantities, formats):
            # pre-defined positions are needed to skip the existing parameters in headers (new feature)
            if 'position' in header:
                start = header['position'].get(qnt, None) # fix for `qnt` not in header['position']
                if start is None:
                    start = end
                    header['position'][qnt] = start
            else:
                start = end
            dtype = types[fmt[-1]]
            aux = fmt[fmt.index('%')+1:-1]
            if '.' in aux:
                aux = aux[:aux.index('.')]
            size = int(aux)
            end = start + size
            def cfunc(line, dtype=dtype, start=start, end=end, qnt=qnt):
                # return dtype(line[start:end]) # this will fail on the float number with D exponent (Fortran notation)
                if dtype in (float,int): # assign NaN if value is hashtagged
                    if line[start:end].strip()=='#':
                        return np.nan
                if dtype==float:
                    try:
                        return dtype(line[start:end])
                    except ValueError: # possible D exponent instead of E 
                        try:
                            return dtype(line[start:end].replace('D','E'))
                        except ValueError: # this is a special case and it should not be in the main version tree!
                            # Dealing with the weird and unparsable intensity format such as "2.700-164, i.e with no E or D characters.
                            res = re.search('(\d\.\d\d\d)\-(\d\d\d)',line[start:end])
                            if res:
                                return dtype(res.group(1)+'E-'+res.group(2))
                            else:
                                raise Exception('PARSE ERROR: unknown format of the par value (%s)'%line[start:end])
                elif dtype==int and qnt=='local_iso_id':
                    if line[start:end]=='0': return 10
                    try:
                        return dtype(line[start:end])
                    except ValueError:
                        # convert letters to numbers: A->11, B->12, etc... ; .par file must be in ASCII or Unicode.
                        return 11+ord(line[start:end])-ord('A')
                else:
                    return dtype(line[start:end])
            #cfunc.__doc__ = 'converter {} {}'.format(qnt, fmt) # doesn't work in earlier versions of Python
            converters.append(cfunc)
            #start = end
        #data_matrix = [[cvt(line) for cvt in converters] for line in InfileData]
        flag_EOF = False
        line_count = 0
        data_matrix = []
        while True:
            if nlines is not None and line_count>=nlines: break   
            line = InfileData.readline()
            if line=='': # end of file is represented by an empty string
                flag_EOF = True
                break 
            data_matrix.append([cvt(line) for cvt in converters])
            line_count += 1
        data_columns = zip(*data_matrix)
        for qnt, col in zip(quantities, data_columns):
            #hapi.LOCAL_TABLE_CACHE[TableName]['data'][qnt].extend(col) # old code
            if type(col[0]) in {int,float}:
                hapi.LOCAL_TABLE_CACHE[TableName]['data'][qnt] = np.array(col) # new code
            else:
                hapi.LOCAL_TABLE_CACHE[TableName]['data'][qnt].extend(col) # old code
            #hapi.LOCAL_TABLE_CACHE[TableName]['data'][qnt] = list(col)
            #hapi.LOCAL_TABLE_CACHE[TableName]['data'][qnt] = col
        header['number_of_rows'] = line_count = (
            len(hapi.LOCAL_TABLE_CACHE[TableName]['data'][quantities[0]]))
                        
    # Convert all columns to numpy arrays
    par_names = hapi.LOCAL_TABLE_CACHE[TableName]['header']['order'].copy()
    if 'extra' in header and header['extra']:
        par_names += hapi.LOCAL_TABLE_CACHE[TableName]['header']['extra']
    for par_name in par_names:
        column = hapi.LOCAL_TABLE_CACHE[TableName]['data'][par_name]
        hapi.LOCAL_TABLE_CACHE[TableName]['data'][par_name] = np.array(column)                    
            
    # Additionally: convert numeric arrays of the hapi.LOCAL_TABLE_CACHE to masked arrays.
    # This is done to avoid "nan" values in the arithmetic ope  rations involving these columns.
    for par_name in hapi.LOCAL_TABLE_CACHE[TableName]['header']['order']:
        par_format = hapi.LOCAL_TABLE_CACHE[TableName]['header']['format'][par_name]
        regex = hapi.FORMAT_PYTHON_REGEX
        (lng,trail,lngpnt,ty) = re.search(regex,par_format).groups()
        if ty.lower() in ['d','e','f']:
            column = hapi.LOCAL_TABLE_CACHE[TableName]['data'][par_name]
            colmask = np.isnan(column)
            hapi.LOCAL_TABLE_CACHE[TableName]['data'][par_name] = np.ma.array(column,mask=colmask)
    
    if 'extra' in header and header['extra']:
        for par_name in hapi.LOCAL_TABLE_CACHE[TableName]['header']['extra']:
            par_format = hapi.LOCAL_TABLE_CACHE[TableName]['header']['extra_format'][par_name]
            regex = hapi.FORMAT_PYTHON_REGEX
            (lng,trail,lngpnt,ty) = re.search(regex,par_format).groups()
            if ty.lower() in ['d','e','f']:
                column = hapi.LOCAL_TABLE_CACHE[TableName]['data'][par_name]
                colmask = np.isnan(column)
                hapi.LOCAL_TABLE_CACHE[TableName]['data'][par_name] = np.ma.array(column,mask=colmask)
    
    # Delete all character-separated values, treat them as column-fixed.
    try:
        del hapi.LOCAL_TABLE_CACHE[TableName]['header']['extra']
        del hapi.LOCAL_TABLE_CACHE[TableName]['header']['extra_format']
        del hapi.LOCAL_TABLE_CACHE[TableName]['header']['extra_separator']
    except:
        pass
    # Update header.order/format with header.extra/format if exist.
    hapi.LOCAL_TABLE_CACHE[TableName]['header']['order'] = glob_order
    hapi.LOCAL_TABLE_CACHE[TableName]['header']['format'] = glob_format
    hapi.LOCAL_TABLE_CACHE[TableName]['header']['default'] = glob_default
    if flag_EOF:
        InfileData.close()
        hapi.LOCAL_TABLE_CACHE[TableName]['filehandler'] = None
    InfileHeader.close()
    print('                     Lines parsed: %d' % line_count)
    return flag_EOF