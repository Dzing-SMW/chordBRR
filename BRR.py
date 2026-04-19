#   ChordBRR v1.04 - A program for generating chord samples for the SPC700
#   Copyright (C) 2026  Dzing

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.

#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import struct

def overflow_check(v):
    v = int(v)  # Make the value an integer
    if v < 0:   # Change the value to unsigned
        v += 0x10000
    v = v & 0xFFFF  # Remove the overflow
    
    return v-0x10000 if v >= 0x8000 else v # Change back to signed integer

def apply_filter(b_filter,nib1, nib2):
    if b_filter == 1:
        return nib1 - (nib1 >> 4)
    elif b_filter == 2:
        p = nib1 << 1
        p += (-(nib1 + (nib1 << 1))) >> 5
        p -= nib2
        p += nib2 >> 4
        return p
    elif b_filter == 3:
        p = nib1 << 1
        p += (-(nib1 + (nib1 << 2) + (nib1 << 3))) >> 6
        p -= nib2
        p += (nib2 + (nib2 << 1)) >> 4
        return p
    else:
        return 0

def get_loop_point(fileContent):
    return struct.unpack('H', fileContent[:2])[0] / 9 * 16

def get_nibble_data(fileContent):
    if (len(fileContent) - 2) % 9 > 0:
        return -1
    data_bytes = struct.unpack('BBBBBBBBB' * ((len(fileContent)-2) // 9), fileContent[2:])
    
    nibbles = [0] * (len(data_bytes) // 9 *16)
    j = 0
    
    for i in range(0, len(data_bytes), 9):
        header = data_bytes[i]        # Header byte
        b_end = header & 1            # END bit is bit 0
        b_loop = header & 2           # LOOP bit is bit 1
        b_filter = (header >> 2) & 3  # FILTER is bits 2 and 3
        b_range = header >> 4         # RANGE is the upper 4 bits
        
        for tmp in data_bytes[i+1:i+9]:
            nib = tmp >> 4 # Get first nibble
            nib &= 0xF
            if nib >= 8:   # Nibble is negative
                nib -= 16
            nibbles[j] = overflow_check((nib << b_range) + apply_filter(b_filter, nibbles[j-1], nibbles[j-2]))
            j += 1
            nib = tmp & 0xF # Get the second nibble
            if nib >= 8:   # Nibble is negative
                nib -= 16
            nibbles[j] = overflow_check((nib << b_range) + apply_filter(b_filter, nibbles[j-1], nibbles[j-2]))
            j += 1
   
    return nibbles

def calc_range(b_filter, nib1, nib2, nibbles):

    unib = [0.0] * 16

    for i in range(16): # Create unfiltered values
        unib[i] = calc_unfilteredvalue(b_filter, nib1, nib2, nibbles[i])
        nib2 = nib1
        nib1 = nibbles[i]
    
    r = 0
    while max(unib) > 7.5 or min(unib) < -8.5:
        unib = np.divide(unib, 2)
        r += 1
    return r, nibbles

def calc_unfilteredvalue(b_filter, nib1, nib2, nibble):
    return nibble - apply_filter(b_filter, int(nib1), int(nib2))

def calc_filteredvalue(b_filter, nib1, nib2, nibble):
    return nibble + apply_filter(b_filter, int(nib1), int(nib2))

def calc_blockvalues(r, f, nib1, nib2, nibbles):
    n1 = nib1
    n2 = nib2
    
    nnib = [0.0] * 16
    nerror = 0
    
    for i in range(16):
        n = int(round(calc_unfilteredvalue(f, n1, n2, nibbles[i]) / 2**r))
        nv = calc_filteredvalue(f, n1, n2, n << r) # Check if overflow has been done properly
        if nv > 32767 :
            n -= 1
            nv = calc_filteredvalue(f, n1, n2, n << r)
        if nv < -32768:
            n += 1
            nv = calc_filteredvalue(f, n1, n2, n << r)
        
        if n < -8 or n > 7: # Make sure that values don't exceed valid values (as the range value is calculated approximately)
            r += 1 # Increase range and try again
            return calc_blockvalues(r, f, nib1, nib2, nibbles)
            
        nerror += (nibbles[i] - nv) ** 2
        if n < 0:
            n += 16
        nnib[i] = int(n)
        n2 = n1
        n1 = int(nv)
    
    return nnib, nerror, n1, n2, r

    
def calc_block(nib1, nib2, nibbles, looped, filters):
    
    nnibb = [0.0] * 16
    nerrorb = 0x7FFFFFFFFFFFFFFF
    fb = 0
    rb = 0
    n1b = 0
    n2b = 0
    
    for f in range(4):
        if filters & (1 << f) == (1 << f):
            r, nb = calc_range(f, nib1, nib2, nibbles)
                           
            nnib, nerror, n1, n2, r = calc_blockvalues(r, f, nib1, nib2, nb)
            
            if nerror < nerrorb:
                nerrorb = nerror
                nnibb = nnib[:]
                fb = f
                rb = r
                n1b = n1
                n2b = n2
                
    
    ndata = [0] * 9 # Make BRR block from data
    ndata[0] = (looped << 1) + (fb << 2) + (rb << 4)
    for i in range(8):
        ndata[i+1] = (nnibb[i*2] << 4) + nnibb[i*2+1]
    
    return n1b, n2b, ndata, fb

def checkBRRloop(file_name, loop_point):
    
    with open(file_name, mode = 'rb') as file:
        fileContent = file.read()
    
    data_bytes = struct.unpack('BBBBBBBBB' * ((len(fileContent)-2) // 9), fileContent[2:])
    
    n_looppoint = loop_point // 16 * 9
    loop_size = len(data_bytes) - n_looppoint
    
    nibbles = [0] * int((n_looppoint + loop_size * 2) // 9 *16)
    j = 0
    
    
    for k in range(0, n_looppoint + loop_size * 2, 9):
        if k > n_looppoint:
            i = int((k - n_looppoint) % loop_size + n_looppoint)
        else:
            i = k
        
        header = data_bytes[i]        # Header byte
        b_end = header & 1            # END bit is bit 0
        b_loop = header & 2           # LOOP bit is bit 1
        b_filter = (header >> 2) & 3  # FILTER is bits 2 and 3
        b_range = header >> 4         # RANGE is the upper 4 bits
        
        for tmp in data_bytes[i+1:i+9]:
            nib = tmp >> 4 # Get first nibble
            nib &= 0xF
            if nib >= 8:   # Nibble is negative
                nib -= 16
            nibbles[j] = overflow_check((nib << b_range) + apply_filter(b_filter, nibbles[j-1], nibbles[j-2]))
            j += 1
            nib = tmp & 0xF # Get the second nibble
            if nib >= 8:   # Nibble is negative
                nib -= 16
            nibbles[j] = overflow_check((nib << b_range) + apply_filter(b_filter, nibbles[j-1], nibbles[j-2]))
            j += 1
            
    loop_point2 = (loop_size + n_looppoint) // 9 * 16
    
    l1 = np.array(nibbles[loop_point:loop_point+16])
    l2 = np.array(nibbles[loop_point2:loop_point2+16])
    
    return abs(l1 - l2).mean() > 1000

def saveBRR(loop_point, data, file_name,force_filteratloop=False):
    
    nib1 = 0
    nib2 = 0
    
    if loop_point > -1:
        l = 1
        lz = (16 - loop_point % 16) & 15
        data = np.append(np.zeros(lz), data)
        loop_point += lz
    else:
        l = 0
        data = np.append(np.zeros( (16 - len(data) % 16) & 15 ), data)
        loop_point = 0
    
    with open(file_name, mode = 'wb') as file:
        file.seek(0)
        lp = int(loop_point / 16 * 9)
        file.write(bytearray([lp & 0xFF, lp >> 8]))
        
        for i in range(0, len(data) - 16, 16):
            n1 = nib1
            n2 = nib2
            if i == 0 or (i == loop_point) and force_filteratloop:
                nib1, nib2, ndata, filter = calc_block(nib1, nib2, data[i:i+16], l, 1)
            else:
                nib1, nib2, ndata, filter = calc_block(nib1, nib2, data[i:i+16], l, 15)
            file.write(bytearray(ndata))
            
        file.truncate()
        nib1, nib2, ndata, filter = calc_block(nib1, nib2, data[-16:], l, 15)
        ndata[0] = ndata[0] + 1
        file.write(bytearray(ndata))
        
    if checkBRRloop(file_name, loop_point):
            saveBRR(loop_point, data, file_name, True)
        