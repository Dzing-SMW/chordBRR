import dearpygui.dearpygui as dpg
import tkinter as tk
from tkinter import filedialog
import os
import math
import struct
import numpy as np
from scipy import interpolate
from scipy.io import wavfile
from scipy import signal
import sounddevice as sd

# ChordBRR
# Version 0.90b
# by Dzing


# Initialize global variables
_octaves =  ["o1","o2","o3","o4","o5","o6","o7"]
_notes = ["c","c+","d","d+","e","f","f+","g","g+","a","a+","b"]
num_notes = 2
note_rows = [0] * 5
sel_octaves = [3] * 5
sel_notes = [0] * 5

_graphs = [0] * 5
graph_lines = [0] * 5

window_state = 0

loop_point = 0
loop_len = 0
nloop_point = 0
nibbles = []
n_loops = []

listbox_items = []
_volumes = [100] * 5
_delays = [0] * 5

n_precision = 10
scale_factor = 1

# Opens a BRR file
def open_BRR_file():
    global nibbles, nloop_point, loop_point, loop_len
    file_path = filedialog.askopenfilename(title = "Open BRR file...", filetypes = [("BRR files","*.brr")])
    if os.path.isfile(file_path):
        with open(file_path, mode = 'rb') as file:
            fileContent = file.read()
        
        loop_point = int(get_loop_point(fileContent))
        
        nibbles = get_nibble_data(fileContent)
        
        if sum(nibbles[:15]) == 0:  #Remove the first block if it is 0
            nibbles = nibbles[16:]
            loop_point -= 16
        
        loop_len = len(nibbles) - loop_point
        x_data = list(range(len(nibbles)))
        
        #Get tuning values from !patterns.txt file
        p_root, f_name = os.path.split(file_path)
        pattern_file = os.path.join(p_root, '!patterns.txt')

        if os.path.isfile(pattern_file):
            with open(pattern_file) as file:
                for line in file:
                    if line.find(f_name) >= 0:
                        v = line.split('$')
                        dpg.set_value("tuningh", v[4].strip())
                        dpg.set_value("tuningl", v[5].strip())
        
        dpg.set_value("text_filename", os.path.basename(file_path))
        dpg.set_value("text_looppoint", int(loop_point))
        dpg.set_value("text_size", len(nibbles))
        
        dpg.set_value("BRRplot", [x_data, nibbles])
        dpg.set_value("BRRlooppoint", [ [loop_point, loop_point], [-1, 1]] )
        dpg.set_axis_limits("x_axis", 0, len(nibbles))
    
        dpg.configure_item("button_next", enabled=True)
        
        #Upsample the data with polyphase interpolation, add 16 bits of data at the start and end of the loop to determine the boundary conditions for the interpolation
        nloop_point = loop_point * n_precision
        st = signal.resample_poly(nibbles[:loop_point+16],20*n_precision,20)[:nloop_point]
        nl = np.array(nibbles[loop_point:])
        nl = np.append(np.append(nl[-16:],nl),nl[:16])
        nibbles = np.append(st, signal.resample_poly(nl,20*n_precision,20)[16*n_precision:-16*n_precision])

def overflow_check(v):
    v = int(v)  # Make the value an integer
    if v < 0:   # Change the value to unsigned
        v += 0x10000
    v = v & 0xFFFF  # Remove the overflow
    
    return v-0x10000 if v >= 0x8000 else v # Change back to signed integer

def apply_filter(b_filter,nib1, nib2):
    if b_filter == 1:
        return overflow_check(float(nib1 * 15/16))
    elif b_filter == 2:
        return overflow_check(float(nib1 * 61/32) - float(nib2 * 15/16))
    elif b_filter == 3:
        return overflow_check(float(nib1 * 115/64) - float(nib2 * 13/16))
    else:
        return 0

def get_loop_point(fileContent):
    return struct.unpack('H', fileContent[:2])[0] / 9 * 16

def get_nibble_data(fileContent):
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
    
    nib_max = float(max(abs(x) for x in nibbles))
    norm_nib = [0.0] * len(nibbles)
    i = 0
    for x in nibbles:
        norm_nib[i] = float(x) / nib_max
        i += 1
   
    return norm_nib



def button_next():
    global window_state
    if window_state == 0:
        dpg.move_item("BRRLoad", parent="stage1")
        dpg.move_item("editnotes", parent="disprow")
        window_state = 1
        dpg.configure_item("button_back", enabled=True)
    else:
        dpg.move_item("editvolume", parent="disprow")
        dpg.move_item("editnotes", parent="stage1")
        window_state = 2
        dpg.configure_item("button_next", enabled=False)
        change_tuning()
        generate_graphs()

def button_back():
    global window_state
    if window_state == 1:
        dpg.move_item("BRRLoad", parent="disprow")
        dpg.move_item("editnotes", parent="stage1")
        window_state = 0
        dpg.configure_item("button_back", enabled=False)
    else:
        dpg.move_item("editnotes", parent="disprow")
        dpg.move_item("editvolume", parent="stage1")
        dpg.configure_item("button_next", enabled=True)
        window_state = 1

def note_number_change(sender, app_data):
    global num_notes
    num_notes = int(app_data)
    for r in note_rows:
        dpg.delete_item(r)
    for i in range(5):
        if i < num_notes:
            with dpg.group(parent="notegroup", horizontal=True):
                note_rows[i] = dpg.last_item()
                dpg.add_combo(_octaves, default_value=_octaves[sel_octaves[i]], tag="oct" + str(i), callback=oct_change, width=80)
                dpg.add_combo(_notes, default_value=_notes[sel_notes[i]], tag="note" + str(i), callback=note_change, width=80)
        else:
            note_rows[i] = 0
    calc_matches()

def oct_change(sender, app_data):
    sel_octaves[int(sender[-1:])] = _octaves.index(app_data)
    calc_matches()
        
def note_change(sender, app_data):
    sel_notes[int(sender[-1:])] = _notes.index(app_data)
    calc_matches()

def calc_matches():
    global listbox_items
    loop_error, n_loops = get_matches(sel_octaves, sel_notes, num_notes, dpg.get_value("Threshold") )
    s = [""] * len(n_loops)
    for i in range(len(n_loops)):
        s[i] = "#" + str(i+1) + " - " + str(n_loops[i][0]) + " loops - " + str(round(max(loop_error[i]), 2)) + " error"
    listbox_items = s
    dpg.configure_item(listbox1, items=s)

def generate_graphs():
    global _graphs, graph_lines
    for r in _graphs:
        dpg.delete_item(r)
    s = [0] * num_notes
    for i in range(num_notes):    
        s[i] = _octaves[sel_octaves[i]] + " " + _notes[sel_notes[i]]
    s.sort()
    for i in range(5):
        if i < num_notes:
            with dpg.group(parent="editvolume", horizontal=True):
                _graphs[i] = dpg.last_item()
                with dpg.plot(width = 600, height=60):
                    x_axis = dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels = True, tag = "ax" + str(i))
                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, no_tick_labels = True, label=s[i])
                    dpg.set_axis_limits(y_axis, -1, 1)
                    graph_lines[i] = dpg.add_line_series([],[], parent=y_axis)
                dpg.add_slider_int(min_value=1, max_value=100, default_value=_volumes[i], vertical = True, height=60, callback=volume_change)
                dpg.add_input_int(default_value=_delays[i], width=80, step=100, callback=delay_change)
        else:
            _graphs[i] = 0
    nnibbles, x, h_l = calculate_wavesequence(True)
    update_graphs(x, nnibbles)

def volume_change(sender, app_data):
    global _volumes
    i = _graphs.index(sender - 5)
    x,n = dpg.get_value(graph_lines[i])
    for j in range(len(n)):
        n[j] = n[j]*app_data/_volumes[i]
    _volumes[i] = int(app_data)
    dpg.set_value(graph_lines[i], [x,n])

def delay_change(sender, app_data):
    global _delays
    if app_data < 0:
        app_data = 0
        dpg.set_value(sender, 0)
    _delays[_graphs.index(sender - 6)] = int(app_data)
    nnibbles, x, h_l = calculate_wavesequence(True)
    update_graphs(x, nnibbles)

        
def calculate_wavesequence(reduced):
    
    loop_error, n_loops = get_matches(sel_octaves, sel_notes, num_notes, dpg.get_value("Threshold") )
    
    sel_match = listbox_items.index(dpg.get_value(listbox1))
    
    t = [0] * num_notes
    n_wl = [0] * num_notes
    n = [0] * num_notes
    
    for i in range(num_notes):
        n[i] = loop_len / n_loops[sel_match][i]
    
    for i in range(num_notes):
        n_wl[i] = n[i]/n[0]
        t[i] = loop_point * n_wl[i] + _delays[i]

    h_l = max(t)
    l = h_l + loop_len * n_loops[sel_match][0]
    
    dpg.set_value("text_newsize", int(math.ceil(l/16*scale_factor) * 9 +2))
    
    if reduced:
        stp = l/2000
    else:
        stp = 1/scale_factor
    
    new_x_val = np.arange(0, l + 1, stp, dtype=float)
    n_nibbles = [0] * 5
    
    for i in range(num_notes):
        l = math.ceil(((h_l + loop_len * 2 - _delays[i] - loop_point * n_wl[i]) / n[i]))
        nnibbles = np.array(nibbles[:nloop_point])
        for j in range(l):
            nnibbles = np.append(nnibbles, nibbles[nloop_point:])
        
        x_val = np.arange(0, nnibbles.size * n_wl[i], n_wl[i]/n_precision, dtype=float)
        if x_val.size > nnibbles.size:
            x_val = x_val[:nnibbles.size]
        
        new_y = interpolate.interp1d(x_val, nnibbles)(new_x_val[:-int(_delays[i]/stp)-1])
        n_nibbles[i] = np.append(np.zeros(int(_delays[i]/stp)), new_y) * _volumes[i] / 100

    return (n_nibbles, new_x_val, int(h_l*scale_factor))

def calc_wl_error(n_wl, s_wl, nloops):
    c = [0] * len(n_wl)
    for i in range(len(n_wl)):
        f1 = 1 / n_wl[i]
        f2 = 1/( s_wl[0] / nloops[i])
        c[i] = 1200 * math.log2(f1/f2)
    return c

def get_note_wavelength(sel_octaves,sel_notes,num_notes):
    note_wl = [0] * num_notes
    for i in range(num_notes):
        n = float(sel_octaves[i]*12 + sel_notes[i] - 9)
        note_wl[i] = 1.0 / (55.0 * 2.0 ** (n / 12.0))

    note_wl.sort(reverse=True)
    return note_wl

def get_JI_list(sel_octaves,sel_notes,num_notes):
    note_nr = [0] * num_notes
    for i in range(num_notes):
        note_nr[i] = sel_octaves[i]*12 + sel_notes[i]
    note_nr.sort(reverse=True)
    
    JI = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0]
    JI_list = [0.0] * num_notes
    
    for i in range(num_notes):
        JI_list[i] = JI[(note_nr[i]-note_nr[0]) % 12]
    
    return JI_list

def change_tuning_dialog(sender, app_data):
    #Tuning change dialog here
    dpg.configure_item("modal_id", show=False)
    change_tuning()

def change_tuning():
    global scale_factor
    loop_error, n_loops = get_matches(sel_octaves, sel_notes, num_notes, dpg.get_value("Threshold") )
    t = dpg.get_value("tuning_type")
    if t == "Tune for quality":
        l = n_loops[listbox_items.index(dpg.get_value(listbox1))][0]
        ll = n_loops[listbox_items.index(dpg.get_value(listbox1))][-1]
        #Get original tuning
        th=int(dpg.get_value("tuningh"),16)
        tl=int(dpg.get_value("tuningl"),16)
        o_t = (th * 256 + tl)
        #Get new tuning
        nwl = max(get_note_wavelength(sel_octaves,sel_notes,num_notes))
        n_t1 = round(o_t * ll/l) # Tuning for retaining tuning for the upper note in the chord
        n_t2 = round(1960 * nwl * 261.63) # Tuning for playing the sample at 32khz
        if n_t1 < n_t2:
            n_t = n_t1
        else:
            n_t = n_t2
        #Get number of blocks in the loop
        blks = loop_len/16
        #Round tuning towards nearest fitting tuning in order for the loop to become a division of 16
        n_t = round(round(n_t * l * blks / o_t) * o_t / l / blks)
        th = math.floor(n_t/256)
        tl = n_t % 256
        dpg.set_value("outputtuning","$" + "{:02X}".format(int(th)) + "$" + "{:02X}".format(int(tl)))
        scale_factor = n_t/o_t
    elif t == "Custom tuning":
        l = n_loops[listbox_items.index(dpg.get_value(listbox1))][0]
        #Get original tuning
        th=int(dpg.get_value("tuningh"),16)
        tl=int(dpg.get_value("tuningl"),16)
        o_t = (th * 256 + tl)
        #Get new tuning
        th=int(dpg.get_value("otuningh"),16)
        tl=int(dpg.get_value("otuningl"),16)
        n_t = float(th * 256 + tl)
        #Get number of blocks in the loop
        blks = loop_len/16
        #Round tuning towards nearest fitting tuning in order for the loop to become a division of 16
        n_t = round(round(n_t * l * blks / o_t) * o_t / l / blks)
        th = math.floor(n_t/256)
        tl = n_t % 256
        dpg.set_value("outputtuning","$" + "{:02X}".format(int(th)) + "$" + "{:02X}".format(int(tl)))
        scale_factor = n_t/o_t
    else:
        dpg.set_value("outputtuning","$" + dpg.get_value("tuningh") + "$" + dpg.get_value("tuningl"))
        scale_factor = 1
    
    # Update estimated data size
    sel_match = listbox_items.index(dpg.get_value(listbox1))

    t = [0] * num_notes
    n_wl = [0] * num_notes
    n = [0] * num_notes
    
    for i in range(num_notes):
        n[i] = loop_len / n_loops[sel_match][i]
    
    for i in range(num_notes):
        n_wl[i] = n[i]/n[0]
        t[i] = loop_point * n_wl[i] + _delays[i]

    h_l = max(t)
    l = h_l + loop_len * n_loops[sel_match][0]
    
    dpg.set_value("text_newsize", int(math.ceil(l/16*scale_factor) * 9 +2))
    

def get_matches(sel_octaves,sel_notes,num_notes,error_threshold):

    note_wl = get_note_wavelength(sel_octaves,sel_notes,num_notes)
    n = note_wl[0]

    for i in range(num_notes):
        note_wl[i] = note_wl[i]/n
        
    w = note_wl.copy()
    l = [1] * num_notes
    i = 0
    j = 0
    loop_error = [[0]] * 10
    n_loops = [[0]] * 10
    
    
    while (j < 10) & (i < 2000):
        c = w.index(min(w))
        w[c] += note_wl[c]
        l[c] += 1
        er = calc_wl_error(note_wl, w, l)
        if dpg.get_value("JImode"):
            JI_list = get_JI_list(sel_octaves,sel_notes,num_notes)
            for k in range(num_notes):
                if (er[k]*JI_list[k]) < 0:
                    er[k] = 50.0
                    
        for k in range(num_notes):
            er[k] = abs(er[k])
        
        if max(er) < error_threshold:
            loop_error[j] = er
            n_loops[j] = l.copy()
            j += 1
        i += 1

    return (loop_error, n_loops)



# Saves a wav file with loop point
def save_wav():
    file_path = filedialog.asksaveasfilename(title = "Save as...", filetypes = [("wav files","*.wav")], confirmoverwrite=True, defaultextension=".wav")
    if os.path.isdir(os.path.dirname(file_path)):

        n_nibbles, new_x_val, h_l = calculate_wavesequence(False)
        
        data = n_nibbles[0]
        for i in range(1, num_notes):
            data = np.add(data, n_nibbles[i])
        
        nwl = max(get_note_wavelength(sel_octaves,sel_notes,num_notes))
        th=int(dpg.get_value("outputtuning")[1:3],16)
        tl=int(dpg.get_value("outputtuning")[4:6],16)
        
        fs = int((th * 256 + tl)/(16*nwl))
        
        data = data / np.max(np.absolute(data)) * np.iinfo(np.int16).max
        data = np.append(np.zeros(int(math.ceil(h_l/16)*16 - h_l)), data)
        wavfile.write(file_path, fs, data.astype(np.int16))
        
        # Add the loop point to the file
        fout = open(file_path, 'ab')
        
        fout.write(struct.pack("<lllllllllll", 1819307379, 60, 0, 0, int(1000000000/fs), 60, 0, 0, 0, 1, 0))
        fout.write(struct.pack("<llllll", 0, 0, int(math.ceil(h_l/16)*16), data.size - 1, 0, 0))
        
        fout.close()
        
        # Update the chunk size of the main RIFF
        
        fout = open(file_path, 'r+b')
        fout.seek(4)
        l = struct.unpack("<l", fout.read(4))[0]
        fout.seek(4)
        fout.write(struct.pack("<l", l + 60 ))
        fout.close()

        
# Update the note graphs in page 3
def update_graphs(x, nnibbles):
    for i in range(num_notes):
        dpg.set_value(graph_lines[i], [x, nnibbles[i]])
        dpg.set_axis_limits("ax"+str(i), 0, x[x.size-1])
        
# Play a test sound
def play_sound():
    n_nibbles, new_x_val, h_l = calculate_wavesequence(False)
    data = n_nibbles[0]
    for i in range(1, num_notes):
        data = np.add(data, n_nibbles[i])
    
    data = data / np.max(np.absolute(data)) * 0.5 
    
    lp = data[int(h_l):]
    
    nwl = max(get_note_wavelength(sel_octaves,sel_notes,num_notes))
    th=int(dpg.get_value("outputtuning")[1:3],16)
    tl=int(dpg.get_value("outputtuning")[4:6],16)
    fs = int((th * 256 + tl)/(16*nwl))
    
    
    for i in range(int(2 * fs/(data.size - h_l))):
        data = np.append(data, lp)
    
    if data.size > fs*2:
        data = data[:fs*2]
    
    sd.play(data, fs)
        


dpg.create_context()

with dpg.theme() as disabled_theme:             # Generate a disabled theme for the buttons
    with dpg.theme_component(dpg.mvButton, enabled_state=False):
        dpg.add_theme_color(dpg.mvThemeCol_Text, [192, 192, 192])
        dpg.add_theme_color(dpg.mvThemeCol_Button, [51, 51, 55])
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [51, 51, 55])
        
        
dpg.bind_theme(disabled_theme)


with dpg.stage(tag="stage1"):                   # Generate GUI pages for the note data input and the volume + wav generation

# GUI for note input from users

    with dpg.table(resizable=True, policy=dpg.mvTable_SizingStretchProp, header_row=False, borders_outerH=False, borders_innerV=True, borders_innerH=True, borders_outerV=False, tag="editnotes"):
        dpg.add_table_column()
        dpg.add_table_column()
        with dpg.table_row():
            with dpg.table_cell():
                dpg.add_text("Number of notes")
                dpg.add_combo((2,3,4,5), default_value=2, callback=note_number_change)
                dpg.add_text("")
                dpg.add_group(tag="notegroup")
                    
            with dpg.table_cell():
                dpg.add_text("Error threshold (cent)")
                dpg.add_slider_int(min_value=1, max_value=20, default_value=5, tag="Threshold", callback=calc_matches)
                dpg.add_text("")
                dpg.add_checkbox(label="Just intonation mode", tag="JImode", callback=calc_matches)
                dpg.add_text("")
                listbox1 = dpg.add_listbox([], num_items=10)
        
    note_number_change(0, 2)
    
# GUI for volume input from users    
    
    with dpg.group(tag="editvolume"):
        dpg.add_text("")
        dpg.add_text("Vol", pos=[616,52])
        dpg.add_text("Delay", pos=[650,52])
        with dpg.group(horizontal=True, pos=[8,410]):
            dpg.add_button(label = "Play", callback=play_sound)
            dpg.add_button(label = "Save file", callback=save_wav)
            dpg.add_text("       Output tuning:")
            dpg.add_text("$04$00", tag="outputtuning", color=(255, 0, 255))
            dpg.add_button(label = "Change...")
            with dpg.popup(dpg.last_item(), mousebutton=dpg.mvMouseButton_Left, modal=True, tag="modal_id"):
                dpg.configure_item("modal_id", label="Change tuning...")
                dpg.add_radio_button(("Keep original tuning", "Tune for quality", "Custom tuning"),tag="tuning_type")
                with dpg.group(horizontal=True):
                    dpg.add_input_text(default_value="04", tag="otuningh", hexadecimal=True, width=30)
                    dpg.add_input_text(default_value="00", tag="otuningl", hexadecimal=True, width=30)
                dpg.add_text("")
                dpg.add_button(label="Ok", width=80, callback=change_tuning_dialog)
            dpg.add_text("       ")
            dpg.add_text("Estimated BRR size:")
            dpg.add_text("0", tag="text_newsize", color=(255, 0, 255))

with dpg.window(tag="BRR_data_window"):
    
# Header GUI
    
    with dpg.table(header_row=False, borders_outerH=False, borders_innerV=False, borders_innerH=False, borders_outerV=False):
        dpg.add_table_column()
        with dpg.table_row():
            with dpg.table_cell():
                with dpg.group(horizontal=True):
                    dpg.add_text("Filename: ")
                    dpg.add_text("", tag="text_filename", color=(255, 0, 255))
                    dpg.add_text("Size: ")
                    dpg.add_text("", tag="text_size", color=(255, 0, 255))
                    dpg.add_text("Loop point: ")
                    dpg.add_text("", tag="text_looppoint", color=(255, 0, 255))
                with dpg.group(horizontal=True):
                    dpg.add_text("Tuning")
                    dpg.add_input_text(default_value="04", tag="tuningh", hexadecimal=True, width=30, callback=change_tuning)
                    dpg.add_input_text(default_value="00", tag="tuningl", hexadecimal=True, width=30, callback=change_tuning)
        with dpg.table_row(tag="disprow", height=390):

# GUI for loading BRR files

            with dpg.group(horizontal=False, tag="BRRLoad"):
                dpg.add_button(label="Open BRR file", callback=open_BRR_file)
                with dpg.plot(width = -1, height=360,tag="plotwindow"):
                    dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis")
                    dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis")
                    dpg.set_axis_limits("y_axis", -1, 1)
                    dpg.add_line_series([],[], tag = "BRRplot", parent="y_axis")
                    dpg.add_line_series([],[], tag = "BRRlooppoint", parent="y_axis")

# Footer GUI (back and next buttons)

        with dpg.table_row():
            with dpg.group(horizontal=True):
                dpg.add_button(label = "back", enabled=False, tag="button_back", callback=button_back)
                dpg.add_button(label = "next", enabled=False, tag="button_next", callback=button_next)
                




dpg.create_viewport(title='ChordBRR', width=760, height=520, resizable=False)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("BRR_data_window", True)
dpg.start_dearpygui()
dpg.destroy_context()
