#   ChordBRR v1.03 - A program for generating chord samples for the SPC700
#   Copyright (C) 2025  Dzing

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

import dearpygui.dearpygui as dpg
import tkinter as tk
from tkinter import filedialog
import os
import math
import struct
import numpy as np
from scipy.io import wavfile
from scipy import signal
import sounddevice as sd
import BRR


# Initialize global variables
_octaves =  ["o1","o2","o3","o4","o5","o6","o7"]
_notes = ["c","c+","d","d+","e","f","f+","g","g+","a","a+","b"]
num_notes = 2
note_rows = [0] * 5
sel_octaves = [3] * 5
sel_notes = [0] * 5
sel_octaves_u = [3] * 5
sel_notes_u = [0] * 5

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
        try:
            with open(file_path, mode = 'rb') as file:
                fileContent = file.read()
        except OSError as e:
            tk.messagebox.showwarning('Error opening BRR file', "I/O error({0}): {1}".format(e.errno, e.strerror))
            dpg.configure_item("button_next", enabled=False)
            return
        loop_point = int(BRR.get_loop_point(fileContent))
        
        nibbles = BRR.get_nibble_data(fileContent)
        
        if nibbles == -1:
            tk.messagebox.showwarning('Error opening BRR file', 'Invalid file size')
            dpg.configure_item("button_next", enabled=False)
            return
        
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

def sort_notelist():
        global sel_octaves, sel_notes
        n = [0] * num_notes
        for i in range(num_notes):
            n[i] = sel_octaves_u[i] * 12 + sel_notes_u[i]
        ns = n[:]
        ns.sort()
        for i in range(num_notes):
            sel_octaves[i] = sel_octaves_u[n.index(ns[i])]
            sel_notes[i] = sel_notes_u[n.index(ns[i])]

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
                dpg.add_combo(_octaves, default_value=_octaves[sel_octaves_u[i]], tag="oct" + str(i), callback=oct_change, width=80)
                dpg.add_combo(_notes, default_value=_notes[sel_notes_u[i]], tag="note" + str(i), callback=note_change, width=80)
        else:
            note_rows[i] = 0
    sort_notelist()
    calc_matches()

def oct_change(sender, app_data):
    sel_octaves_u[int(sender[-1:])] = _octaves.index(app_data)
    sort_notelist()
    calc_matches()
        
def note_change(sender, app_data):
    sel_notes_u[int(sender[-1:])] = _notes.index(app_data)
    sort_notelist()
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
        
    for i in range(5):
        if i < num_notes:
            s = _octaves[sel_octaves[i]] + " " + _notes[sel_notes[i]]
            with dpg.group(parent="editvolume", horizontal=True):
                _graphs[i] = dpg.last_item()
                dpg.add_input_text(default_value=s, enabled=False, width=50)
                dpg.add_slider_int(min_value=1, max_value=100, default_value=_volumes[i], vertical = False, height=60, width=300, callback=volume_change)
                dpg.add_input_int(default_value=_delays[i], width=80, step=100, callback=delay_change)
        else:
            _graphs[i] = 0

def volume_change(sender, app_data):
    global _volumes
    i = _graphs.index(sender - 2)
    _volumes[i] = int(app_data)

def delay_change(sender, app_data):
    global _delays
    if app_data < 0:
        app_data = 0
        dpg.set_value(sender, 0)
    _delays[_graphs.index(sender - 3)] = int(app_data)
    calculate_filesize()

def calculate_filesize():
    loop_error, n_loops = get_matches(sel_octaves, sel_notes, num_notes, dpg.get_value("Threshold") )
    
    sel_match = listbox_items.index(dpg.get_value(listbox1))
    
    t = [0] * num_notes # Length of section before the loop point
    n_wl = [0] * num_notes # Length of one loop
    n = [0] * num_notes
    d = [0] * num_notes # Delay length
    
    th=int(dpg.get_value("tuningh"),16)
    tl=int(dpg.get_value("tuningl"),16)
    s_f = (th * 16 + tl/16) * 55.0 * scale_factor # Sample frequency
    
    for i in range(num_notes):
        n[i] = loop_len / n_loops[sel_match][i]
        d[i] = int(_delays[i] / 1000 * s_f)
    
    for i in range(num_notes):
        n_wl[i] = n[i]/n[0]
        t[i] = int(math.ceil(loop_point * n_wl[i] * scale_factor)) + d[i]

    h_l = max(t) # Maximum length of section before the loop point
    l = h_l + int(round(loop_len * n_loops[sel_match][0] * scale_factor, 0))
    
    dpg.set_value("text_newsize", int(math.ceil(l/16*scale_factor) * 9 + 11))
        
def calculate_wavesequence():
    
    loop_error, n_loops = get_matches(sel_octaves, sel_notes, num_notes, dpg.get_value("Threshold") )
    
    sel_match = listbox_items.index(dpg.get_value(listbox1))
    
    t = [0] * num_notes # Length of section before the loop point
    n_wl = [0] * num_notes # Length of one loop
    n = [0] * num_notes
    d = [0] * num_notes # Delay length
    
    th=int(dpg.get_value("tuningh"),16)
    tl=int(dpg.get_value("tuningl"),16)
    s_f = (th * 16 + tl/16) * 55.0 * scale_factor # Sample frequency
    
    for i in range(num_notes):
        n[i] = loop_len / n_loops[sel_match][i]
        d[i] = int(_delays[i] / 1000 * s_f)
    
    for i in range(num_notes):
        n_wl[i] = n[i]/n[0]
        t[i] = int(math.ceil(loop_point * n_wl[i] * scale_factor)) + d[i]

    h_l = max(t) # Maximum length of section before the loop point
    l = h_l + int(round(loop_len * n_loops[sel_match][0] * scale_factor, 0))
    
    y_val_h = nibbles[:nloop_point + 1] # Generate arrays for Y values used for interpolation
    y_val = nibbles[nloop_point:]
    ll = len(y_val)
    data = np.zeros(l) # Generate empty array for sample data
    
    new_x_val = np.arange(0, l, 1, dtype=float) # Generate array for new X values
    
    for i in range(num_notes): # Interpolate and add numbers to data array
        for j in range(l-d[i]):
            x, r = divmod(j * n_precision / (n_wl[i] * scale_factor), 1)
            x = int(x)
            if x < nloop_point:
                data[j + d[i]] += (y_val_h[x] + (y_val_h[x+1] - y_val_h[x]) * r) * _volumes[i] / 100
            else:
                x = (x - nloop_point + 1) % ll
                data[j + d[i]] += (y_val[x - 1] + (y_val[x] - y_val[x - 1]) * r) * _volumes[i] / 100

    n = (16 - h_l % 16) & 15
    h_l += n
    data = np.append(np.zeros(n), data)

    return (data, h_l)

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
    calculate_filesize()
    

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
    file_path = filedialog.asksaveasfilename(title = "Save as...", filetypes = [("BRR file","*.brr"), ("wav file","*.wav")], confirmoverwrite=True, defaultextension=".brr")
    if os.path.isdir(os.path.dirname(file_path)):

        data, h_l = calculate_wavesequence()
        
        if file_path[-3:].casefold() == 'brr':
            data = data / np.max(np.absolute(data)) * np.iinfo(np.int16).max
            BRR.saveBRR(h_l, data, file_path)
        else:
            nwl = max(get_note_wavelength(sel_octaves,sel_notes,num_notes))
            th=int(dpg.get_value("outputtuning")[1:3],16)
            tl=int(dpg.get_value("outputtuning")[4:6],16)
            
            fs = int((th * 256 + tl)/(16*nwl))
            
            data = data / np.max(np.absolute(data)) * np.iinfo(np.int16).max
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
        
# Play a test sound
def play_sound():
    data, h_l = calculate_wavesequence()
    
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
        dpg.add_text("Volume (%)", pos=[66,58])
        dpg.add_text("Delay (ms)", pos=[376,58])
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