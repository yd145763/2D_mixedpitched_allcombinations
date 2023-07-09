# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 20:42:58 2023

@author: Lim Yudian
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 18:18:07 2023

@author: Lim Yudian
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import imp
from matplotlib.ticker import StrMethodFormatter


# Specify the path
path = 'C:\\Users\\Lim Yudian\\Documents\\focusinggrating2Dgds\\all possible combinations of mixed pitch'

# Get a list of all files in the directory
files = os.listdir(path)

# Filter the GDS files and extract their names without the extension
gds_files = [filename[:-4] for filename in files if filename.endswith('.gds')]


os.add_dll_directory("C:\\Program Files\\Lumerical\\v231\\api\\python\\lumapi.py") #the lumapi.py path in your pc, remember the double \\
lumapi = imp.load_source("lumapi","C:\\Program Files\\Lumerical\\v231\\api\\python\\lumapi.py") #the lumapi.py path in your pc, remember the double \\
    
fdtd = lumapi.FDTD(r"C:\Users\Lim Yudian\Documents\focusinggrating2Dgds\grating_coupler_2D.fsp")

for g in gds_files:
    fdtd.gdsimport("C:\\Users\\Lim Yudian\\Documents\\focusinggrating2Dgds\\all possible combinations of mixed pitch\\"+g+".GDS", g, 1, "Si3N4 (Silicon Nitride) - Phillip", 0, 0.4e-6)
    fdtd.set("name", g)
    fdtd.set("z span", 0.4e-6)
    fdtd.set("z", 0.0)
    fdtd.set("x", 0.0)
    fdtd.set("y", 0.0)
    fdtd.set("z", 0.0)
    
        
    fdtd.run()
    
    E = fdtd.getresult("E","E")
    E2 = E["E"]
    Ex1 = E2[:,:,:,0,0]
    Ey1 = E2[:,:,:,0,1]
    Ez1 = E2[:,:,:,0,2]
    Emag1 = np.sqrt(np.abs(Ex1)**2 + np.abs(Ey1)**2 + np.abs(Ez1)**2)
    Emag1 = Emag1[:,:,0]
    x1 = E["x"]
    x1 = x1[:,0]
    x1 = [i*1000000 for i in x1]
    y1 = E["y"]
    y1 = y1[:,0]
    y1 = [j*1000000 for j in y1]
    
    Emag1 = Emag1.transpose()
    Emag1_df = pd.DataFrame(Emag1)
    Emag1_df.to_csv('C:\\Users\\Lim Yudian\\Documents\\focusinggrating2Dgds\\all possible combinations of mixed pitch data\\'+g+'.csv', index=False)
    
    fig,ax=plt.subplots(1,1)
    cp=ax.contourf(x1,y1,Emag1, 200, zdir='z', offset=-100, cmap='jet')
    clb=fig.colorbar(cp)
    clb.ax.set_title('Electric Field (eV)', fontweight="bold")
    for l in clb.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(12)
    ax.set_xlabel('x-position (µm)', fontsize=13, fontweight="bold", labelpad=1)
    ax.set_ylabel('y-position (µm)', fontsize=13, fontweight="bold", labelpad=1)
    ax.xaxis.label.set_fontsize(13)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(13)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.show()
    plt.close()
    
    fdtd.switchtolayout()
    fdtd.select(g)
    fdtd.delete()

import subprocess

# Specify the process name of the non-window application you want to close
process_name = "fdtd-solutions.exe"

# Find the process ID (PID) of the process by name
cmd = f'tasklist /FI "IMAGENAME eq {process_name}" /FO CSV'
output = subprocess.check_output(cmd, shell=True).decode().strip().split('\n')
print(output)
print(output[1])
if len(output) > 1:
    pid = int(output[1].split(',')[1].strip('"'))

    # Terminate the process by its PID
    subprocess.call(f"taskkill /F /T /PID {pid}")
else:
    print(f"No running process found with name {process_name}")