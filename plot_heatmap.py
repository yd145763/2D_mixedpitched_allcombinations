# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:16:08 2023

@author: limyu
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter
import math

labels = ['Focusing_z', 'Focusing_e_field', 'Focusing_FWHM', 'Focusing_x']
y_axis_labels = ["Focal Point Height (µm)", "Focal Point E-Field (eV)", "Focal Point FWHM (µm)", "Focal Point along x-axis (µm)"]

for label, y_axis_label in zip(labels, y_axis_labels):

    url = "https://raw.githubusercontent.com/yd145763/2D_mixedpitched_allcombinations/main/df_result.csv"
    df = pd.read_csv(url)
    
    pitch_legend = []
    
    
    filtering_pitches = 700,800,900,1000,1100
    for filtering_pitch in filtering_pitches:
        legend = str(filtering_pitch/1000)+'/1.2µm'
        pitch_legend.append(legend)
    
    x_labels = []
    for i in range(1,10):
        x_label = str(i)+':'+str(10-i)
        x_labels.append(x_label)
    
    fig = plt.figure(figsize=(7, 4))
    ax = plt.axes()
    for filtering_pitch in filtering_pitches:
        df_filterbypitch = df[df["small_p"] ==filtering_pitch]
        ax.scatter(df_filterbypitch['small_c'], df_filterbypitch[label], s = 50)

    for filtering_pitch in filtering_pitches:
        df_filterbypitch = df[df["small_p"] ==filtering_pitch]
        ax.plot(df_filterbypitch['small_c'], df_filterbypitch[label])
    
    #graph formatting     
    ax.tick_params(which='major', width=2.00)
    ax.tick_params(which='minor', width=2.00)
    ax.xaxis.label.set_fontsize(15)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(15)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(x_labels, weight='bold')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.xlabel("Mixed Ratio")
    plt.ylabel(y_axis_label)
    plt.legend(pitch_legend, prop={'weight': 'bold','size': 10}, loc = "upper right")
    
    plt.show()
    plt.close()

y_axis_labels = ["Focal Point\nHeight (µm)", "Focal Point\nE-Field (eV)", "Focal Point\nFWHM (µm)", "Focal Point\nx-position (µm)"]


df_heatmap = df
for filtering_pitch in filtering_pitches:
    df_heatmap = df_heatmap.replace(filtering_pitch, str(filtering_pitch/1000)+'/1.2')
for i in range(1,10):
    df_heatmap['small_c'] = df_heatmap['small_c'].replace(i, str(i)+':'+str(10-i))

import seaborn as sns
for label, y_axis_label in zip(labels, y_axis_labels):
    
    mat = df_heatmap.pivot('small_c', 'small_p', label)
    fig, ax = plt.subplots()
    ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold", fontsize = 12)
    ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold", fontsize = 12)
    sns.heatmap(mat, annot=True, cmap='jet', annot_kws={"fontsize":10, "fontweight":"bold"}, fmt=".2f")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight("bold")
    cbar.ax.set_title(y_axis_label, fontweight="bold")
    font = {'color': 'black', 'weight': 'bold', 'size': 12}
    ax.set_ylabel("Mixed Ratio", fontdict=font)
    ax.set_xlabel("Pitch Mixture (µm)", fontdict=font)
    ax.tick_params(axis='both', labelsize=12, size= 5, width=2)
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    plt.show()
    plt.close()