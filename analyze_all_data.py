# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 06:31:53 2023

@author: limyu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy.signal import find_peaks, peak_widths
import math 


Big_Pitch = [1200, 1100, 1000, 900, 800]
Small_Pitch = [700,800,900,1000,1100]
Small_Count = [1,2,3,4,5,6,7,8,9]
Big_Count = [10-i for i in Small_Count]
small_p = []
small_c = []
big_p = []
big_c = []
Focusing_z = []
Focusing_e_field = []
Focusing_FWHM = []
Focusing_beam_waist = []
Focusing_x = []

#function to find the second highest value in a panda series
def second_highest(arr):
  sorted_arr = sorted(set(arr), reverse=True)
  return sorted_arr[1] if len(sorted_arr) > 1 else None

#function to find the FWHM and peaks in a panda series
def FWHM(x,y):
    peaks, _ = find_peaks(y)
    results_half = peak_widths(y, peaks, rel_height=0.5)
    #convert peaks from index to x
    height_plot = results_half[1]
    x_min = results_half[2]
    x_min_int = x_min.astype(int)
    x_min_plot = x[x_min_int]
    x_max = results_half[3]
    x_max_int = x_max.astype(int)
    x_max_plot = x[x_max_int]
    width = results_half[0]
    width_plot = x_max_plot - x_min_plot
    results_half_plot = (width_plot, height_plot, x_min_plot, x_max_plot)
    return peaks, results_half_plot

#function to find the waists and peaks in a panda series
def Waist(x,y):
    peaks, _ = find_peaks(y)
    results_half = peak_widths(y, peaks, rel_height=0.865)
    #convert peaks from index to x
    height_plot = results_half[1]
    x_min = results_half[2]
    x_min_int = x_min.astype(int)
    x_min_plot = x[x_min_int]
    x_max = results_half[3]
    x_max_int = x_max.astype(int)
    x_max_plot = x[x_max_int]
    width = results_half[0]
    width_plot = x_max_plot - x_min_plot
    results_half_plot = (width_plot, height_plot, x_min_plot, x_max_plot)
    return peaks, results_half_plot





for s in Small_Pitch:
    for b in Big_Pitch:
        if s>=b:
            print('small pitch larger than big pitch, doesnt make sense, skip')
            fig = plt.figure(figsize=(7, 4))
            ax = plt.axes()
            plt.title('small pitch larger than big pitch, doesnt make sense, skip')
            plt.show()
            plt.close()
            continue
        else:
            for sc, bc in zip(Small_Count, Big_Count):
                print('big'+str(b)+'-'+str(sc)+'_small'+str(s)+'-'+str(bc))
                print('importing data from github....')
                #append initial parameters for recording
                small_p.append(s)
                small_c.append(sc)
                big_c.append(bc)
                big_p.append(b)
                
                #import data as pdf
                url = 'https://raw.githubusercontent.com/yd145763/2D_mixedpitched_allcombinations/main/big'+str(b)+'-'+str(sc)+'_small'+str(s)+'-'+str(bc)+'.csv'
                df = pd.read_csv(url)
                print('imported data!')
                
                #set the range of x and z
                x = np.linspace(0, 150, num=df.shape[1])
                z = np.linspace(0, 65, num=df.shape[0])
                #set the column index and row index of df into x and z
                df.columns = x
                df.set_index(pd.Index(z), inplace=True)
                
                #calculate the grating length to identify the reactive near distance region
                grating_length = ((b*bc) + (s*sc))/1000
                reactive_distance = ((math.sqrt((grating_length**3)/1.092))*0.62)+12
                
                #plotting the contour plot of the whole df, the whole picture of light coupled out from grating
                colorbarmax = df.max().max()
        
                colorbartick = 9
        
                X,Z = np.meshgrid(x,z)
                df1 = df.to_numpy()
        
                #contour plot 
                fig = plt.figure(figsize=(7, 4))
                ax = plt.axes()
                cp=ax.contourf(X,Z,df, 200, zdir='z', offset=-100, cmap='jet')
                clb=fig.colorbar(cp, ticks=(np.linspace(0, colorbarmax, num = 6)).tolist())
                clb.ax.set_title('cnt/s', fontweight="bold")
                for l in clb.ax.yaxis.get_ticklabels():
                    l.set_weight("bold")
                    l.set_fontsize(15)
                ax.set_xlabel('x-position (µm)', fontsize=18, fontweight="bold", labelpad=1)
                ax.set_ylabel('z-position (µm)', fontsize=18, fontweight="bold", labelpad=1)
        
        
                ax.xaxis.label.set_fontsize(18)
                ax.xaxis.label.set_weight("bold")
                ax.yaxis.label.set_fontsize(18)
                ax.yaxis.label.set_weight("bold")
                ax.tick_params(axis='both', which='major', labelsize=15)
                ax.set_yticklabels(ax.get_yticks(), weight='bold')
                ax.set_xticklabels(ax.get_xticks(), weight='bold')
                ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
                plt.axhline(y=12, color='white', linestyle='--')
                plt.title('big'+str(b)+'-'+str(sc)+'_small'+str(s)+'-'+str(bc), fontweight = 'bold')
                plt.show()
                plt.close()
                
        
                #find the row index where upper part of TOX layer is located (which is z = 12)
                closest_index = min(range(len(z)), key=lambda i: abs(z[i] - 12))
                #filter the df, taking rows of above TOX layer only
                df_filtered = df.iloc[closest_index:, :]
                
                #find the maximum e-field of each row in the filtered df
                max_e_field1 = df_filtered.max(axis=1)
                #reset the index of the max e-field of each row starting from 0
                max_e_field = max_e_field1.reset_index(drop=True)
                #express the new z after filtering df
                z_filtered = z[closest_index:]
                 
                #find the peaks and FWHM present in the plot of max e-field of each row (filtered df)
                max_e_field_FWHM_peaks, max_e_field_peak_FWHM  = FWHM(z_filtered,max_e_field)
           
                #express the height of each peak (plot of max e-field of each row in filtered df)
                max_e_field_peak_height = max_e_field_peak_FWHM[1]
                #find the peak position along z filtered (plot of max e-field of each row in filtered df)
                max_e_field_peak_position = z_filtered[max_e_field_FWHM_peaks]
                #express the FWHM of each peak (plot of max e-field of each row in filtered df)
                max_e_field_FWHM = pd.Series(max_e_field_peak_FWHM[0])
                #find the widest FWHM value (plot of max e-field of each row in filtered df)
                highest_e_field_FWHM = max(max_e_field_FWHM)
                #find the second widest FWHM value (plot of max e-field of each row in filtered df)
                second_highest_e_field_FWHM = second_highest(max_e_field_FWHM)
        
                #find the index of the widest FWHM value among the found peaks (plot of max e-field of each row in filtered df)
                highest_e_field_FWHM_index = max_e_field_FWHM.idxmax()
                #find the index of the second widest FWHM value among the found peaks (plot of max e-field of each row in filtered df)
                second_highest_e_field_FWHM_index = np.where(max_e_field_peak_FWHM[0] == second_highest_e_field_FWHM)[0]
                second_highest_e_field_FWHM_index = second_highest_e_field_FWHM_index[0]
      
                #if the height of the peak where second widest FWHM islocated, is very much lower than the height of the peak 
                #where second widest FWHM is located, we will consider the location of the second widest FWHM (aka the higher peak)
                if max_e_field_peak_height[second_highest_e_field_FWHM_index] > 2*max_e_field_peak_height[highest_e_field_FWHM_index]:
                    highest_e_field_FWHM_index = second_highest_e_field_FWHM_index
        
                #find the index and z value of the peak along z that fulfills the focusing condition, which is having the widest FWHM with acceptable peak height in the plot of E-field along z-axis
                focusing_z_index = max_e_field_FWHM_peaks[highest_e_field_FWHM_index]
                focusing_z = z_filtered[focusing_z_index]
                
                #find the e-field corresponding to focusing z
                focusing_e_field = max_e_field_peak_height[highest_e_field_FWHM_index]
                
                #we want to locate at which row the focusing z is located
                focusing_z_index_in_df = min(range(len(z_filtered)), key=lambda i: abs(z_filtered[i] - focusing_z))
                
                #express the E-field plot along the row of focusing z and reset index from 0
                e_field_along_focusing_z_in_df = df_filtered.iloc[focusing_z_index_in_df, :]
                e_field_along_focusing_z_in_df = e_field_along_focusing_z_in_df.reset_index(drop=True)
                
                max_e_field_along_focusing_z_in_df_index = e_field_along_focusing_z_in_df[e_field_along_focusing_z_in_df == max(e_field_along_focusing_z_in_df)].index[0]
                  
                focusing_x = x[max_e_field_along_focusing_z_in_df_index]
                
                #find the peaks and FWHM of e-field plots along focusing z level in filtered df
                e_field_along_focusing_z_FWHM_peaks, e_field_along_focusing_z_FWHM= FWHM(x, e_field_along_focusing_z_in_df)
        
                #express the FWHM of e-field plots along focusing z level in filtered df
               
                focusing_z_index = min(range(len(e_field_along_focusing_z_FWHM[1])), key=lambda i: abs(e_field_along_focusing_z_FWHM[1][i] - focusing_e_field)) 
                focusing_fwhm = e_field_along_focusing_z_FWHM[0][focusing_z_index]
                
                #express the beam waists of e-field plots along focusing z level in filtered df
             
                e_field_along_focusing_z_waist_peaks, e_field_along_focusing_z_waist = Waist(x,e_field_along_focusing_z_in_df)
                focusing_beam_waist = e_field_along_focusing_z_waist[0][focusing_z_index]
                
                #express the filtering condition of focusing
                if focusing_z < reactive_distance:
                    Focusing_z.append(focusing_z)
                    Focusing_e_field.append(focusing_e_field)
                    Focusing_FWHM.append(focusing_fwhm)
                    Focusing_beam_waist.append(focusing_beam_waist)
                    Focusing_x.append(focusing_x)
                
                else:
                    Focusing_z.append(math.nan)
                    Focusing_e_field.append(math.nan)
                    Focusing_FWHM.append(math.nan)
                    Focusing_beam_waist.append(math.nan)
                    Focusing_x.append(math.nan)
                
                fig = plt.figure(figsize=(7, 4))
                ax = plt.axes()
                ax.plot(x, e_field_along_focusing_z_in_df, color = "red")
                ax.plot(x[e_field_along_focusing_z_FWHM_peaks], e_field_along_focusing_z_in_df[e_field_along_focusing_z_FWHM_peaks], "o")
                ax.hlines(*e_field_along_focusing_z_FWHM[1:], color="C2")
                ax.hlines(*e_field_along_focusing_z_waist[1:], color="C3")
                ax.tick_params(which='major', width=2.00)
                ax.tick_params(which='minor', width=2.00)
                ax.xaxis.label.set_fontsize(15)
                ax.xaxis.label.set_weight("bold")
                ax.yaxis.label.set_fontsize(15)
                ax.yaxis.label.set_weight("bold")
                ax.tick_params(axis='both', which='major', labelsize=15)
                ax.set_yticklabels(ax.get_yticks(), weight='bold')
                ax.set_xticklabels(ax.get_xticks(), weight='bold')
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.spines['bottom'].set_linewidth(2)
                ax.spines['left'].set_linewidth(2)
                plt.xlabel("Height, z (µm)")
                plt.ylabel("Maximum E-Field (eV)")
                plt.legend(["E-field (eV)", "Peaks", "FWHM"], prop={'weight': 'bold'})
                plt.title('big'+str(b)+'-'+str(sc)+'_small'+str(s)+'-'+str(bc), fontweight = 'bold')
                plt.show()
                plt.close()
                
                
                fig = plt.figure(figsize=(7, 4))
                ax = plt.axes()
                ax.plot(z_filtered, max_e_field, color = "red")
                ax.plot(z_filtered[max_e_field_FWHM_peaks], max_e_field[max_e_field_FWHM_peaks], "o")
                ax.hlines(*max_e_field_peak_FWHM[1:], color="C2")
                #graph formatting     
                ax.tick_params(which='major', width=2.00)
                ax.tick_params(which='minor', width=2.00)
                ax.xaxis.label.set_fontsize(15)
                ax.xaxis.label.set_weight("bold")
                ax.yaxis.label.set_fontsize(15)
                ax.yaxis.label.set_weight("bold")
                ax.tick_params(axis='both', which='major', labelsize=15)
                ax.set_yticklabels(ax.get_yticks(), weight='bold')
                ax.set_xticklabels(ax.get_xticks(), weight='bold')
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.spines['bottom'].set_linewidth(2)
                ax.spines['left'].set_linewidth(2)
                plt.xlabel("Height, z (µm)")
                plt.ylabel("Maximum E-Field (eV)")
                plt.title('big'+str(b)+'-'+str(sc)+'_small'+str(s)+'-'+str(bc), fontweight = 'bold')
                plt.show()
                plt.close()
                
                #contour plot 
                fig = plt.figure(figsize=(7, 4))
                ax = plt.axes()
                cp=ax.contourf(X[closest_index:],Z[closest_index:],df_filtered, 200, zdir='z', offset=-100, cmap='jet')
                clb=fig.colorbar(cp, ticks=(np.linspace(0, colorbarmax, num = 6)).tolist())
                clb.ax.set_title('cnt/s', fontweight="bold")
                for l in clb.ax.yaxis.get_ticklabels():
                    l.set_weight("bold")
                    l.set_fontsize(15)
                ax.axhline(y = focusing_z, color = 'white', linestyle = '--')
                
                ax.set_xlabel('x-position (µm)', fontsize=18, fontweight="bold", labelpad=1)
                ax.set_ylabel('z-position (µm)', fontsize=18, fontweight="bold", labelpad=1)
                ax.xaxis.label.set_fontsize(18)
                ax.xaxis.label.set_weight("bold")
                ax.yaxis.label.set_fontsize(18)
                ax.yaxis.label.set_weight("bold")
                ax.tick_params(axis='both', which='major', labelsize=15)
                ax.set_yticklabels(ax.get_yticks(), weight='bold')
                ax.set_xticklabels(ax.get_xticks(), weight='bold')
                ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
                plt.title('big'+str(b)+'-'+str(sc)+'_small'+str(s)+'-'+str(bc), fontweight = 'bold')
                plt.show()
                plt.close()
                
                fig = plt.figure(figsize=(7, 4))
                ax = plt.axes()
                if focusing_z < reactive_distance:
                    plt.title("got focus")
                else:
                    plt.title("no focus")
                plt.show()
                plt.close()
        
                max_FWHM_list = []
                for index, row in df_filtered.iterrows():
                    peaks, _ = find_peaks(row.values)
                    results_half = peak_widths(row.values, peaks, rel_height=0.5)
                    #convert peaks from index to x
                    height_plot = results_half[1]
                    x_min = results_half[2]
                    x_min_int = x_min.astype(int)
                    x_min_plot = row.index[x_min_int]
                    x_max = results_half[3]
                    x_max_int = x_max.astype(int)
                    x_max_plot = row.index[x_max_int]
                    width = results_half[0]
                    width_plot = x_max_plot - x_min_plot
                    results_half_plot = (width_plot, height_plot, x_min_plot, x_max_plot)
                    
                    FWHM_list = results_half_plot[0]
                    peak_list = results_half_plot[1]
                    max_index = np.argmax(peak_list)
                    max_FWHM = FWHM_list[max_index]
                    max_FWHM_list.append(max_FWHM)
          
df_results = pd.DataFrame()
df_results["small_p"] = small_p
df_results["small_c"] = small_c
df_results["big_p"] = big_p
df_results["big_c"] = big_c
df_results["Focusing_z"] = Focusing_z
df_results["Focusing_x"] = Focusing_x
df_results["Focusing_e_field"] = Focusing_e_field
df_results["Focusing_FWHM"] = Focusing_FWHM 
df_results["Focusing_beam_waist"] = Focusing_beam_waist

df_results.to_csv('C:\\Users\\limyu\\Google Drive\\Machine Learning photonics\\2D grating all possible mixtures\\df_result.csv', index=False)

"""
            ax2 = plt.axes()
            ax2.plot(row.index, row.values)
            ax2.plot(row.index[peaks], row.values[peaks], "o")
            ax2.hlines(*results_half_plot[1:], color="C2")
            ax2.tick_params(which='major', width=2.00)
            ax2.tick_params(which='minor', width=2.00)
            ax2.xaxis.label.set_fontsize(18)
            ax2.xaxis.label.set_weight("bold")
            ax2.yaxis.label.set_fontsize(18)
            ax2.yaxis.label.set_weight("bold")
            ax2.tick_params(axis='both', which='major', labelsize=15)
            ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
            ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
            ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
            ax2.spines["right"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.spines['bottom'].set_linewidth(2)
            ax2.spines['left'].set_linewidth(2)
            plt.xlabel("x-position (µm)")
            plt.ylabel("Photon count (cnt/s)")
            plt.legend(["E-field (eV)", "Peaks", "FWHM"], prop={'weight': 'bold'})
            plt.show()
            plt.close()
"""
            


"""
s1 = 1100
sc1 = 5
bc1 = 5
url_single = 'https://raw.githubusercontent.com/yd145763/2D_mixedpitched_allcombinations/main/big1200-'+str(sc1)+'_small'+str(s1)+'-'+str(bc1)+'.csv'
df_single = pd.read_csv(url_single)

x = np.linspace(0, 150, num=df_single.shape[1])
z = np.linspace(0, 65, num=df_single.shape[0])
df_single.columns = x
df_single.set_index(pd.Index(z), inplace=True)

colorbarmax = df_single.max().max()

colorbartick = 9

X,Z = np.meshgrid(x,z)


#contour plot for multiple beams
fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
cp=ax.contourf(X,Z,df_single, 200, zdir='z', offset=-100, cmap='jet')
clb=fig.colorbar(cp, ticks=(np.linspace(0, colorbarmax, num = 6)).tolist())
clb.ax.set_title('cnt/s', fontweight="bold")
for l in clb.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)
ax.set_xlabel('x-position (µm)', fontsize=18, fontweight="bold", labelpad=1)
ax.set_ylabel('z-position (µm)', fontsize=18, fontweight="bold", labelpad=1)

ax.xaxis.label.set_fontsize(18)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(18)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
plt.axhline(y=15, color='white', linestyle='--')
plt.show()
plt.close()

max_FWHM_list = []

closest_index_single = min(range(len(z)), key=lambda i: abs(z[i] - 10))
df_single_filtered = df_single.iloc[closest_index_single:, :]
# Iterate over each row and plot
for index, row in df_single_filtered.iterrows():
    peaks, _ = find_peaks(row.values)
    results_half = peak_widths(row.values, peaks, rel_height=0.5)
    #convert peaks from index to x
    height_plot = results_half[1]
    x_min = results_half[2]
    x_min_int = x_min.astype(int)
    x_min_plot = row.index[x_min_int]
    x_max = results_half[3]
    x_max_int = x_max.astype(int)
    x_max_plot = row.index[x_max_int]
    width = results_half[0]
    width_plot = x_max_plot - x_min_plot
    results_half_plot = (width_plot, height_plot, x_min_plot, x_max_plot)
    
    FWHM_list = results_half_plot[0]
    peak_list = results_half_plot[1]
    max_index = np.argmax(peak_list)
    max_FWHM = FWHM_list[max_index]
    max_FWHM_list.append(max_FWHM)
    ax2 = plt.axes()

    ax2.plot(row.index, row.values)
    ax2.plot(row.index[peaks], row.values[peaks], "o")
    ax2.hlines(*results_half_plot[1:], color="C2")
    ax2.tick_params(which='major', width=2.00)
    ax2.tick_params(which='minor', width=2.00)
    ax2.xaxis.label.set_fontsize(18)
    ax2.xaxis.label.set_weight("bold")
    ax2.yaxis.label.set_fontsize(18)
    ax2.yaxis.label.set_weight("bold")
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
    ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    plt.xlabel("x-position (µm)")
    plt.ylabel("Photon count (cnt/s)")
    plt.legend(["Photon count (cnt/s)", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
    plt.show()
    plt.close()
    
    print(len(max_FWHM_list))
    print(len(z[closest_index_single:]))
    plt.scatter(z[closest_index_single:], max_FWHM_list)
    plt.show()

    plt.plot(row.index, row.values, label=f'Row {index+1}')

    # Add legend and labels
    plt.xlabel('Column')
    plt.ylabel('Value')

    # Show the plot
    plt.show()
    plt.close()
"""