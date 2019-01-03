# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:05:33 2018

Help functions for strain simulations of a COMSOL model.

The data in a COMSOL output text file is 6 columns: 1-3 is the X Y and Z coordinates of the
data points. 4-6 is the u, v, w (= x, y z) displacement. 

@author: Sanna & Megan 
"""

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D


def read_COMSOLdata(path, sample):    
    path = 'C:\\ptypy_tmp\\tutorial\\Megan_scripts\\'
    #path = 'C:/Users/Sanna/Documents/Simulations/ptypySim/Bragg/'
    sample = '3D_u_disp_NW'
    file1 = np.loadtxt(path + sample +'.txt',skiprows=9)
    return file1
#file1 = read_COMSOLdata()  
#data = 3 # u=3 v=4, w =5 

# plot scatter data on COMSOL grid from mesh
def plot_3d_scatter(file1, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.5, 0.5, 1, 1])) # changes the scaling of the axes in the scatter plot
    sc=ax.scatter(file1[:,0],file1[:,1],file1[:,2], c=file1[:,3], marker ='o',cmap='jet')#,alpha=0.2)
    plt.title(title)
    plt.colorbar(sc); plt.axis('scaled')
    ax.set_xlabel('x [nm]'); ax.set_ylabel('y [nm]'); ax.set_zlabel('z [nm]')
    #plt.savefig('./savefig/'+ sample)

def rotate_data(file1, chi=0 , phi=0 , psi=0):
    
    # coordinates directly read from COMSOL model,axes centered around 0    
    coordx = file1[:,0] - np.mean(file1[:,0])
    coordy = file1[:,1] - np.mean(file1[:,1])
    coordz = file1[:,2] - np.mean(file1[:,2])
    # put the coordinates into asingle np arrary
    coord = np.transpose(np.array([coordx, coordy, coordz]))
       
    # define rotational angles 
    #chi = 90    # pos chi rotates counter clockwise around X
    #phi = 90    # pos chi rotates counter clockwise around Z
    #psi = 0    # pos psi rotates counter clockwise around Y  - it is the same as the theta motion
    
    # define rotation matrices Rx Ry Rz and apply to coordinate system
    # ------------------------------------------------------------------------
    Rx = np.array([[1, 0, 0], [0, np.cos(np.deg2rad(chi)), np.sin(np.deg2rad(chi))] ,[0, -np.sin(np.deg2rad(chi)), np.cos(np.deg2rad(chi))]])
    # the new rotated coord
    coord_rotx = np.transpose(Rx.dot(np.array(np.transpose(coord))))
    
    Rz = np.array([[np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi)), 0], [-np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi)), 0], [0, 0, 1]])
    coord_rotz = np.transpose(Rz.dot(np.array(np.transpose(coord_rotx)))) 
        
    Ry = np.array([[np.cos(np.deg2rad(psi)), 0, np.sin(np.deg2rad(psi))], [0, 1, 0], [-np.sin(np.deg2rad(psi)), 0, np.cos(np.deg2rad(psi))]])
    coord_rot = np.transpose(Ry.dot(np.array(np.transpose(coord_rotz))))
    
    # plot the different rotations
    # ------------------------------------------------------------------------
    
#    def plot_3d_scatter():
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        #ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.5, 0.5, 1, 1])) # changes the scaling of the axes in the scatter plot
#        sc=ax.scatter(coord_rotx[:,0]*1E9,coord_rotx[:,1]*1E9,coord_rotx[:,2]*1E9, c=file1[:,3], marker ='o',cmap='jet')#,alpha=0.2)
#        plt.title('Scatter plot from the raw data from COMSOL (base)')
#        plt.colorbar(sc); plt.axis('scaled')
#        ax.set_xlabel('x [nm]'); ax.set_ylabel('y [nm]'); ax.set_zlabel('z [nm]')
#        #plt.savefig('./savefig/'+ sample)
#    #plot_3d_scatter()
#    # plot scatter data on COMSOL grid from mesh, after rotation around z
#    def plot_3d_scatter():
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        #ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.5, 0.5, 1, 1])) # changes the scaling of the axes in the scatter plot
#        sc=ax.scatter(coord_rotz[:,0]*1E9,coord_rotz[:,1]*1E9,coord_rotz[:,2]*1E9, c=file1[:,3], marker ='o',cmap='jet')#,alpha=0.2)
#        plt.title('Scatter plot from the raw data from COMSOL (base)')
#        plt.colorbar(sc); plt.axis('scaled')
#        ax.set_xlabel('x [nm]'); ax.set_ylabel('y [nm]'); ax.set_zlabel('z [nm]')
#        #plt.savefig('./savefig/'+ sample)
#    #plot_3d_scatter()
#    # plot scatter data on COMSOL grid from mesh, after rotation around y
#    def plot_3d_scatter():
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        #ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.5, 0.5, 1, 1])) # changes the scaling of the axes in the scatter plot
#        sc=ax.scatter(coord_rot[:,0]*1E9,coord_rot[:,1]*1E9,coord_rot[:,2]*1E9, c=file1[:,3], marker ='o',cmap='jet')#,alpha=0.2)
#        plt.title('Scatter plot from the raw data from COMSOL (base)')
#        plt.colorbar(sc); plt.axis('scaled')
#        ax.set_xlabel('x [nm]'); ax.set_ylabel('y [nm]'); ax.set_zlabel('z [nm]')
#        #plt.savefig('./savefig/'+ sample)
#    plot_3d_scatter()
    return coord_rot  
#coord_rot = rotate_data(file1)

##### Calc H_110 for GaAs NW
def calc_H110():
    # lattice constant InP
    lattice_constant_a = 5.65E-10    #ZB ?
    d = lattice_constant_a / np.sqrt(8) 
    q_abs = 2*np.pi / d
    return q_abs
#H_110 = calc_H110()    
     	
##### Calc H_111
def calc_H111():
    # lattice constant InP
    lattice_constant_a = 5.93E-10    #WZ ?
    # distance between planes (111), correct?, NO if WZ, should be 200
    d = lattice_constant_a / np.sqrt(3) 
    q_abs = 2*np.pi / d
    return q_abs
#H_111 = calc_H111()  

# convert displacement data to phase
def calc_complex_obj(file1, Q_vect):
    phase = (1j*Q_vect*1E-9*file1[:,3]) 
    comp = np.copy(phase)
    #for all not nan, set amplitude to 1, and phase to phase values
    comp[np.invert(np.isnan(phase))] = 1 + 1j*np.imag(phase[np.invert(np.isnan(phase))]) 
    #comp[np.isnan(phase)] = 0
    comp[np.invert(np.isnan(phase))].real = 1
    return comp
#comp = calc_complex_obj(file1,H_110)    

