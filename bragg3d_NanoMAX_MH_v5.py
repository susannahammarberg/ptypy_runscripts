'''
Script for reading COMSOL simulated data and create simulated Bragg 


This script is adapted from 'bragg3d_initial' written by Alxander Björling. 
Alex: 
    This script is a demonstration of how the new container and geometry
    classes can be used to do 3d Bragg ptychography. Everything is done
    manually on the View and Geo_Bragg instances, no PtyScan objects or
    model managers or pods are involved yet.

Written by Megan and Sanna
'''

import ptypy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mayavi.mlab as mlab
import scipy.interpolate as inter
import sys
#sys.path.insert(0, r'C:\\ptypy_tmp\\tutorial\\Megan_scripts\\')
sys.path.insert(0, 'C:/Users/Sanna/Documents/Simulations/ptypySim/Bragg/')
from read_COMSOL_data import read_COMSOLdata
from read_COMSOL_data import rotate_data
from read_COMSOL_data import calc_H111
from read_COMSOL_data import calc_H110
from read_COMSOL_data import calc_complex_obj
from read_COMSOL_data import plot_3d_scatter
plt.close("all")

# Set up a 3D geometry and a scan
# -------------------------------

# The following geometry corresponds to an oversampling ratio of 1 along
# the rocking curve, which means that the numerical field of view
# tightly contains the exit wave.
g = ptypy.core.geometry_bragg.Geo_Bragg(psize=(0.01/4, 55e-6, 55e-6), shape=(9*21, 300, 300), energy=10.4, distance=2.0, theta_bragg=17.3)

# The Geo_Bragg object contains mostly the same things as Geo, but in
# three dimensions. The first element of the shape is the number of
# rocking curve positions, the first element of the psize denotes theta
# step in degrees. 
print g

# Set up scan positions along y, perpendicular to the incoming beam and
# to the thin layer stripes.
Npos = 3
positions = np.zeros((Npos,3))
positions[:, 2] = np.arange(Npos) - Npos/2.0
positions *= .5e-6

# Set up the object and its views
# -------------------------------

# Create a container for the object array, which will represent the
# object in the non-orthogonal coordinate system conjugate to the
# q-space measurement frame.
C = ptypy.core.Container(data_type=np.complex128, data_dims=3)

# For each scan position in the orthogonal coordinate system, find the
# natural coordinates and create a View instance there.
views = []
for pos in positions:
    pos_ = g._r3r1r2(pos)
    views.append(ptypy.core.View(C, storageID='Sobj', psize=g.resolution, coord=pos_, shape=g.shape))
S = C.storages['Sobj']
C.reformat()

# Define the test sample based on the orthogonal position of each voxel.
# First, the cartesian grid is obtained from the geometry object, then
# this grid is used as a condition for the sample's magnitude.
xx, zz, yy = g.transformed_grid(S, input_space='real', input_system='natural')
#TODO ?
S.fill(0.0)

# 4D arrays --> 3D arrays
xx = np.squeeze(xx)
yy = np.squeeze(yy)
zz = np.squeeze(zz)

#---------------------------------------------------------
# Read in COMSOL model
# -------------------------------------------------------

# define path to COMSOL data
#path = 'C:\\ptypy_tmp\\tutorial\\Megan_scripts\\'
path = 'C:/Users/Sanna/Documents/Simulations/ptypySim/Bragg/'
# define sample name (name of COMSOL data output-file)
sample = '3D_u_disp_NW'

# load the data (coordinates + displacement field)
file1 = np.loadtxt(path + sample +'.txt',skiprows=9)
# plot raw data
plot_3d_scatter(file1,title ='Scatter plot COMSOL raw data ('+str(sample)+')')
 
# Sample rotation
# Following Berenguer notation. (this is the order in which to rotate)
# pos chi rotates counter clockwise around X
# pos chi rotates counter clockwise around Z
# pos psi rotates counter clockwise around Y  - it is the same as the theta motion
coord_rot = rotate_data(file1, chi=90 , phi=90 , psi=0) # this is now in nm

# put together the data and the rotated coordinate system
coord_rot_data = np.concatenate((coord_rot, file1[:,3:]),axis=1)

# plot data in rotated coordinate system
plot_3d_scatter(coord_rot_data,title ='Scatter plot rotated data ('+str(sample)+')')
# calculate the scattering vector
H_110 = calc_H110()

# return a complex object from the displacement field. object defined by the COMSOL model
comp = calc_complex_obj(file1,H_110)

# temp comment: easier to work with [xx,1] array then [xx,]
# put together the data and the rotated coordinate system (for plotting)
coord_rot_comp = np.concatenate((coord_rot, comp.reshape(len(comp),1)),axis=1)
del file1
# plot the object in the rotated coordinate system
plot_3d_scatter(coord_rot_comp,title ='Object in rotated coordinates ('+str(sample)+')')

# define incoming and outcoming wave vectors ki and kf 
lam = (1.23981e-6)/(g.energy*1000)
th_tmp = g.theta_bragg
mag_qbragg = np.sqrt(2*(1-np.cos((2*th_tmp)*(np.pi/180))))   # calculate the length of the qvector given unit vectors of ki and kf (according to the 2theta value)
#TODO: not sure yet if this applies to all geometries of just horizontal scattering. 

ki = [1, 0, 0]
qbragg = np.array([0, 0, 1])   # this is fixed according to the geometry (q always along the [0 0 1])
kmag = 2*np.pi/lam
# rotate ki according to theta angle. again not sure if this applies to all geometries, need to consider Laue diffraction and non-horizontal scattering
Ry = np.array([[np.cos(np.deg2rad(-th_tmp)), 0, np.sin(np.deg2rad(-th_tmp))], [0, 1, 0], [-np.sin(np.deg2rad(-th_tmp)), 0, np.cos(np.deg2rad(-th_tmp))]])
ki = np.transpose(Ry.dot(np.array(np.transpose(ki)))) 

ki = -ki   #ki is pointing towards origin not away from origin, so do: [0,0,0]-ki
qbragg = qbragg*mag_qbragg  #this is the magintude determined by the distance between the unit vectors of ki and kf given 2theta, may not work for non-horizontal scattering
kf = qbragg + ki



# for clarity define the coordinate axes seperately
xi = coord_rot[:,0]
yi = coord_rot[:,1]
zi = coord_rot[:,2]

# to avoid having NaNs when interpolation the COMSOL data to a regular 
# meaurement grid, we create an index for all non-NaN values
ind1 = np.where(np.invert(np.isnan(comp)))

# interpolate the complex object calc from COMSOL simulation onto the grid 
# defined by the geometry of the experiment
s_1 = inter.griddata((xi[ind1],yi[ind1],zi[ind1]),comp[ind1],(xx,yy,zz))
S.fill(s_1)

#center the coordinate axes around 0
xx = xx-np.mean(xx)
yy = yy-np.mean(yy)
zz = zz-np.mean(zz)

# put the NaNs in the object to 0
ind = np.where(np.isnan(s_1))
s_1[ind] = 0
# calculate max and min values of coordinate axes (for plotting)
xmax = np.nanmax(xi); xmin = np.nanmin(xi)
ymax = np.nanmax(yi); zmin = np.nanmin(yi)
zmax = np.nanmax(zi); ymin = np.nanmin(zi)

ind = np.where(np.isnan(comp))
comp[ind] = 0


# plot isosurface of amplitude values and scattering vectors to visualize diffraction geometry. 
sc = mlab.figure()
src = mlab.pipeline.scalar_scatter(xi[ind1],yi[ind1],zi[ind1], comp.real[ind1], extent = [xmin, xmax, ymin, ymax, zmin, zmax])
g_1 = mlab.pipeline.glyph(src, mode='point')
gs = mlab.pipeline.gaussian_splatter(src)
gs.filter.radius = 0.15   #sets isosurface value, may need to tune. 
iso=mlab.pipeline.iso_surface(gs, transparent=True, opacity = 0.1)

qscale = np.round(np.max([xmax,ymax,zmax]))*1.5   # 1.5* the maximum value of the input geometry for the purpose of plotting the scattering vectors at visual scale. 
black = (0,0,0); red = (1,0,0); blue = (0,0,1); white = (1,1,1)
mlab.points3d(0, 0, 0, color=white, scale_factor=10)  #plot origin
#plot ki, kf, and qbragg scaled by size of object, qscale (for visualization purposes only)
mlab.plot3d([0, -ki[0]*qscale], [0, -ki[1]*qscale], [0, -ki[2]*qscale], color=red, tube_radius=2.)
mlab.plot3d([0, kf[0]*qscale], [0, kf[1]*qscale], [0, kf[2]*qscale], color=black, tube_radius=2.)
mlab.plot3d([0, qbragg[0]*qscale], [0, qbragg[1]*qscale], [0, qbragg[2]*qscale], color=blue, tube_radius=2.)
mlab.text3d(-ki[0]*qscale+qscale*0.05, -ki[1]*qscale+qscale*0.05, -ki[2]*qscale+qscale*0.05, 'ki', color=red, scale=25.)
mlab.text3d(kf[0]*qscale+qscale*0.05, kf[1]*qscale+qscale*0.05, kf[2]*qscale+qscale*0.05, 'kf', color=black, scale=25.)
mlab.text3d(qbragg[0]*qscale+qscale*0.05, qbragg[1]*qscale+qscale*0.05, qbragg[2]*qscale+qscale*0.05, 'qbragg', color=blue, scale=25.)



#quit()

'''
for ii in range(60,130):
    fig=plt.figure(1)
    plt.clf()
    sc = plt.imshow(np.angle(s_1[ii,100:200,223:423]))
    plt.colorbar(sc); plt.axis('scaled')
    plt.pause(0.1)
'''



# Set up the probe and calculate diffraction patterns
# ---------------------------------------------------

# First set up a two-dimensional representation of the probe, with
# arbitrary pixel spacing. The probe here is defined as a 1.5 um by 3 um
# flat square, but this container will typically come from a 2d
# transmission ptycho scan of an easy test object.
Cprobe = ptypy.core.Container(data_dims=2, data_type='float')
Sprobe = Cprobe.new_storage(psize=10e-9, shape=500)
zi, yi = Sprobe.grids()
# square probe
Sprobe.data[(zi > -.75e-6) & (zi < .75e-6) & (yi > -1.5e-6) & (yi < 1.5e-6)] = 1
# gaussian probe
Sprobe.data = np.exp(-zi**2 / (2 * (.75e-6)**2) - yi**2 / (2 * (1.0e-6)**2))

# The Bragg geometry has a method to prepare a 3d Storage by extruding
# the 2d probe and interpolating to the right grid. The returned storage
# contains a single view compatible with the object views.
Sprobe_3d = g.prepare_3d_probe(Sprobe, system='natural')
probeView = Sprobe_3d.views[0]

# Calculate diffraction patterns by using the geometry's propagator.
diff = []
for v in views:
    diff.append(np.abs(g.propagator.fw(v.data * probeView.data))**2)
## Sanna
# plot the diffraction patterns
# ---------------------------------
    
plt.figure()
plt.imshow(Sprobe_3d.data[0][33])
    
# Visualize a single field of view with probe and object
# ------------------------------------------------------

# A risk with 3d Bragg ptycho is that undersampling along theta leads to
# a situation where the actual exit wave is not contained within the
# field of view. This plot shows the situation for this combination of
# object, probe, and geometry. Note that the thin sample is what makes
# the current test experiment possible.

# In order to visualize the field of view, we'll create a copy of the
# object storage and set its value equal to 1 where covered by the first
# view.
S_display = S.copy(owner=C, ID='Sdisplay')
S_display.fill(0.0)
S_display[S.views[0]] = 1

# Then, to see how the probe is contained by this field of view, we add
# the probe and the object itself to the above view.
S_display[S.views[0]] += probeView.data
S_display.data += S.data * 2

# To visualize how this looks in cartesian real space, make a shifted
# (nearest-neighbor interpolated) copy of the object Storage.
S_display_cart = g.coordinate_shift(S_display, input_system='natural', input_space='real', keep_dims=False)

# Plot that
fig, ax = plt.subplots(nrows=1, ncols=2)
x, z, y = S_display_cart.grids()
ax[0].imshow(np.mean(np.abs(S_display_cart.data[0]), axis=2).T, extent=[x.min(), x.max(), z.min(), z.max()], interpolation='none', origin='lower')
plt.setp(ax[0], ylabel='z', xlabel='x', title='side view')
ax[1].imshow(np.mean(np.abs(S_display_cart.data[0]), axis=1).T, extent=[x.min(), x.max(), y.min(), y.max()], interpolation='none', origin='lower')
plt.setp(ax[1], ylabel='y', xlabel='x', title='top view')

# Visualize the probe positions along the scan
# --------------------------------------------

# beam/sample overlap in non-orthogonal coordinates
import matplotlib.gridspec as gridspec
plt.figure()
gs = gridspec.GridSpec(Npos, 2, width_ratios=[3,1], wspace=.0)
ax, ax2 = [], []
r3, r1, r2 = Sprobe_3d.grids()
for i in range(len(S.views)):
    # overlap
    ax.append(plt.subplot(gs[i, 0]))
    ax[-1].imshow(np.mean(np.abs(views[i].data + probeView.data), axis=1), vmin=0, vmax=.07, extent=[r2.min(), r2.max(), r3.min(), r3.max()])
    plt.setp(ax[-1], xlabel='r2', ylabel='r3', xlim=[r2.min(), r2.max()], ylim=[r3.min(), r3.max()], yticks=[])
    # diffraction
    ax2.append(plt.subplot(gs[i, 1]))
    ax2[-1].imshow(diff[i][18,:,:])
    plt.setp(ax2[-1], ylabel='q1', xlabel='q2', xticks=[], yticks=[])
plt.suptitle('Probe, sample, and slices of 3d diffraction peaks along the scan')
plt.draw()
'''




'''
# Reconstruct the numerical data
# ------------------------------
'''
# Here I compare different algorithms and scaling options.
algorithm = 'PIE'

# Keep a copy of the object storage, and fill the actual one with an
# initial guess (like zeros everywhere).
S_true = S.copy(owner=C, ID='Strue')
# zero everything
S.fill(0.0)
# unit magnitude, random phase:
# S.data[:] = 1.0 * np.exp(1j * (2 * np.random.rand(*S.data.shape) - 1) * np.pi)
# random magnitude, random phase
# S.data[:] = np.random.rand(*S.data.shape) * np.exp(1j * (2 * np.random.rand(*S.data.shape) - 1) * np.pi)

# Here's an implementation of the OS (preconditioned PIE) algorithm from
# Pateras' thesis.
if algorithm == 'OS':
    alpha, beta = .1, 1.0
    fig, ax = plt.subplots(ncols=3)
    errors = []
    criterion = []
    # first calculate the weighting factor Lambda, here called scaling = 1/Lambda
    scaling = S.copy(owner=C, ID='Sscaling')
    scaling.fill(alpha)
    for v in views:
        scaling[v] += np.abs(probeView.data)**2
    scaling.data[:] = 1 / scaling.data
    # then iterate with the appropriate update rule
    for i in range(100):
        print i
        criterion_ = 0.0
        obj_error_ = 0.0
        for j in range(len(views)):
            prop = g.propagator.fw(views[j].data * probeView.data)
            criterion_ += np.sum(np.sqrt(diff[j]) - np.abs(prop))**2
            prop_ = np.sqrt(diff[j]) * np.exp(1j * np.angle(prop))
            gradient = 2 * probeView.data * g.propagator.bw(prop - prop_)
            views[j].data -= beta * gradient * scaling[views[j]]
        errors.append(np.abs(S.data - S_true.data).sum())
        criterion.append(criterion_)

        if not (i % 5):
            ax[0].clear()
            ax[0].plot(errors/errors[0])
            #ax[0].plot(criterion/criterion[0])
            ax[1].clear()
            S_cart = g.coordinate_shift(S, input_space='real', input_system='natural', keep_dims=False)
            x, z, y = S_cart.grids()
            ax[1].imshow(np.mean(np.abs(S_cart.data[0]), axis=1).T, extent=[x.min(), x.max(), y.min(), y.max()], interpolation='none', origin='lower')
            plt.setp(ax[1], ylabel='y', xlabel='x', title='top view')
            ax[2].clear()
            ax[2].imshow(np.mean(np.abs(S_cart.data[0]), axis=2).T, extent=[x.min(), x.max(), z.min(), z.max()], interpolation='none', origin='lower')
            plt.setp(ax[2], ylabel='z', xlabel='x', title='side view')
            plt.draw()
            plt.pause(.01)


# Here's a PIE/cPIE implementation
if algorithm == 'PIE':
    beta = 1.0
    eps = 1e-3
    fig, ax = plt.subplots(ncols=3)
    errors = []
    ferrors = []
    for i in range(10):
        print i
        ferrors_ = []
        for j in range(len(views)):
            exit_ = views[j].data * probeView.data
            prop = g.propagator.fw(exit_)
            ferrors_.append(np.abs(prop)**2 - diff[j])
            prop[:] = np.sqrt(diff[j]) * np.exp(1j * np.angle(prop))
            exit = g.propagator.bw(prop)
            # ePIE scaling (Maiden2009)
            #views[j].data += beta * np.conj(probeView.data) / (np.abs(probeView.data).max())**2 * (exit - exit_)
            # PIE and cPIE scaling (Rodenburg2004 and Godard2011b)
            views[j].data += beta * np.abs(probeView.data) / np.abs(probeView.data).max() * np.conj(probeView.data) / (np.abs(probeView.data)**2 + eps) * (exit - exit_)
        errors.append(np.abs(S.data - S_true.data).sum())
        ferrors.append(np.mean(ferrors_))

        if not (i % 5):
            ax[0].clear()
            ax[0].plot(errors/errors[0])
            ax[0].plot(ferrors/ferrors[0])
            #ax[0].plot(criterion/criterion[0])
            ax[1].clear()
            S_cart = g.coordinate_shift(S, input_space='real', input_system='natural', keep_dims=False)
            x, z, y = S_cart.grids()
            ax[1].imshow(np.mean(np.abs(S_cart.data[0]), axis=1).T, extent=[x.min(), x.max(), y.min(), y.max()], interpolation='none', origin='lower')
            plt.setp(ax[1], ylabel='y', xlabel='x', title='top view')
            ax[2].clear()
            ax[2].imshow(np.mean(np.abs(S_cart.data[0]), axis=2).T, extent=[x.min(), x.max(), z.min(), z.max()], interpolation='none', origin='lower')
            plt.setp(ax[2], ylabel='z', xlabel='x', title='side view')
            plt.draw()
            plt.pause(.01)

if algorithm == 'DM':
    alpha = 1.0
    fig, ax = plt.subplots(ncols=3)
    errors = []
    ferrors = []
    # create initial exit waves
    exitwaves = []
    for j in range(len(views)):
        exitwaves.append(views[j].data * probeView.data)
    # we also need a constant normalization storage, which contains the
    # denominator of the DM object update equation.
    Snorm = S.copy(owner=C)
    Snorm.fill(0.0)
    for j in range(len(views)):
        Snorm[views[j]] += np.abs(probeView.data)**2

    # iterate
    for i in range(100):
        print i
        ferrors_ = []
        # fourier update, updates all the exit waves
        for j in range(len(views)):
            # in DM, you propagate the following linear combination
            im = g.propagator.fw((1 + alpha) * probeView.data * views[j].data - alpha * exitwaves[j])
            im = np.sqrt(diff[j]) * np.exp(1j * np.angle(im))
            exitwaves[j][:] += g.propagator.bw(im) - views[j].data * probeView.data
        # object update, now skipping the iteration because the probe is constant
        S.fill(0.0)
        for j in range(len(views)):
            views[j].data += np.conj(probeView.data) * exitwaves[j]
        S.data[:] /= Snorm.data + 1e-10
        errors.append(np.abs(S.data - S_true.data).sum())
        ferrors.append(np.mean(ferrors_))

        if not (i % 5):
            ax[0].clear()
            ax[0].plot(errors/errors[0])
            #ax[0].plot(criterion/criterion[0])
            ax[1].clear()
            S_cart = g.coordinate_shift(S, input_space='real', input_system='natural', keep_dims=False)
            x, z, y = S_cart.grids()
            ax[1].imshow(np.mean(np.abs(S_cart.data[0]), axis=1).T, extent=[x.min(), x.max(), y.min(), y.max()], interpolation='none', origin='lower')
            plt.setp(ax[1], ylabel='y', xlabel='x', title='top view')
            ax[2].clear()
            ax[2].imshow(np.mean(np.abs(S_cart.data[0]), axis=2).T, extent=[x.min(), x.max(), z.min(), z.max()], interpolation='none', origin='lower')
            plt.setp(ax[2], ylabel='z', xlabel='x', title='side view')
            plt.draw()
            plt.pause(.01)

plt.show()
'''