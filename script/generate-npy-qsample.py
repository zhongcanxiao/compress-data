# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import mantid.plots.datafunctions as mdf

import sys

sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python310.zip')
sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python3.10')
sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python3.10/lib-dynload')
sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python3.10/site-packages')

import skimage
from mantid import config
config['Q.convention'] = 'Crystallography'

#detector_mask='10-12,15,21,23-25,30-32,34,35,40-45,50-55'
#No sample goniometer position
z_offset =0.0
omega = mtd['TOPAZ_47994.nxs'].getRun()['omega'].value[0]
omega_modified = float(omega) - float(z_offset)

# LiFePO4 101 peak at 4 K
run=47805
Load(Filename='data/TOPAZ_47805.nxs.h5', OutputWorkspace='TOPAZ_47805.nxs', FilterByTofMin=1000, FilterByTofMax=16660)
#MaskBTP(Workspace=BKG, Bank=detector_mask)
#MaskBTP(Workspace=BKG, Pixel="0-18,237-255")
#MaskBTP(Workspace=BKG, Tube="0-18,237-255")
FilterBadPulses(InputWorkspace='TOPAZ_47994.nxs', OutputWorkspace='TOPAZ_47994.nxs', LowerCutoff=95)

omega_sample = mtd['TOPAZ_47805.nxs'].getRun()['omega'].value[0]
omega_modified_sample = float(omega_sample)
print('Sample Goniometer Omega = {: 8.3f}'.format(omega_sample))
proton_charge = mtd['TOPAZ_47805.nxs'].getRun().getProtonCharge() * 0.0036  # get proton charge

print("\nSample ", str(run), " has integrated proton charge of", proton_charge, "\n")

# Goniometer angle for empty can
omega = mtd['TOPAZ_47994.nxs'].getRun()['omega'].value[0]
omega_modified = omega
print('Use No-sample Goniometer Omega = {: 8.3f}'.format(omega))

AddSampleLog(Workspace='TOPAZ_47805.nxs', LogName='omega', 
    LogText='%.4f'%omega_modified, LogType='Number Series')
AddSampleLog(Workspace='TOPAZ_47805.nxs', LogName='omegaRequest', 
    LogText='%.4f'%omega_modified, LogType='Number Series')
#
SetGoniometer(Workspace='TOPAZ_47805.nxs', Goniometers='Universal')

Rebin(InputWorkspace='TOPAZ_47805.nxs', OutputWorkspace='TOPAZ_47805.nxs', Params='1000,10,16600')
ConvertToMD(InputWorkspace='TOPAZ_47805.nxs', QDimensions='Q3D', dEAnalysisMode='Elastic', Q3DFrames='Q_sample', 
    LorentzCorrection=True, Uproj='1,0,0', Vproj='0,1,0', Wproj='0,0,1', 
    OutputWorkspace='TOPAZ_47805_md', MinValues='-12.5,-12.5,-12.5', MaxValues='12.5,12.5,12.5')

BinMD(InputWorkspace='TOPAZ_47805_md', AxisAligned=False, 
    BasisVector0='Q_sample_x,Angstrom^-1,1.0,0.0,0.0', 
    BasisVector1='Q_sample_y,Angstrom^-1,0.0,1.0,0.0', 
    BasisVector2='Q_sample_z,Angstrom^-1,0.0,0.0,1.0', 
    OutputExtents='-12.5,12.5,-12.5,12.5,-12.5,12.5', 
    OutputBins='501,501,501', OutputWorkspace='TOPAZ_47805_md_3D')

CloneWorkspace(InputWorkspace='TOPAZ_47805_md_3D', OutputWorkspace='TOPAZ_47805_md_3D_mask')

w = mtd['TOPAZ_47805_md_3D']
data_qsample = w.getSignalArray().copy()

wl_min, wl_max = 0.3, 3.5

two_theta = np.array(mtd['PreprocessedDetectorsWS'].column('TwoTheta'))
azimthal = np.array(mtd['PreprocessedDetectorsWS'].column('Azimuthal'))

kx_hat = np.sin(two_theta)*np.cos(azimthal)
ky_hat = np.sin(two_theta)*np.sin(azimthal)
kz_hat = np.cos(two_theta)-1

Qx_1 = 2*np.pi/wl_max*kx_hat
Qy_1 = 2*np.pi/wl_max*ky_hat
Qz_1 = 2*np.pi/wl_max*kz_hat

Qx_2 = 2*np.pi/wl_min*kx_hat
Qy_2 = 2*np.pi/wl_min*ky_hat
Qz_2 = 2*np.pi/wl_min*kz_hat

R = mtd['TOPAZ_47805_md_3D_mask'].getExperimentInfo(0).run().getGoniometer().getR()

Q1_1, Q2_1, Q3_1 = np.einsum('ij,jk->ik', R.T, [Qx_1, Qy_1, Qz_1])
Q1_2, Q2_2, Q3_2 = np.einsum('ij,jk->ik', R.T, [Qx_2, Qy_2, Qz_2])

Q1_bins = np.linspace(w.getDimension(0).getMinimum(), w.getDimension(0).getMaximum(), w.getDimension(0).getNBoundaries())
Q2_bins = np.linspace(w.getDimension(1).getMinimum(), w.getDimension(1).getMaximum(), w.getDimension(1).getNBoundaries())
Q3_bins = np.linspace(w.getDimension(2).getMinimum(), w.getDimension(2).getMaximum(), w.getDimension(2).getNBoundaries())

Q1_bins = 0.5*(Q1_bins[1:]+Q1_bins[:-1])
Q2_bins = 0.5*(Q2_bins[1:]+Q2_bins[:-1])
Q3_bins = 0.5*(Q3_bins[1:]+Q3_bins[:-1])

box_min = [Q1_bins[0], Q2_bins[0], Q3_bins[0]]
box_max = [Q1_bins[-1], Q2_bins[-1], Q3_bins[-1]]

print(np.column_stack([Q1_1, Q1_2]))

def clip_lines(x1, y1, z1, x2, y2, z2, box_min, box_max):
    def clip_axis(p1, p2, axis_min, axis_max):
        t0 = (axis_min - p1) / (p2 - p1)
        t1 = (axis_max - p1) / (p2 - p1)
        tmin = np.minimum(t0, t1)
        tmax = np.maximum(t0, t1)
        return tmin, tmax

    # Clip for each axis
    tmin_x, tmax_x = clip_axis(x1, x2, box_min[0], box_max[0])
    tmin_y, tmax_y = clip_axis(y1, y2, box_min[1], box_max[1])
    tmin_z, tmax_z = clip_axis(z1, z2, box_min[2], box_max[2])

    # Overall tmin and tmax
    tmin = np.maximum(np.maximum(tmin_x, tmin_y), tmin_z)
    tmax = np.minimum(np.minimum(tmax_x, tmax_y), tmax_z)

    # Mask for valid intersections
    mask = tmin <= tmax

    # Ensure tmin is within the valid range [0, 1]
    tmin = np.clip(tmin, 0, 1)
    tmax = np.clip(tmax, 0, 1)

    # Clipping points
    x1_clipped = x1 + tmin * (x2 - x1)
    y1_clipped = y1 + tmin * (y2 - y1)
    z1_clipped = z1 + tmin * (z2 - z1)

    x2_clipped = x1 + tmax * (x2 - x1)
    y2_clipped = y1 + tmax * (y2 - y1)
    z2_clipped = z1 + tmax * (z2 - z1)

    # Apply mask to determine final points, keeping original points for lines that do not intersect the box
    x1_final = np.where(mask, x1_clipped, x1)
    y1_final = np.where(mask, y1_clipped, y1)
    z1_final = np.where(mask, z1_clipped, z1)

    x2_final = np.where(mask, x2_clipped, x2)
    y2_final = np.where(mask, y2_clipped, y2)
    z2_final = np.where(mask, z2_clipped, z2)

    return x1_final, y1_final, z1_final, x2_final, y2_final, z2_final

Q1_1, Q2_1, Q3_1, Q1_2, Q2_2, Q3_2 = clip_lines(Q1_1, Q2_1, Q3_1, Q1_2, Q2_2, Q3_2, box_min, box_max)

print(np.column_stack([Q1_1, Q1_2]))


Q1_1_ind = np.digitize(Q1_1, Q1_bins, right=True)
Q2_1_ind = np.digitize(Q2_1, Q2_bins, right=True)
Q3_1_ind = np.digitize(Q3_1, Q3_bins, right=True)

Q1_2_ind = np.digitize(Q1_2, Q1_bins, right=True)
Q2_2_ind = np.digitize(Q2_2, Q2_bins, right=True)
Q3_2_ind = np.digitize(Q3_2, Q3_bins, right=True)

_, indices = np.unique(np.column_stack([Q1_1_ind,Q2_1_ind,Q3_1_ind,
                                        Q1_2_ind,Q2_2_ind,Q3_2_ind]), axis=0, return_index=True)

norm = data_qsample*0

for i in indices:
    x1, y1, z1 = Q1_1_ind[i], Q2_1_ind[i], Q3_1_ind[i]
    x2, y2, z2 = Q1_2_ind[i], Q2_2_ind[i], Q3_2_ind[i]
    ix, iy, iz = skimage.draw.line_nd([x1,y1,z1], [x2,y2,z2], endpoint=False)
    norm[ix,iy,iz] = 1

data_norm = data_qsample/norm


mtd['TOPAZ_47805_md_3D_mask'].clearOriginalWorkspaces()
mtd['TOPAZ_47805_md_3D_mask'].setSignalArray(data_norm)
# Output folder
output_folder = '/SNS/TOPAZ/IPTS-31189/shared/background_removal'
output_folder = './'

# Create the folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# File path
file_path = os.path.join(output_folder, 'data_LFP_47805.npy')

# Save the array
np.save(file_path, data_qsample.astype(np.float32))

data_loaded=np.load('/SNS/TOPAZ/IPTS-31189/shared/background_removal/data_LFP_47805.npy')
print(data_loaded.shape)



# Save mask
# File path
file_path_mask = os.path.join(output_folder, 'mask__LFP_bool_47805.npy')

# Save the array
np.save(file_path_mask, norm.astype(bool))

mask_loaded=np.load(file_path_mask)
print(mask_loaded.shape)
print(mask_loaded.min(), mask_loaded.max())
