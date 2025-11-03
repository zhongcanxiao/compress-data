from mantid.simpleapi import *
import skimage
#from clip_lines import clip_lines

import numpy as np
def clip_lines(x1, y1, z1, x2, y2, z2, box_min, box_max):
########## redcue the outershell vector, that extends out of box, to the box boundary
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



LoadEventNexus(Filename='data/TOPAZ_44752.nxs.h5', OutputWorkspace='TOPAZ_44752.nxs', 
               FilterByTofMin=1000, FilterByTofMax=16660, FilterByTimeStop=1000)
#FilterBadPulses(InputWorkspace='TOPAZ_44752.nxs', OutputWorkspace='TOPAZ_44752.nxs', LowerCutoff=95)
LoadIsawUB(InputWorkspace='TOPAZ_44752.nxs', Filename='data/TOPAZ_44752_Tetragonal.mat')

ConvertToMD(InputWorkspace='TOPAZ_44752.nxs', QDimensions='Q3D', dEAnalysisMode='Elastic', Q3DFrames='HKL' ,
    QConversionScales='Orthogonal HKL', OutputWorkspace='TOPAZ_44752.md.hkl', 
    LorentzCorrection=True, Uproj='1,0,0', Vproj='0,1,0', Wproj='0,0,1', 
     MinValues='-25,-25,-25', MaxValues='25,25,25')

BinMD(InputWorkspace='TOPAZ_44752.md.hkl', AxisAligned=True, 
    AlignedDim0='[H,0,0],-25,25,500', 
    AlignedDim1='[0,K,0],-25,25,500', 
    AlignedDim2='[0,0,L],-25,25,500', 
     OutputWorkspace='TOPAZ_44752_md_3D')

w = mtd['TOPAZ_44752_md_3D']
data_hkl = w.getSignalArray().copy()
np.save('44752.hkl.npy', data_hkl.astype(np.float32))

data_qsample = w.getSignalArray().copy()

wl_min, wl_max = 0.3, 3.5
CloneWorkspace(InputWorkspace='TOPAZ_44752_md_3D', OutputWorkspace='TOPAZ_44752_md_3D_mask')

#def generate_mask(ws,mask):

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

Rmantid = mtd['TOPAZ_44752_md_3D_mask'].getExperimentInfo(0).run().getGoniometer().getR()
UBmantid=mtd['TOPAZ_44752_md_3D_mask'].getExperimentInfo(0).sample().getOrientedLattice().getUB()

UBt=np.array([
[ 0.02816059, -0.06196579,  0.05857223], 
[-0.01873959, -0.06467350, -0.05941086], 
[ 0.08169699,  0.00629365, -0.03262031] ])

#its coordinates are a right-hand coordinate system where x is the beam direction and z is vertically upward.(IPNS convention)    
#its coordinates are a right-hand coordinate system where z is the beam direction and y is vertically upward.(mantid convention)    
# 1->3; 3->2; 2->1 
# 11 12 13
# 21 22 23
# 31 32 33
#
# 33 31 32
# 13 11 12
# 23 21 22



Rchi=np.array([
[ -0.7071067811865476, -0.7071067811865476, 0.], 
[ 0.7071067811865476,  -0.7071067811865476, 0.], 
[ 0.0000000000000000,  0.0000000000000000, 1.]] )

UBipns_t=np.array([
[ 0.02816059, -0.06196579,  0.05857223], 
[-0.01873959, -0.06467350, -0.05941086], 
[ 0.08169699,  0.00629365, -0.03262031]] )
UBipns=UBipns_t.T
permuteR=np.array([
[0, 1, 0], 
[0, 0, 1], 
[1, 0, 0] ] )
reflectxz=np.array([
[-1, 0, 0], 
[ 0, 1, 0], 
[ 0, 0,-1] ] )
reflectxyz=np.array([
[-1, 0, 0], 
[ 0,-1, 0], 
[ 0, 0,-1] ] )
UBxzreflect=permuteR@UBipns@permuteR.T

UB=reflectxz@UBxzreflect@reflectxz.T
UB=UBxzreflect
#UB=UBipns
R=Rmantid@Rchi
#R=Rmantid
#UB=reflectxz@UBxzreflect

R=Rmantid
UB=UBmantid
UB=UBmantid@reflectxyz
twopiRUB=2*np.pi*R@UB
invtwopiRUB=np.linalg.inv(twopiRUB)


Q1_1, Q2_1, Q3_1 = np.einsum('ij,jk->ik', invtwopiRUB, [Qx_1, Qy_1, Qz_1])
Q1_2, Q2_2, Q3_2 = np.einsum('ij,jk->ik', invtwopiRUB, [Qx_2, Qy_2, Qz_2])

Q1_bins = np.linspace(w.getDimension(0).getMinimum(), w.getDimension(0).getMaximum(), w.getDimension(0).getNBoundaries())
Q2_bins = np.linspace(w.getDimension(1).getMinimum(), w.getDimension(1).getMaximum(), w.getDimension(1).getNBoundaries())
Q3_bins = np.linspace(w.getDimension(2).getMinimum(), w.getDimension(2).getMaximum(), w.getDimension(2).getNBoundaries())

Q1_bins = 0.5*(Q1_bins[1:]+Q1_bins[:-1])
Q2_bins = 0.5*(Q2_bins[1:]+Q2_bins[:-1])
Q3_bins = 0.5*(Q3_bins[1:]+Q3_bins[:-1])

box_min = [Q1_bins[0], Q2_bins[0], Q3_bins[0]]
box_max = [Q1_bins[-1], Q2_bins[-1], Q3_bins[-1]]

Q1_1, Q2_1, Q3_1, Q1_2, Q2_2, Q3_2 = clip_lines(Q1_1, Q2_1, Q3_1, Q1_2, Q2_2, Q3_2, box_min, box_max)

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

data_norm = data_qsample*norm
mtd['TOPAZ_44752_md_3D_mask'].clearOriginalWorkspaces()
mtd['TOPAZ_44752_md_3D_mask'].setSignalArray(norm)
# Output folder
output_folder = './'
file_path_mask = os.path.join(output_folder, 'mask_empty_44752_bool_5C-v2.npy')

# Save the array
np.save(file_path_mask, norm.astype(bool))

mask_loaded=np.load(output_folder+'mask_empty_44752_bool_5C.npy')
print(mask_loaded.shape)
print(mask_loaded.min(), mask_loaded.max())
#######################################################################################################
#
#LoadEventNexus(Filename='../data-9-16-hardcopy/TOPAZ_44752.nxs.h5', OutputWorkspace='TOPAZ_44752.nxs', 
#               FilterByTofMin=1000, FilterByTofMax=16660, FilterByTimeStop=1000)
#FilterBadPulses(InputWorkspace='TOPAZ_44752.nxs', OutputWorkspace='TOPAZ_44752.nxs', LowerCutoff=95)
#LoadIsawUB(InputWorkspace='TOPAZ_44752.nxs', Filename='../data-9-16-hardcopy/TOPAZ_44752_Tetragonal.mat')
#
#ConvertToMD(InputWorkspace='TOPAZ_44752.nxs', QDimensions='Q3D', dEAnalysisMode='Elastic', Q3DFrames='HKL' ,
#    QConversionScales='Orthogonal HKL', OutputWorkspace='TOPAZ_44752.md.hkl', 
#    LorentzCorrection=True, Uproj='1,0,0', Vproj='0,1,0', Wproj='0,0,1', 
#     MinValues='-25,-25,-25', MaxValues='25,25,25')
#
#
#BinMD(InputWorkspace='TOPAZ_44752_md', AxisAligned=True, 
#    AlignedDim0='[H,0,0],-25,25,500', 
#    AlignedDim1='[0,K,0],-25,25,500', 
#    AlignedDim2='[0,0,L],-25,25,500', 
#     OutputWorkspace='TOPAZ_44752_md_3D')
#
#CloneWorkspace(InputWorkspace='TOPAZ_44752_md_3D', OutputWorkspace='TOPAZ_44752_md_3D_mask')
#
#w = mtd['TOPAZ_44752_md_3D']
#data_hkl = w.getSignalArray().copy()
#np.save('44752.hkl.npy', data_hkl.astype(np.float32))
#
#######################################################################################################
#
##mtd['TOPAZ_44752_md_3D_mask'].clearOriginalWorkspaces()
##mtd['TOPAZ_44752_md_3D_mask'].setSignalArray(data_norm)
