from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from skimage.morphology import disk, cube
from skimage.measure import label

import glob, os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from int_plot import int_sliceviewer
from scipy.optimize import curve_fit
from multiprocessing import Process, Pool

def sigmoid(x, x0=0.0, k=10):
    y = 1.0 / (1.0 + np.exp(-k * (x - x0)))
    return y

def convert_cart_sphere_withmask(datacart, mask, center, B):
    [lenx, leny, lenz] = datacart.shape
    idx_all_x, idx_all_y, idx_all_z = np.meshgrid(
        np.arange(lenx), np.arange(leny), np.arange(lenz), indexing="ij"
    )
    idx_all_x_flat = idx_all_x[mask]
    idx_all_y_flat = idx_all_y[mask]
    idx_all_z_flat = idx_all_z[mask]
    datacart_flat = datacart[mask]
    idx_all_flat = np.column_stack((idx_all_x_flat, idx_all_y_flat, idx_all_z_flat)).T
    r = ((B @ idx_all_flat).T - center).T
    rnorm = np.linalg.norm(r, axis="0")
    theta = np.arccos(r[2, :] / rnorm)
    phi = np.arctan2(r[1], r[0])
    data1d = np.column_stack((datacart_flat, rnorm, theta, phi))
    return data1d


def convert_cart_sphere(datacart, center, B):
    [lenx, leny, lenz] = datacart.shape
    idx_all_x, idx_all_y, idx_all_z = np.meshgrid(
        np.arange(lenx), np.arange(leny), np.arange(lenz), indexing="ij"
    )
    idx_all_x_flat = idx_all_x.flatten()
    idx_all_y_flat = idx_all_y.flatten()
    idx_all_z_flat = idx_all_z.flatten()
    datacart_flat = datacart.flatten()
    idx_all_flat = np.column_stack((idx_all_x_flat, idx_all_y_flat, idx_all_z_flat)).T
    r = ((B @ idx_all_flat).T - center).T
    rnorm = np.linalg.norm(r, axis="0")
    theta = np.arccos(r[2, :] / rnorm)
    phi = np.arctan2(r[1], r[0])
    data1d = np.column_stack((datacart_flat, rnorm, theta, phi))

    return data1d


def gaussian(x, amp, cen, sigma):
    return 1 / np.sqrt(2) / np.pi / sigma * np.exp(-((x - cen) ** 2) / (2 * sigma**2))

def multi_gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amp, cen, sigma = params[i : i + 3]
        y += gaussian(x, amp, cen, sigma)
    return y

def multi_gaussian_3d(x, *params):
    y = zeros_like(x)
    r = np.array([np.linalg.norm(i) for i in x])
    theta = np.array([math.acos(abs(i[2]) / np.linalg.norm(i)) for i in x])
    phi = np.array([math.atan2(i[1], i[0]) for i in x])

    def poly_param_angle(theta, phi, *param_angle):
        yt = param_angle[0]
        yp = param_angle[1]
        for i in range(2, len(param_angle), 2):
            power_theta, power_phi = param_angle[i : i + 2]
            yt += theta * yt * power_theta
            yp += phi * yp * power_phi
        return yt + yp

    for i in range(0, len(params), 9):
        amp, cen, sigma, pt1, pt2, pt3, pf1, pf2, pf3 = params[i : i + 9]
        polyangle_param = [pt1, pt2, pt3, pf1, pf2, pf3]
        y += gaussian(x, amp, cen, sigma) * poly_param_angle(
            theta, phi, polyangle_param
        )


def br_old(data, x, y, center):
    [lenx, leny, lenz] = data.shape
    databr = data.copy()
    for i in range(lenx):
        for j in range(leny):
            for k in range(lenz):
                r = B @ np.array([i, j, k]) - center
                radius = np.linalg.norm(r)
                br = np.sum(np.multiply(gaussian(x, 1, radius, 0.01), y)) / np.sum(
                    gaussian(x, 1, radius, 0.01)
                )
                databr[i, j, k] -= br
    # databr[np.where(databr<0)]=0
    return np.array(databr)


def br_new(data, x, y, var, B, center):
    i, j, k = np.meshgrid(
        np.arange(data.shape[0]),
        np.arange(data.shape[1]),
        np.arange(data.shape[2]),
        indexing="ij",
    )

    # Calculate r and radius for all points at once
    coords = np.stack([i, j, k], axis=-1).reshape(-1, 3)
    r = (coords @ B.T) - center
    # radius = np.linalg.norm(r, axis=1).reshape(data.shape)

    radius = np.linalg.norm(r, axis=1).reshape(-1)

    # Vectorized gaussian calculation
    gauss = np.array([gaussian(x, 1, single_r, 0.01) for single_r in radius])
    # print(gauss.shape)
    # aa=gauss@y
    # print(aa.shape,np.sum(gauss,axis=1).shape)
    br_values = gauss @ y / np.sum(gauss, axis=1)
    var_values = gauss @ var / np.sum(gauss, axis=1)
    # br_values = (gauss * y).sum(axis=1) / gauss.sum(axis=1)
    # br_values = np.sum(y * gaussian(x, 1, radius[:, np.newaxis], 0.01)) / np.sum(gaussian(x, 1, radius[:, np.newaxis], 0.01))
    # var_values = np.sum(var * gaussian(x, 1, radius[:, np.newaxis], 0.01)) / np.sum(gaussian(x, 1, radius[:, np.newaxis], 0.01))

    bg_mean = br_values.reshape(data.shape)
    bg_var = var_values.reshape(data.shape)
    zero_mask = np.where(bg_mean + bg_var > data)
    br_results = data - bg_mean
    br_results[zero_mask] = 0
    return br_results


def br_phi_old(data, x, y, center, phix, phiy):
    [lenx, leny, lenz] = data.shape
    # data1d=np.zeros((np.size(data),4))
    databr = data.copy()
    idx = np.meshgrid
    for i in range(lenx):
        for j in range(leny):
            for k in range(lenz):
                r = B @ np.array([i, j, k]) - center
                radius = np.linalg.norm(r)
                phi = np.atan2(r[1], r[0])
                phi_factor = (
                    3
                    * np.sum(np.multiply(gaussian(phix, 1, phi, 1), phiy))
                    / np.sum(gaussian(phix, 1, phi, 1))
                )
                # print('phi,phifactr',phi,phix,phi_factor)
                # print( 'gaussian',gaussian(phix,1,phi,1))
                br = (
                    np.sum(np.multiply(gaussian(x, 1, radius, 0.01), y))
                    / np.sum(gaussian(x, 1, radius, 0.01))
                    * phi_factor
                )

                # databr[i,j,k]-= br
                if databr[i, j, k] < limit * br:
                    databr[i, j, k] = 0
                else:
                    databr[i, j, k] -= br

    # databr[np.where(databr<0)]=0
    return np.array(databr)

def br_phi(data, x, y, var, B, center, phix, phiy):
    # Create coordinate grid
    i, j, k = np.meshgrid(
        np.arange(data.shape[0]),
        np.arange(data.shape[1]),
        np.arange(data.shape[2]),
        indexing="ij",
    )

    # Calculate r, radius and phi for all points at once
    coords = np.stack([i, j, k], axis=-1).reshape(-1, 3)
    r = (coords @ B.T) - center
    radius = np.linalg.norm(r, axis=1)
    phi = np.arctan2(r[:, 1], r[:, 0])

    # Vectorized gaussian calculations
    gauss_r = gaussian(x[:, None], 1, radius[:, None], 0.01)
    gauss_phi = gaussian(phix[:, None], 1, phi[:, None], 1)

    # Calculate br and phi factors
    br_values = (gauss_r * y).sum(axis=1) / gauss_r.sum(axis=1)
    var_values = (gauss_r * var).sum(axis=1) / gauss_r.sum(axis=1)
    phi_factor = 3 * (gauss_phi * phiy).sum(axis=1) / gauss_phi.sum(axis=1)

    br_total = (br_values * phi_factor).reshape(data.shape)

    # Apply limit condition
    mask = data < ((br_values + var_values) * phi_factor).reshape(data.shape)
    result = data.copy()
    result[mask] = 0
    result[~mask] -= br_total[~mask]

    return result


def fit_2d_polynomial(data, degree=2):
    """Fit polynomial surface to 3D data"""
    # Create coordinate grids
    x, y = np.meshgrid(
        np.arange(data.shape[0]), np.arange(data.shape[1]), indexing="ij"
    )

    # Prepare input points
    points = np.column_stack((x.flatten(), y.flatten()))
    z = data.flatten()

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(points)

    # Fit model
    reg = LinearRegression()
    reg.fit(X_poly, z)

    # Generate fitted surface
    z_fit = reg.predict(X_poly)
    surface = z_fit.reshape(data.shape)

    r2 = reg.score(X_poly, z)

    return surface, r2, reg.coef_


def br_gauss_capout(data1d_cut,)->np.array:
    ####################cap data ############################
    data1d_cut_cap = data1d_cut.copy()
    print('max',np.max(data1d_cut))
    print('min',np.min(data1d_cut))
    data_percentile_cap = 30
    #print(data_percentile_cap)
    data_percentile_cap = 2 * np.percentile(data1d_cut, 99.9)
    data_percentile_cap = 2 * np.percentile(data1d_cut, 99)
    if np.max(data1d_cut)<=data_percentile_cap :data_percentile_cap =  np.percentile(data1d_cut, 99.9)
    data_percentile_cap = 30
    print('Cap at ',data_percentile_cap)
    ################################################################################
    ################################################################################
    mask_largerthan_cap_tmp = np.array(np.where(data1d_cut_cap[:,0] > data_percentile_cap))
    mask_largerthan_cap =mask_largerthan_cap_tmp[0]
    print('0',mask_largerthan_cap.shape)
    print(mask_largerthan_cap)
    print(mask_largerthan_cap_tmp)
    mask_smallerthan_cap = np.where(data1d_cut_cap[:,0] <= data_percentile_cap)
    mask_smallerthan_cap =mask_smallerthan_cap[0]
    ################################################################################################################################
    # NOTE:
    #           above result has shape (1,Neffective_pixels)
    #           to use, needs to pick the first element
    ################################################################################################################################
    #data1d_cut_maskcap=data1d_cut[mask_smallerthan_cap,:]
    #data1d_cut_cap=data1d_cut_maskcap
    data1d_cut_cap[mask_largerthan_cap,0] = data_percentile_cap
    print(data1d_cut_cap.shape)
    print(data1d_cut_cap[mask_largerthan_cap,:].shape)

    ################################################################################
    ################################################################################
    ####################r weight for extrapolate###########################
    # shape (Nr_extroplate,Neffective_pixels)
    rxmax = 0.7
    rxmin = 0.0
    Nr = 10
    rx = np.linspace(rxmin, rxmax, Nr)
    rwidth = (rxmax - rxmin) / Nr
    gauss_wt_r = np.array([gaussian(data1d_cut_cap[:, 1], 1, i, rwidth) for i in rx])
    #print('0',gauss_wt_r.shape, data1d_cut_cap[:, 0].shape)


    #################### extroplated data on r grids ###########################
    # shape (Nr_extroplate)
    ry = np.sum(np.multiply(gauss_wt_r, data1d_cut_cap[:, 0]), axis=1) / np.sum( gauss_wt_r, axis=1)
    var_ry = ( np.sum(np.multiply(gauss_wt_r, data1d_cut_cap[:, 0] ** 2), axis=1) / np.sum(gauss_wt_r, axis=1) - ry**2)

    # print(ry.shape)

    #
    #################### angle weight for extrapolate###########################
    thetaxmax = +3.0
    thetaxmin = -3.0
    Ntheta = 10
    thetax = np.linspace(thetaxmin, thetaxmax, Ntheta)
    thetawidth = (thetaxmax - thetaxmin) / Ntheta
    gauss_wt_theta = np.array( [gaussian(data1d_cut_cap[:, 2], 1, i, thetawidth) for i in thetax])
    print(gauss_wt_theta.shape, data1d_cut_cap[:, 0].shape)
    #
    phixmax = +3.0
    phixmin = -3.0
    Nphi = 10
    phix = np.linspace(phixmin, phixmax, Nphi)
    phiwidth = (phixmax - phixmin) / Nphi
    gauss_wt_phi = np.array([gaussian(data1d_cut_cap[:, 3], 1, i, phiwidth) for i in phix])
    print(gauss_wt_phi.shape, data1d_cut_cap[:, 0].shape)



    #################### weights over spehrical grids###########################
    # shape (Nr_extroplate,Ntheta_extrapolate,Nphi_extrapolate,Neffective_pixels)

    original_y = data1d_cut_cap[:, 0]
    gauss_wt_r_reshape = gauss_wt_r[:, np.newaxis, np.newaxis, :]
    gauss_wt_theta_reshape = gauss_wt_theta[np.newaxis, :, np.newaxis, :]
    gauss_wt_phi_reshape = gauss_wt_phi[np.newaxis, np.newaxis, :, :]
    gauss_wt_all_3d = gauss_wt_r_reshape * gauss_wt_theta_reshape * gauss_wt_phi_reshape

    print('1',gauss_wt_all_3d.shape, (gauss_wt_all_3d @ original_y).shape)


    #################### flatten weights matrix over spehrical grids###########################
    # shape (Nr_extroplate*Ntheta_extrapolate*Nphi_extrapolate,Neffective_pixels)
    gauss_wt_all_3d_reshape = gauss_wt_all_3d.reshape(-1, gauss_wt_all_3d.shape[-1])

    #################### extrapolated data  on spehrical grids###########################
    # shape (Nr_extroplate,Ntheta_extrapolate,Nphi_extrapolate)
    angle_y = gauss_wt_all_3d @ original_y / np.sum(gauss_wt_all_3d, axis=3)
    print(angle_y.shape,gauss_wt_all_3d.shape,original_y.shape)
    ################################################################################
    ################################################################################
    # TODO introduce mask for peak on forward transfer (cartesian to spherical)
    gauss_wt_all_3d_mask=gauss_wt_all_3d[:,:,:,mask_smallerthan_cap] 
    original_y_mask= original_y[mask_smallerthan_cap] 
    angle_y = gauss_wt_all_3d[:,:,:,mask_smallerthan_cap] @ original_y[mask_smallerthan_cap] / np.sum(gauss_wt_all_3d[:,:,:,mask_smallerthan_cap], axis=3)
    angle_y = gauss_wt_all_3d_mask @ original_y_mask / np.sum(gauss_wt_all_3d_mask, axis=3)
    print(angle_y.shape,gauss_wt_all_3d_mask.shape,original_y_mask.shape)
    ################################################################################
    ################################################################################

    #################### expand var on r grid to spehrical grids###########################
    # shape (Nr_extroplate,Ntheta_extrapolate,Nphi_extrapolate)
    var_ry_tmp = var_ry[:, np.newaxis, np.newaxis]
    var_shape = np.ones(angle_y.shape)
    var_ry_full = var_ry_tmp * var_shape
    # shape (Nr_extroplate*Ntheta_extrapolate*Nphi_extrapolate)
    var_ry_reshape = var_ry_full.reshape(-1)



    ################################# convert spehrical grid to cartesian grid ################################
    # shape (Nr_extroplate*Ntheta_extrapolate*Nphi_extrapolate)
    fittedangle_y = angle_y.reshape(-1)

    ################################# smoothen with gaussian fitting: convert spehrical grid to cartesian grid ################################
    #TODO introduce more fitting/smoothening techniques
    bg = fittedangle_y @ gauss_wt_all_3d_reshape



    ################################################################################
    ################################################################################

    print(fittedangle_y.shape, gauss_wt_all_3d_reshape.shape)

    # shape Neffective_pixels
    var_full = var_ry_reshape @ gauss_wt_all_3d_reshape



    print(np.sum(data1d_cut_cap[:, 0] - bg))

    sm = data1d_cut[:, 0]

    brzeromask = np.where(sm < bg + var_full)

    br = sm - bg
    br[brzeromask] = 0
    print(np.sum(br))
    print(np.min(br))
    print(np.max(br))
    return br