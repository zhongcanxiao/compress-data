import numpy as np
import math
import time
import warnings
import matplotlib.pyplot as plt
import random
import cv2
import cProfile
from PIL import Image
import copy
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce
from collections import deque
import itertools
from scipy.optimize import minimize

def safe_multiply_scalar(x, y, warn=True):
    """Scalar multiplication with build-in underflow error handling.

    Parameters
    ----------
    x : int or float
        Multiplicand
    y : int or float
        Multiplier
    warn : bool, optional
        Whether handled underflow errors would be warned, 
        by default True

    Returns
    -------
    int or float
        Product
    """
    try:
        return x * y
    except FloatingPointError:
        if warn:
            warnings.warn('Product rounded to 0.')
        return 0
    
def safe_divide_scalar(x, y, warn=True):
    try:
        return x / y
    except FloatingPointError:
        if warn:
            warnings.warn('Quotient rounded to 0.')
        return 0

def safe_multiply(array, coefficient, round=True, norm=False) -> np.ndarray:
    # TODO: add overflow error protection
    if coefficient >= 1:
        return array * coefficient
    
    if norm:
        target_sum = safe_multiply_scalar(np.sum(array), coefficient)

    if np.issubdtype(array.dtype, np.integer):
        # TODO: add integer, but i'm not sure what the point of it is
        raise NotImplementedError
    elif np.issubdtype(array.dtype, np.inexact):
        min = np.finfo(array.dtype).tiny / coefficient
    else:
        raise TypeError('Array must have an integer or inexact dtype subclass.')

    array[np.where(np.logical_and(array < min, array > -min))] = 0
    output = array * coefficient
    if norm:
        output *= target_sum / np.sum(output)
    return output

def safe_divide(array, divisor, round=True, norm=False):
    if divisor <= 1:
        return array / divisor
    
    if norm:
        target_sum = safe_divide_scalar(np.sum(array), divisor)
    
    if np.issubdtype(array.dtype, np.integer):
        # TODO: this is a hotfix prob not good to cast directly to float
        array = np.array(array, dtype=float)
        min = np.finfo(array.dtype).tiny * divisor
    elif np.issubdtype(array.dtype, np.inexact):
        min = np.finfo(array.dtype).tiny * divisor
    else:
        raise TypeError('Array must have an integer or inexact dtype subclass.')
    
    array[np.where(np.logical_and(array < min, array > -min))] = 0
    output = array / divisor
    if norm and not math.isclose(np.sum(output), target_sum, abs_tol=1e-8):
        output *= target_sum / np.sum(output)
    return output

def my_round(num, base=1):
    """Round a number to the nearest multiple of a base.

    Parameters
    ----------
    num : int or float
        Number that gets rounded
    base : int or float, optional
        Base whose nearest multiple is rounded to, by default 1

    Returns
    -------
    int or float
        Nearest multiple of base to num
    """
    return base * round(num / base)

def save(array, fname, format, cmap='viridis', vmin=None, vmax=None, scaling=None):
    """Save array as npy or image.

    Parameters
    ----------
    array : ndarray
        Array to be saved.
    fname : str
        File location to be saved at.
    format : str
        Descriptor to specify whether to save using numpy, matplotlib, 
        or pillow.
    cmap : str, optional
        matplotlib colormap for matplotlib and pillow, 
        by default 'viridis'
    vmin : int, float, or None, optional
        _description_, by default None
    vmax : _type_, optional
        _description_, by default None
    scaling : _type_, optional
        _description_, by default None

    Raises
    ------
    ValueError
        _description_
    """
    # TODO: add option to specify file extensions
    # TODO: add option for actual filepaths using os objects
    match format:
        case 'n' | 'np' | 'npy' | 'numpy' | 'npy':
            np.save(f'{fname}.npy', array)
        case 'm' | 'mpl' | 'matplotlib' | 'plt' | 'pyplot':
            img = plt.imshow(array, interpolation='none', vmin=vmin, vmax=vmax)
            img.set_cmap(plt.get_cmap(cmap))
            plt.axis('off')
            plt.colorbar()
            plt.savefig(f'{fname}.png', bbox_inches='tight')
        case 'p' | 'pil' | 'pillow':
            # TODO: figure out what to do for potential underflow errors
            if scaling == 0:
                raise ValueError('Scaling is 0, divide by 0 error.')
            if np.max(array) == 0:
                warnings.warn('Array is all 0.')
            norm = safe_divide(array, np.max(array)) if scaling is None else safe_divide(array, scaling)
            if np.max(norm) > 255:
                recommended_scaling = np.max(norm) / 255
                warnings.warn(f'Saved PIL image has pixels with >255 luminance. Recommended to scale by {np.round(recommended_scaling, 2)}.')
            im = Image.fromarray(np.uint8(255 * plt.get_cmap(cmap)(norm)))
            im.save(f'{fname}.png')
        case _:
            raise ValueError('Unsupported save format.')
        
def regression_loss(arr1, arr2, loss_mask=None, function='mae'):
    if np.isscalar(arr1):
        arr1 = np.full_like(arr2, arr1)
    if np.isscalar(arr2):
        arr2 = np.full_like(arr1, arr2)

    diff = arr1 - arr2
    if loss_mask is not None:
        diff = diff[loss_mask]

    if diff.size == 0:
        return 0
    
    match function.lower():
        case 'mae':
            return np.mean(np.abs(diff), dtype=np.float32)
        case 'mse':
            return np.mean(np.square(diff))
        case 'rmse':
            return np.sqrt(np.mean(np.square(diff)))
        case _:
            raise ValueError('Unknown loss function')
        
def tree_subtract(tree, sample) -> None:

    ######### to update
    """Deterministically use node's estimation to subtract from 
    section of sample.

    Parameters
    ----------
    sample : ndarray
        The entire sample array to subtract from. The section is
        generated automitically using vertex.
    """
    slices = tuple(slice(start, start + length) for start, length in 
                   zip(tree.vertex, tree.partition().shape))

    # This version is ~40% faster, but it's more explicit and less Pythonic.
    # I'm leaving both here in case efficiency is important down the line.
    # If we want to use this, then replace all slices with the faster version.
    #slices = tuple(slice(tree.vertex[i], 
    #                     tree.vertex[i] + tree.partition().shape[i]) 
    #                     for i in range(tree.vertex.size))
    
    subsample = sample[slices]
    mass_to_subtract = np.sum(tree.partition())
    original_mass = np.sum(subsample)

    if mass_to_subtract >= original_mass:
        subsample[...] = 0
        return
    sorted_pixels = np.sort(subsample[subsample > 0], axis=None)

    num_pixels = sorted_pixels.size
    previous = 0
    subtrahend = 0
    for i in range(num_pixels):
        subtrahend += (sorted_pixels[i] - previous) * (num_pixels - i)
        previous = sorted_pixels[i]
        if subtrahend > mass_to_subtract:
            i -= 1
            break
    if i != -1:
        subsample -= sorted_pixels[i]
        subsample[subsample < 0] = 0
        subtracted = original_mass - np.sum(subsample)
        mass_to_subtract -= subtracted
    nonzero = np.count_nonzero(subsample)
    if nonzero:
        subsample -= mass_to_subtract / nonzero
        subsample[subsample < 0] = 0
        

    
        
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

