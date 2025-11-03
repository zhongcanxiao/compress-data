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

from kdtree import KDTree
from node import Node
from calc import regression_loss

from multiprocessing import Process,Pool
# TODO: not at all done

GLOBAL_SEED=42
'''
def tune_gd(self, metric, init, lr, tolerance, max_iterations, loss_metric='mae', round=False, seed=None):
    """Perform gradient descent to find optimal stopping criteria.

    Parameters
    ----------
    metric : _type_
        _description_
    init : _type_
        _description_
    lr : _type_
        _description_
    tolerance : _type_
        _description_
    max_iterations : _type_
        _description_
    loss_metric : str, optional
        _description_, by default 'mae'
    round : bool, optional
        _description_, by default False
    seed : _type_, optional
        _description_, by default None
    """
    RNG = np.random.default_rng(seed=seed)

    # TODO: Figure out how to control validation
    validation_percent = 0.85
    validation_mask = RNG.choice(a=[True, False], size=self.tensor.shape, p=[validation_percent, 1 - validation_percent])
    print(validation_mask.shape)
    #train = self.tensor[validation_mask]
    train_mask = np.logical_and(self.root.mask, validation_mask)
    #validate = self.tensor[~validation_mask]
    validate_mask = np.logical_and(self.root.mask, ~validation_mask)

    tree = KDTree(self.tensor.copy(), train_mask)
    # TODO: this function call is outdated
    tree.build(metric, init, loss_metric, seed)
    tree.estimate()
    
    x = init
    loss_metric = regression_loss(self.tensor, tree.tensor, 
                           validate_mask, loss_metric)
    new_x = x - lr

    for i in range(max_iterations):
        tree = KDTree(self.tensor.copy(), train_mask)
        # TODO: this function call is outdated
        tree.build(metric, init, loss_metric, seed)
        tree.estimate()
        new_loss = regression_loss(self.tensor, tree.tensor, 
                                   validate_mask, loss_metric)
        gradient = (new_loss - loss_metric) / (new_x - x)

        x = new_x
        loss_metric = new_loss

        new_x = x - lr * gradient

        if abs(new_x - x) <= tolerance:
            break

        print(x, loss_metric)
'''

'''
    """_summary_

    Parameters
    ----------
    validate : ndarray
        Tensor used as the validation set.
    mask : ndarray
        Mask for the validation set.
    possibilities/renamed to hyper_parameters : dict
        Dictionary of which splitting criteria to use, which are 
        stored as keys. They map to an iterable of thresholds to try 
        for that metric. Every possible combination of thresholds 
        for each metric are tested.
    scalars : iterable
        Iterable of possible coefficients for the validation tensor.
    repeat : int, optional
        Number of times a tree will be generated and tested for each
        combination, by default 1
    loss_metric : str, optional
        Loss metric to use when evaluating validation tensor, by 
        default 'mae'
    seed : int, optional
        Seed for building trees. Seeding makes trees grow with more 
        consistent structure, reducing noise, by default None

    """
'''
# Gridsearch
def bg_remove (empty,  mask,sample,holes,metrics,threshold_combination,scalars,loss_metric,seed):
####################################################################################
########################  checck  ##################################################
####################################################################################
    valid = lambda node: all(metric(node, threshold) 
                             for metric, threshold in 
                             zip(metrics, threshold_combination))
####################################################################################
########################  checck  ##################################################
####################################################################################
    tree = KDTree(empty.copy(), mask)

    #tree = KDTree(tree0.tensor.copy(), tree0.mask)
    tree.apply_mask()
    tree.build(valid=valid, loss_metric=loss_metric, seed=seed )
    tree.estimate()
    loss=[]
    for scalar in scalars:
        error = regression_loss(tree.tensor, scalar * sample, loss_mask= holes , function=loss_metric)
        #losses[counter] = (threshold_combination + (scalar, error))
        #counter += 1
        loss.append(list(threshold_combination) + [scalar, error])
 
    #print(loss)
    return loss

def tune_gs(empty,  mask,sample,holes, hyper_parameters, scalars , loss_metric='mae', seed=None,n_process=1):
    np.set_printoptions(legacy='1.25') # ommit showing type information
    metrics = tuple(hyper_parameters)
    thresholds = tuple(hyper_parameters.values())
    threshold_combinations = tuple(itertools.product(*thresholds))
    num_trees = len(threshold_combinations) 

    #losses = np.zeros((num_trees * len(scalars), len(metrics) + 2))
    losses = []
    


    input_list=[]

    for threshold_combination in threshold_combinations:
        input_list.append( (empty,  mask,sample,holes,metrics,threshold_combination,scalars,loss_metric,seed))
        #print(loss)
        #losses.append(loss)

    with Pool(processes=n_process) as pool:
        losses = pool.starmap(bg_remove, input_list)


    np.save('losses', losses)



def br_run(nc_empty,  nc_sample,mask , n_count,n_pix,var_th,red_scale_list,seed=None,n_process=1):
    print('start building tree with n_count:{}, n_pix:{}, var:{}'.format(n_count,n_pix,var_th),flush=True)
    tree0 = KDTree(nc_empty.copy(), mask=mask, debug=True)
    tree0.apply_mask()
    tree0.build({Node.valid_num_counts: n_count,Node.valid_num_masked:n_pix, Node.valid_variance: var_th}, seed=GLOBAL_SEED)
    tree0.estimate()
    tree0.apply_mask()
    br0=[None]*len(red_scale_list)
    for i  in range(len(red_scale_list)):
        red_scale=red_scale_list[i]
        br_sample=nc_sample.copy()*red_scale
        print('start subtraction with scaling factor:{}'.format(red_scale),flush=True)
        
        br0[i]=tree0.distance_select_subtract_op(br_sample)
    return tree0,np.array(br0)