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

from node import Node
from calc import safe_divide,my_round,regression_loss
import pickle

#TODO
'''
class cleantree:
    def __init__(self)->None:
        self.root=None
    def copy_kltree(self,kdtree)->None:
        self.root=kdtree.root
        
'''




class KDTree:
    def __init__(self, data, mask, debug=False) -> None:
        """Create a KDTree.

        Parameters
        ----------
        data : ndarray
            Data used to construct tree.
        mask : ndarray
            Mask to represent valid sections of data.
        debug : bool, optional
            Print runtime info, by default False

        Raises
        ------
        ValueError
            Whether there will be floating point errors with data.
        """

        # TODO: this is a really inefficient hotfix by converting the 
        # ENTIRE dataset to a higher precision float instead of using 
        # higher precision floats for each numpy function. I don't know
        # an elegant way to do that though.
        max_used_value = np.sum(np.square(data), dtype=np.float64)
        possible_dtypes = [np.float16, np.float32, np.float64] 
        # TODO: add np.float128, but this may not work for some 
        # architectures and I don't feel like testing this. 99.9% this 
        # will never be needed, but it's fun
        float_precision = None
        for dtype in possible_dtypes:
            if max_used_value <= np.finfo(dtype).max:
                float_precision = dtype
                break
        if float_precision is None:
            raise ValueError('Data values are too large to be processed within np.float64.')
        
        self.root = Node(data.astype(float_precision), mask, 
                         np.zeros(data.ndim, dtype=int))
        self.tensor = self.root.data
        self.mask = self.root.mask
        self.debug = debug
        self.remainpixel=self.root.data.size
    #TODO
    '''
    def save_tree(self,treefile)->None:

        try:
            with open(treefile, 'wb') as f:
                pickle.dump(self, f)
            print(f"Successfully saved KD-tree to {treefile}")
        except Exception as e:
            print(f"Error saving KD-tree: {str(e)}")


    def load_tree(filename: str) -> 'KDTree':
        
        try:
            with open(filename, 'rb') as f:
                tree = pickle.load(f)
            print(f"Successfully loaded KD-tree from {filename}")
            return tree
        except Exception as e:
            print(f"Error loading KD-tree: {str(e)}")
            return None
    '''
        
    # Copy pasted from __build_recurse. There's code duplication, 
    # but I didn't feel like making ANOTHER helper function
    def build_bfs(self, stopping_criteria=None, valid=None, loss_metric='mae', 
                  max_leaves=-1, seed=None) -> None:
        """Build the tree structure using a breadth-first search instead 
        of recursion. This allows trees to be made with an upper bound 
        on the number of leaves.

        Parameters
        ----------
        stopping_criteria : dict, optional
            Dictionary with keys being stopping criteria. Each key maps 
            to its corresponding criterion. Stops splitting whenever any 
            of the stopping criteria are met, so it will continue only 
            if within thresholds for all keys, by default None
        valid : function, optional
            Overrides stopping_criteria with a predefined validation 
            function. This function would otherwise be automatically 
            created using stopping_criteria. Recommended to use 
            stopping_criteria instead of this when using manually. valid 
            is used by other functions which efficiently build their 
            validation functions directly, by default None
        loss_metric : str, optional
            Loss function to minimize when determining split direction, 
            by default 'mae'
        max_leaves : int, optional
            Max number of leaves allowed when building the tree. If max 
            is reached, remaining leaves will be approximately the same 
            size, by default -1
        seed : int, optional
            Seed for when ties are decided randomly, by default None
        """
        if self.debug:
            start = time.perf_counter()

        RNG = np.random.default_rng(seed)

        if valid is None:
            valid = lambda node: all(metric(node, stopping_criteria[metric]) 
                                     for metric in stopping_criteria)

        #valid = (lambda node: all(metric(node, stopping_criteria[metric]) 
        #                           for metric in stopping_criteria)) if valid is None else valid

        node_queue = deque([self.root])
        num_leaves = 1
        while node_queue and num_leaves != max_leaves:
            node = node_queue.popleft()

            if not node.create_children(valid, RNG, loss_metric):
                continue
            
            node_queue.append(node.left)
            node_queue.append(node.right)
            num_leaves += 1

        if self.debug:
            stop = time.perf_counter()
            print(f'build_bfs() execution (s): {stop - start}')

    def build(self, stopping_criteria=None, valid=None, 
              loss_metric='mae', seed=None) -> None:
        """Build tree structure using recursion.

        Parameters
        ----------
        stopping_criteria : dict or None, optional
            Dictionary with keys being stopping criteria. Each key maps 
            to its corresponding criterion. Stops splitting whenever any 
            of the stopping criteria are met, so it will continue only 
            if within thresholds for all keys, by default None
        valid : function, optional
            Overrides stopping_criteria with a predefined validation 
            function. This function would otherwise be automatically 
            created using stopping_criteria. Recommended to use 
            stopping_criteria instead of this when using manually. valid 
            is used by other functions which efficiently build their 
            validation functions directly, by default None
        loss_metric : str, optional
            Loss function to minimize when determining split direction, 
            by default 'mae'
        seed : int, optional
            Seed for when ties are decided randomly, by default None
        """
        if self.debug:
            start = time.perf_counter()

        RNG = np.random.default_rng(seed)
        self.valid = (lambda node: all(metric(node, stopping_criteria[metric]) 
                                      for metric in stopping_criteria)) if valid is None else valid
        self.loss_metric = loss_metric

        self.__build_recurse(self.root, RNG)

        if self.debug:
            stop = time.perf_counter()
            print(f'build_tree() execution (s): {stop - start}')

    def __build_recurse(self, node: Node, rng) -> None:
        """Recursive call to build tree structure."""
        # Base case
        if not node.create_children(self.valid, rng, self.loss_metric):
            return
    
        self.__build_recurse(node.left, rng)
        self.__build_recurse(node.right, rng)

    def estimate(self) -> None:
        """Use tree structure to replace tensor with piece-wise constant 
        estimate.
        """
        if self.debug:
            start = time.perf_counter()

        self.__estimate_recurse(self.root)

        if self.debug:
            stop = time.perf_counter()
            print(f'estimate() execution (s): {stop - start}')

    def __estimate_recurse(self, node: Node) -> None:
        """Recursive call to generate piecewise constant function. 
           Automatically masks off data. TODO: maybe add option to 
           specify what to set masked values to"""
        node.partitioned = True

        if node.is_leaf():
            node.estimate()
            return
        self.__estimate_recurse(node.left)
        self.__estimate_recurse(node.right)

    def num_nodes(self, node=None) -> int:
        """Number of nodes within node's subtree.

        Returns
        -------
        int
            Number of nodes
        """
        if node is None:
            node = self.root

        if node.is_leaf():
            return 1
        if node.left is None:
            return 1 + self.num_nodes(node.right)
        if node.right is None:
            return 1 + self.num_nodes(node.left)
        return 1 + self.num_nodes(node.left) + self.num_nodes(node.right)
    
    def num_leaves(self, node=None) -> int:
        """Number of leaves within node's subtree.

        Returns
        -------
        int
            Number of leaves
        """
        if node is None:
            node = self.root

        if node.is_leaf():
            return 1
        if node.left is None:
            return self.num_leaves(node.right)
        if node.right is None:
            return self.num_leaves(node.left)
        return self.num_leaves(node.left) + self.num_leaves(node.right)
    
    def distance_select_subtract(self, sample) -> np.ndarray:
        """Deterministically use node's estimation to subtract from sample.

        Parameters
        ----------
        sample : ndarray
            Tensor with peaks and background to subtract from.

        Returns
        -------
        ndarray
            Subtracted tensor with isolated peaks.
        """
        if self.debug:
            start = time.perf_counter()

        self.sample = np.where(self.root.mask, np.floor(sample), 0)
        self.__distance_select_subtract_recurse(self.root)

        if self.debug:
            stop = time.perf_counter()
            print(f'constant_subtract() execution (s): {stop - start}')
        return self.sample

    def __distance_select_subtract_recurse(self, node: Node) -> None:
        """Recursive call to subtract a constant from each leaf."""
        if node.is_leaf():
            node.distance_select_subtract(self.sample)
            self.remainpixel -= node.data.size
            #print('remaining region size: %d'%(self.remainpixel))
            return
        
        self.__distance_select_subtract_recurse(node.left)
        self.__distance_select_subtract_recurse(node.right)

    def distance_select_subtract_op(self, sample) -> np.ndarray:
        """Deterministically use node's estimation to subtract from sample.

        Parameters
        ----------
        sample : ndarray
            Tensor with peaks and background to subtract from.

        Returns
        -------
        ndarray
            Subtracted tensor with isolated peaks.
        """
        if self.debug:
            start = time.perf_counter()

        self.sample = np.where(self.root.mask, np.floor(sample), 0)
        self.__distance_select_subtract_recurse_op(self.root)

        if self.debug:
            stop = time.perf_counter()
            print(f'distance_based_subtract() execution (s): {stop - start}')
        return self.sample

    def __distance_select_subtract_recurse_op(self, node: Node) -> None:
        """Recursive call to subtract a constant from each leaf."""
        if node.is_leaf():
            node.distance_select_subtract_op(self.sample)
            self.remainpixel -= node.data.size
            #print('remaining region size: %d'%(self.remainpixel))
            return
        
        self.__distance_select_subtract_recurse_op(node.left)
        self.__distance_select_subtract_recurse_op(node.right)





    def random_select_subtract(self, sample) -> np.ndarray:
        """Deterministically use node's estimation to subtract from sample.

        Parameters
        ----------
        sample : ndarray
            Tensor with peaks and background to subtract from.

        Returns
        -------
        ndarray
            Subtracted tensor with isolated peaks.
        """
        if self.debug:
            start = time.perf_counter()

        self.sample = np.where(self.root.mask, np.floor(sample), 0)
        self.__random_select_subtract_recurse(self.root)

        if self.debug:
            stop = time.perf_counter()
            print(f'constant_subtract() execution (s): {stop - start}')
        return self.sample

    def __random_select_subtract_recurse(self, node: Node) -> None:
        """Recursive call to subtract a constant from each leaf."""
        if node.is_leaf():
            node.random_select_subtract(self.sample)
            return
        
        self.__random_select_subtract_recurse(node.left)
        self.__random_select_subtract_recurse(node.right)


    def constant_subtract(self, sample) -> np.ndarray:
        """Deterministically use node's estimation to subtract from sample.

        Parameters
        ----------
        sample : ndarray
            Tensor with peaks and background to subtract from.

        Returns
        -------
        ndarray
            Subtracted tensor with isolated peaks.
        """
        if self.debug:
            start = time.perf_counter()

        self.sample = np.where(self.root.mask, sample, 0)
        self.__constant_subtract_recurse(self.root)

        if self.debug:
            stop = time.perf_counter()
            print(f'constant_subtract() execution (s): {stop - start}')
        return self.sample

    def __constant_subtract_recurse(self, node: Node) -> None:
        """Recursive call to subtract a constant from each leaf."""
        if node.is_leaf():
            node.constant_subtract(self.sample)
            return
        
        self.__constant_subtract_recurse(node.left)
        self.__constant_subtract_recurse(node.right)

    def random_subtract(self, sample, mass=None, sample_weight=1, 
                        batch_size=1, seed=None) -> np.ndarray:
        """Randomly sample points in sample to subtract following 
        distribution from the estimated tensor. Uses the entire estimate 
        to form a giant probability distribution function for sampling, 
        giving no gaurantee for accurate subtracted counts from each leaf.

        Parameters
        ----------
        sample : ndarray
            Tensor with peaks and background to subtract from.
        mass : int or float, optional
            Total mass or counts to subtract, by default None
        sample_weight : int or float, optional
            How much mass to subtract for each sampled datapoint, 
            by default 1
        batch_size : int, optional
            How many datapoints to sample in each batch 
            (no effect on result only speed), by default 1
        seed : int, optional
            Seed for randomly sampled subtraction, by default None

        Returns
        -------
        ndarray
            Subtracted tensor with isolated peaks.

        Raises
        ------
        ValueError
            If the given sample has less mass than the estimated 
            background, then the output would be completely empty. 
            Instead, error.
        """
        if self.debug:
            start = time.perf_counter()

        RNG = np.random.default_rng(seed)
        # mass is for if you want to subtract a custom mass instead data
        samples_left = np.sum(self.tensor) if mass is None else mass
        if np.sum(sample) < samples_left:
            # TODO: figure out if this should be an error or warning
            # warnings.warn('Sample is not strong enough.')
            raise ValueError('Sample is not strong enough.')
        self.sample = np.where(self.root.mask, sample, 0)
        self.root.random_subtract(self.sample, sample_weight, 
                                   batch_size, RNG, uniform=None)

        if self.debug:
            stop = time.perf_counter()
            print(f'random_subtract() execution (s): {stop - start}')
        return self.sample

    def partition_random_subtract(self, sample, sample_weight=1, batch_size=1, seed=None) -> np.ndarray:
        """Randomly sample from density estimation to subtract from sample.

        Parameters
        ----------
        sample : ndarray
            Tensor with peaks and background to subtract from.
        sample_weight : int, optional
            How much mass to subtract for each sampled datapoint, 
            by default 1
        batch_size : int, optional
            How many datapoints to sample in each batch 
            (no effect on result only speed), by default 1
        seed : int, optional
            Seed for randomly sampled subtraction, by default None

        Returns
        -------
        ndarray
            Subtracted tensor with isolated peaks.
        """
        if self.debug:
            start = time.perf_counter()

        RNG = np.random.default_rng(seed)

        self.sample = np.where(self.root.mask, sample, 0)
        self.__partition_random_subtract_recurse(self.root, sample_weight, batch_size, RNG)

        if self.debug:
            stop = time.perf_counter()
            print(f'partition_random_subtract() execution (s): {stop - start}')
        #del self.sample
        return self.sample

    def __partition_random_subtract_recurse(self, node: Node, sample_weight, batch_size, rng) -> None:
        """Recursive call to randomly subtract from each leaf."""
        if node.is_leaf():
            node.random_subtract(self.sample, sample_weight, batch_size, rng, uniform=True)
            return
        
        self.__partition_random_subtract_recurse(node.left, sample_weight, batch_size, rng)
        self.__partition_random_subtract_recurse(node.right, sample_weight, batch_size, rng)

    def apply_mask(self, mask=None) -> None:
        """Set all elements outside of mask to 0.

        Parameters
        ----------
        mask : ndarray of bool, optional
            Boolean array used. If None, use the already 
            loaded mask, by default None.
        
        Returns
        -------
        None
        """
        used_mask = self.mask if mask is None else mask
        self.tensor[~used_mask] = 0
        
    def generate(self, sample_weight=1, seed=None) -> np.ndarray:
        """Sample entire density distribution to generate a new tensor 
        with approximate mass to input.

        Parameters
        ----------
        sample_weight : int or float, optional
            How much each sampled point from distribution should 
            increment output tensor, by default 1.
        seed : int, optional
            Seed for random samples, by default None


        Returns
        -------
        ndarray
            Output tensor with approximate shape and mass to input.
        """
        if self.debug:
            start = time.perf_counter()

        RNG = np.random.default_rng(seed)
        out = np.zeros_like(self.partition())
        self.root.generate(sample_weight, out, RNG)

        if self.debug:
            stop = time.perf_counter()
            print(f'generate() execution (s): {stop - start}')
        return out
    
    def partition_generate(self, sample_weight=1, seed=None) -> np.ndarray:
        """Sample density distributions of each leaf independently to 
        generate a new tensor with approximate mass to input.

        Parameters
        ----------
        sample_weight : int or float, optional
            How much each sampled point from distribution should 
            increment output tensor, by default 1.
        seed : int, optional
            Seed for random samples, by default None

        Returns
        -------
        ndarray
            Output tensor with approximate shape and mass to input.
        """
        if self.debug:
            start = time.perf_counter()

        RNG = np.random.default_rng(seed)
        out = np.zeros_like(self.tensor)
        self.__partition_generate_recurse(self.root, sample_weight, RNG, out)

        if self.debug:
            stop = time.perf_counter()
            print(f'partition_generate() execution (s): {stop - start}')
        return out

    def __partition_generate_recurse(self, node: Node, sample_weight, rng, out) -> None:
        """Recursive call to randomly generate leaves that follow the estimate."""
        if node.is_leaf():
            node.generate(sample_weight, out, rng)
            return

        self.__partition_generate_recurse(node.left, sample_weight, rng, out)
        self.__partition_generate_recurse(node.right, sample_weight, rng, out)

# Seems to break once called multiple times, TODO fix this
    def visualize_partitions(self, seed=None) -> np.ndarray:
        """Visualize how the data is partitioned into each node by 
        randomly coloring the edge voxels.

        Parameters
        ----------
        seed : int, optional
            Seed for random edge coloring, by default None

        Returns
        -------
        ndarray
            Tensor with recolored edges.
        """
        if self.debug:
            start = time.perf_counter()

        RNG = np.random.default_rng(seed)
        #backup = copy.deepcopy(self.partition)
        #plt.imshow(backup)
        backup = self.tensor.copy()
        self.__visualize_partitions_recurse(self.root, np.max(self.tensor), RNG)
        #output = copy.deepcopy(self.partition)
        output = self.tensor.copy()
        self.root.data = backup
        
        if self.debug:
            stop = time.perf_counter()
            print(f'visualize_partition() execution (s): {stop - start}')

        return output

    def __visualize_partitions_recurse(self, node: Node, max, rng) -> None:
        """Recursive call to randomly color leaf borders."""
        if node.is_leaf():
            node.visualize_partition(max, rng)
            return
        
        self.__visualize_partitions_recurse(node.left, max, rng)
        self.__visualize_partitions_recurse(node.right, max, rng)


