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

from calc import safe_divide,my_round,regression_loss

class Node:
    def __init__(self, data, mask, vertex) -> None:
        """Create a node.

        Parameters
        ----------
        data : ndarray
            Data used to construct tree. Generated nodes' data is a view 
            of a section of the tree's overall tensor data.
        mask : ndarray
            Mask that represents valid sections of data.
        vertex : iterable
            Index of data.flatten()[0] within the tree's overall tensor.
            This maps (index of node's data) + vertex = (index of tree tensor).
        """
        self.left = None
        self.right = None
        self.data = data
        self.mask = mask
        self.vertex = vertex
        self.partitioned = False

    def is_leaf(self) -> bool:
        """Whether current node is a leaf.

        Returns
        -------
        bool
            Whether current node is a leaf.
        """
        return self.left is None and self.right is None

    def estimate(self) -> None:
        """Set all data to the mean using mask. Usually only call on 
        leaf nodes.
        """
        if np.count_nonzero(self.mask) == 0:
            self.data[...] = np.zeros_like(self.data)
        else:
            self.data[...] = np.full_like(self.data, np.mean(self.data[self.mask]))
            #self.data[~self.mask] = 0

    def partition(self) -> np.ndarray:
        """Return partitioned data.

        Returns
        -------
        ndarray
            Partitioned data

        Raises
        ------
        ValueError
            If data is not partitioned.
        """
        if not self.partitioned:
            raise ValueError('Not partitioned yet.')
        return self.data
    
    def random_select_subtract_0(self, sample) -> None:
        """
        """
        slices = tuple(slice(start, start + length) for start, length in 
                       zip(self.vertex, self.partition().shape))
        subsample = sample[slices]
        mass_to_subtract = int(np.sum(self.partition()))
        original_mass = np.sum(subsample)

        if mass_to_subtract >= original_mass:
            subsample[...] = 0
            return
                
        # Get indices of non-zero elements
        nonzero_indices = np.nonzero(subsample)
        num_nonzero = len(nonzero_indices[0])
        
        # Create array of positions of non-zero elements
        positions = list(zip(*nonzero_indices))
        
        # Get current values at these positions
        values = subsample[nonzero_indices]
        
        # Initialize remaining amount to subtract
        remaining = mass_to_subtract
        print('remaining',remaining,mass_to_subtract)

        ############## 
        while remaining > 0:
            # Randomly select position
            available_positions = [(i, pos) for i, pos in enumerate(positions) 
                                if subsample[pos] > 0]
            if not available_positions:
                break
                
            idx, pos = available_positions[np.random.randint(len(available_positions))]
            
            # Calculate maximum possible subtraction at this position
            max_subtract = int(np.floor(min(subsample[pos], remaining)))
            
            if max_subtract > 0:
                # Randomly choose amount to subtract
                subtract = np.random.randint(1, max_subtract + 1)
                
                # Perform subtraction
                subsample[pos] -= subtract
                remaining -= subtract
            print('remaining',remaining,mass_to_subtract,pos,subtract)
        
        
        return    
    
    def random_select_subtract(self, sample) -> None:
        """
        """
        slices = tuple(slice(start, start + length) for start, length in 
                       zip(self.vertex, self.partition().shape))
        
        subsample = sample[slices]

        mass_to_subtract = int(np.sum(self.partition()))
        original_mass = np.sum(subsample)
        #print(original_mass,mass_to_subtract)
        if mass_to_subtract >= original_mass:
            subsample[...] = 0
            return
                
        # Get indices of non-zero elements
        nonzero_indices = np.nonzero(subsample)
        num_nonzero = len(nonzero_indices[0])
        
        # Create array of positions of non-zero elements
        #positions = list(zip(*nonzero_indices))
        
        # Get current values at these positions
        values = subsample[nonzero_indices]
        
        # Initialize remaining amount to subtract
        remaining = mass_to_subtract
        #print('remaining',remaining,mass_to_subtract)

        
        subtractions = np.zeros(num_nonzero, dtype=int)
        #print(np.sum(subsample),np.sum(subtractions))
        while(remaining>0):
           # print(remaining)
            for i in range(num_nonzero ):
                # For each position except last, generate random number
                # between 0 and remaining value that can still ensure
                # positive result after subtraction
                max_possible = min(remaining, subsample[nonzero_indices[0][i], 
                                                nonzero_indices[1][i], 
                                                nonzero_indices[2][i]]-subtractions[i])
                #print(max_possible)
                if max_possible > 0:
                    sub_rand = np.random.randint(0, max_possible + 1)
                    subtractions[i] += sub_rand
                    remaining -= sub_rand
            
          #  # Assign remaining value to last position
          #  if remaining > 0 and remaining <= subsample[nonzero_indices[0][-1], 
          #                                      nonzero_indices[1][-1], 
          #                                      nonzero_indices[2][-1]]-subtractions[-1]:
          #      subtractions[-1] = remaining
        
            

        # Subtract values
        #print(np.sum(subsample),np.sum(subtractions),subtractions,values)
        for idx, (i, j, k) in enumerate(zip(*nonzero_indices)):
            subsample[i, j, k] -= subtractions[idx]
        #print(np.sum(subsample),np.sum(subtractions))
        return    
    
         
    
    def distance_select_subtract_1(self, sample) -> None:
        """
        """

        def isolate_pixel_island_size(matrix):
            #positive_positions=[tuple i for i in np.where(matrix>0) ]
            #zero_positions=[tuple i for i np.where(matrix==0)]
            positive_positions=np.array(np.where(matrix>0)).T

            zero_positions=np.array(np.where(matrix==0)).T

            #print(np.array(positive_positions).shape,np.array(zero_positions).shape,matrix.shape)
            #print(matrix)
            #print(positive_positions,zero_positions)

            if len(zero_positions)==0: return np.zeros_like(matrix)+1
            island_size=np.zeros_like(matrix)
            island_size[np.where(matrix>0)] +=1
            for idx0 in zero_positions:
                dis_nearest=np.inf
                idx_nearest=[]
                
                for idx1 in positive_positions:
                    #print (idx0,idx1)
                    dis_current=np.linalg.norm(idx1-idx0)
                    if dis_current<dis_nearest:
                        idx_nearest=idx1
                        dis_nearest=dis_current
                island_size[tuple(idx_nearest)] +=1
            if np.sum(island_size)-np.size(matrix)!=0: exit ('zero island set up error')
            
            return island_size
        
        slices = tuple(slice(start, start + length) for start, length in 
                       zip(self.vertex, self.partition().shape))
        
        subsample = sample[slices]
        #print('node size %d start'%np.size(subsample))
        mass_to_subtract = int(np.sum(self.partition()))
        original_mass = np.sum(subsample)
        #print(original_mass,mass_to_subtract)
        if mass_to_subtract >= original_mass:
            subsample[...] = 0
            return

        remaining = mass_to_subtract                
        generate_island=True
     
        while(remaining>0):
            #print('remaining: ',remaining)
            if generate_island:
                probabilities=isolate_pixel_island_size(subsample)
                nonzero_indices = np.array(np.where(subsample>0)).T
                #print('probabilities[np.where(probabilities>0)]',probabilities[np.where(probabilities>0)])
                #print('np.where(probabilities>0)',np.where(probabilities>0))
                indices_prob=probabilities[np.where(probabilities>0)].flatten()/np.sum(probabilities)
                if len(nonzero_indices) != len(indices_prob):
                    print('indices_prob,nonzero_indices,',indices_prob,nonzero_indices)
                    print('shape indices_prob,nonzero_indices,',indices_prob.shape,nonzero_indices.shape)
                    print('probabilities.shape,subsample.shape',probabilities.shape,subsample.shape)
                    print('nonzero pos probabilities.shape,subsample.shape',np.where(probabilities>0),np.where(subsample>0))
                                
            go_to_zero_prob=np.array([np.power(indices_prob[i],subsample[nonzero_indices[i]] ) for i in range(len(indices_prob))])
            first_to_zero_prob=go_to_zero_prob/np.sum(go_to_zero_prob)
            idx_choose=np.random.choice(len(nonzero_indices),p=first_to_zero_prob)

            #if max_possible ==0: exit('error: subtracting zero value pixel')
            sub_rand_total = np.random.negative_binomial(subsample[nonzero_indices[idx_choose]],first_to_zero_prob[idx_choose])
            subsample[tuple(nonzero_indices[idx_choose])] -= sub_rand
            if subsample[tuple(nonzero_indices[idx_choose])] >0: 
                generate_island=False
            else:
                generate_island=True
            remaining -= sub_rand
            
        #print('node size %d done'%np.size(subsample))
        return    
    
        
    
    def distance_select_subtract(self, sample) -> None:
        """
        """

        def isolate_pixel_island_size(matrix):
            #positive_positions=[tuple i for i in np.where(matrix>0) ]
            #zero_positions=[tuple i for i np.where(matrix==0)]
            positive_positions=np.array(np.where(matrix>0)).T

            zero_positions=np.array(np.where(matrix==0)).T

            #print(np.array(positive_positions).shape,np.array(zero_positions).shape,matrix.shape)
            #print(matrix)
            #print(positive_positions,zero_positions)

            if len(zero_positions)==0: return np.zeros_like(matrix)+1
            island_size=np.zeros_like(matrix)
            island_size[np.where(matrix>0)] +=1
            for idx0 in zero_positions:
                dis_nearest=np.inf
                idx_nearest=[]
                
                for idx1 in positive_positions:
                    #print (idx0,idx1)
                    dis_current=np.linalg.norm(idx1-idx0)
                    if dis_current<dis_nearest:
                        idx_nearest=idx1
                        dis_nearest=dis_current
                island_size[tuple(idx_nearest)] +=1
            if np.sum(island_size)-np.size(matrix)!=0: exit ('zero island set up error')
            
            return island_size
        
        slices = tuple(slice(start, start + length) for start, length in 
                       zip(self.vertex, self.partition().shape))
        
        subsample = sample[slices]
        #print('node size %d start'%np.size(subsample))
        mass_to_subtract = int(np.sum(self.partition()))
        original_mass = np.sum(subsample)
        #print(original_mass,mass_to_subtract)
        if mass_to_subtract >= original_mass:
            subsample[...] = 0
            return

        remaining = mass_to_subtract                
        generate_island=True
     
        while(remaining>0):
            #print('remaining: ',remaining)
            if generate_island:
                probabilities=isolate_pixel_island_size(subsample)
                nonzero_indices = np.array(np.where(subsample>0)).T
                ####################################################################
                ####################################################################
                ####################################################################
                #print('probabilities[np.where(probabilities>0)]',probabilities[np.where(probabilities>0)])
                #print('np.where(probabilities>0)',np.where(probabilities>0))
                indices_prob=probabilities[np.where(probabilities>0)].flatten()/np.sum(probabilities)
                if len(nonzero_indices) != len(indices_prob):
                    print('indices_prob,nonzero_indices,',indices_prob,nonzero_indices)
                    print('shape indices_prob,nonzero_indices,',indices_prob.shape,nonzero_indices.shape)
                    print('probabilities.shape,subsample.shape',probabilities.shape,subsample.shape)
                    print('nonzero pos probabilities.shape,subsample.shape',np.where(probabilities>0),np.where(subsample>0))
                                
            idx_choose=np.random.choice(len(nonzero_indices),p=indices_prob)
            
            max_possible = min(remaining, subsample[tuple(nonzero_indices[idx_choose])])
                
            #if max_possible ==0: exit('error: subtracting zero value pixel')
            sub_rand = np.random.randint(0, max_possible + 1)
            subsample[tuple(nonzero_indices[idx_choose])] -= sub_rand
            if subsample[tuple(nonzero_indices[idx_choose])] >0: 
                generate_island=False
            else:
                generate_island=True
            remaining -= sub_rand
            
        #print('node size %d done'%np.size(subsample))
        return    
    
     
    def distance_select_subtract_op(self, sample) -> None:
        """
        """

        
        slices = tuple(slice(start, start + length) for start, length in 
                       zip(self.vertex, self.partition().shape))
        
        subsample = sample[slices]
        #submask=self.mask[slices]
        submask=self.mask
        #print('1',submask.shape)
        #print('2',subsample.shape)
        #print('3',slices)
        #print('4',self.partition().shape)
       # print('5',self.partition())
        #print('node size %d start'%np.size(subsample))
        mass_to_subtract = int(np.sum(self.partition()))
        original_mass = np.sum(subsample)
        #print(original_mass,mass_to_subtract)
        if mass_to_subtract >= original_mass:
            subsample[...] = 0
            return

        remaining = mass_to_subtract                
     
        [lenx,leny,lenz]=subsample.shape
        subsample_valid=subsample[submask]
        num_pixel_valid=np.sum(submask)
        valid_pixel_idx_list=[]
        for i in range(np.size(subsample)):
            idx=np.unravel_index(i,(lenx,leny,lenz))
            if submask[idx]:
                valid_pixel_idx_list.append(idx)

        all_idx_x,all_idx_y,all_idx_z=np.meshgrid(np.arange(lenx),np.arange(leny),np.arange(lenz),indexing='ij')

        idx_choice_list=np.random.randint(0,len(valid_pixel_idx_list),remaining)
        nonzero_list=np.zeros_like(subsample,dtype=int)
        nonzero_list[subsample==0] =1000000000
        
        
        #        nonzero_indices = np.array(np.where(subsample>0)).T
        for idx_choice in idx_choice_list:
            #idx_x=int(np.floor(idx_choice_list[i]/(leny*lenz)))
            #idx_y=int(np.floor((idx_choice_list[i]-idx_x*leny*lenz)/lenz))
            #idx_z=int(np.floor(idx_choice_list[i]-idx_x*leny*lenz-idx_y*lenz))
            #idx_x,idx_y,idx_z=np.unravel_index(idx_choice_list[i],(lenx,leny,lenz))
            idx_x,idx_y,idx_z=valid_pixel_idx_list[idx_choice]
            if subsample[idx_x,idx_y,idx_z]>1:
                subsample[idx_x,idx_y,idx_z] -=1
                if subsample[idx_x,idx_y,idx_z]==0:
                    nonzero_list[idx_x,idx_y,idx_z]=1000000000
            else:
                idx_nearest_x=np.inf
                idx_nearest_y=np.inf
                idx_nearest_z=np.inf
                dis_nearest=np.inf

                pos_idx=np.array([[idx_x,idx_y,idx_z]])
                dis=(idx_x-all_idx_x)**2+(idx_y-all_idx_y)**2+(idx_z-all_idx_z)**2
                #dis_list=pos_idx-all_idx
                dis = dis + nonzero_list
                
                idx_min_flat=np.argmin(dis)
                idx_min=np.unravel_index(idx_min_flat,(lenx,leny,lenz))
                subsample[idx_min] -=1
                if subsample[idx_min]==0:
                    #print('reduce to 0 check',idx_min)
                    nonzero_list[idx_min]=1000000000


                #for idx_dis_x in range(lenx):
                #    for idx_dis_y in range(leny):
                #        for idx_dis_z in range(lenz):
                #            dis_current=np.linalg.norm(subsample[idx_x,idx_y,idx_z]-subsample[idx_dis_x,idx_dis_y,idx_dis_z])
                #            if dis_current
                            


        #while(remaining>0):
        #    #print('remaining: ',remaining)
        #    if generate_island:
        #        probabilities=isolate_pixel_island_size(subsample)
        #        nonzero_indices = np.array(np.where(subsample>0)).T
        #        ####################################################################
        #        ####################################################################
        #        ####################################################################
        #        #print('probabilities[np.where(probabilities>0)]',probabilities[np.where(probabilities>0)])
        #        #print('np.where(probabilities>0)',np.where(probabilities>0))
        #        indices_prob=probabilities[np.where(probabilities>0)].flatten()/np.sum(probabilities)
        #        if len(nonzero_indices) != len(indices_prob):
        #            print('indices_prob,nonzero_indices,',indices_prob,nonzero_indices)
        #            print('shape indices_prob,nonzero_indices,',indices_prob.shape,nonzero_indices.shape)
        #            print('probabilities.shape,subsample.shape',probabilities.shape,subsample.shape)
        #            print('nonzero pos probabilities.shape,subsample.shape',np.where(probabilities>0),np.where(subsample>0))
        #                        
        #    idx_choose=np.random.choice(len(nonzero_indices),p=indices_prob)
        #    
        #    max_possible = min(remaining, subsample[tuple(nonzero_indices[idx_choose])])
        #        
        #    #if max_possible ==0: exit('error: subtracting zero value pixel')
        #    sub_rand = np.random.randint(0, max_possible + 1)
        #    subsample[tuple(nonzero_indices[idx_choose])] -= sub_rand
        #    if subsample[tuple(nonzero_indices[idx_choose])] >0: 
        #        generate_island=False
        #    else:
        #        generate_island=True
        #    remaining -= sub_rand
            
        #print('node size %d done'%np.size(subsample))
        return    
    
        

    


    def constant_subtract(self, sample) -> None:
        """Deterministically use node's estimation to subtract from 
        section of sample.

        Parameters
        ----------
        sample : ndarray
            The entire sample array to subtract from. The section is
            generated automitically using vertex.
        """
        slices = tuple(slice(start, start + length) for start, length in 
                       zip(self.vertex, self.partition().shape))
    
        # This version is ~40% faster, but it's more explicit and less Pythonic.
        # I'm leaving both here in case efficiency is important down the line.
        # If we want to use this, then replace all slices with the faster version.
        #slices = tuple(slice(self.vertex[i], 
        #                     self.vertex[i] + self.partition().shape[i]) 
        #                     for i in range(self.vertex.size))
        
        subsample = sample[slices]
        mass_to_subtract = np.sum(self.partition())
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
        return    
    
    # TODO: experiment and see what the fastest batch_size is and why
    # using that maybe also add adaptive batch_size given the size of the node
    def random_subtract(self, sample, sample_weight, batch_size, rng) -> None:
        """Randomly sample points in sample to subtract following 
        distribution from the estimated tensor.

        Parameters
        ----------
        sample : ndarray
            The entire sample array to subtract from. The section is
            generated automitically using vertex.
        sample_weight : int or float
            How much each randomly sampled point should subtract from sample.
        batch_size : int
            Number of points of sample in each batch. Doesn't affect
            result, but can significantly improve efficiency. 
        rng : Generator or RandomState
            Random number generator used to sample points to subtract.
        """
        mass_to_subtract = my_round(np.sum(self.partition()), sample_weight)
        
        slices = tuple(slice(start, start + length) for start, length in 
                       zip(self.vertex, self.partition().shape))
        
        subsample = sample[slices].flatten()
        
        if self.is_leaf():
            min_value = min(np.min(subsample), self.partition().flat[0])
            mass_to_subtract -= min_value * self.partition().size
            subsample -= min_value
            
        sample_mass = np.sum(subsample)
        idxs = np.arange(self.partition().size)
        
        diff = subsample - self.partition().flatten()
        diff -= np.min(diff) - 1
        if isinstance(diff.dtype, np.integer):
            print('not integer array')
            diff = np.array(diff, dtype=float)
        probs = np.reciprocal(diff)
        
        #probs = np.full(node.partition().size, 1) if uniform else node.partition().flatten()
        probs[subsample.flatten() == 0] = 0
        # simple (unnecessary) optimization :P TODO: check if 
        # count_nonzero would be better
        probs_sum = np.sum(probs)
        # TODO: test if 0
        while mass_to_subtract > 0.01 and sample_mass > 0.01:
            try: norm_probs = probs / np.sum(probs)
            except FloatingPointError: 
                norm_probs = safe_divide(probs, np.sum(probs), norm=True)
            samples = rng.choice(idxs, size=batch_size, p=norm_probs)
            
            for idx in samples:
                if sample_weight < subsample[idx]:
                    subsample[idx]   -= sample_weight
                    sample_mass      -= sample_weight
                    mass_to_subtract -= sample_weight
                else:
                    sample_mass      -= subsample[idx]
                    mass_to_subtract -= subsample[idx]
                    probs_sum        -= probs[idx]
                    probs[idx]        = 0
                    subsample[idx]    = 0
                if mass_to_subtract <= 0 or sample_mass <= 0: break

        sample[slices] = subsample.reshape(self.data.shape)

    def generate(self, sample_weight, out, rng) -> None:
        """Randomly sample points following distribution from estimation
        to create section of a new tensor.

        Parameters
        ----------
        sample_weight : int or float
            How much each randomly sampled point should add.
        out : ndarray
            The entire output array to add to. The section is
            generated automitically using vertex. 
        rng : Generator or RandomState
            Random number generator used to sample points.
        """
        #print(self.is_leaf())
        gen = np.zeros(self.partition().size)
        indices = np.arange(self.partition().size)
        flat = self.partition().flatten()
        sum = np.sum(self.partition())
        slices = tuple(slice(start, start + length) for start, length in 
                       zip(self.vertex, self.partition().shape))
        #print(sum)
        if np.max(flat) == 0:
            out[slices] = np.zeros_like(self.partition())
            return
        #threshold = np.finfo(np.float64).tiny * sum
        probs = safe_divide(flat, sum, norm=True)
        #np.divide(flat, sum, out=probs, where=(flat > threshold))
        num_samples = round(sum / sample_weight)
        #print(np.sum(probs))
        choices = rng.choice(indices, size=num_samples, p=probs)
        for choice in choices:
            gen[choice] += sample_weight
        gen = gen.reshape(self.data.shape)

        out[slices] = gen

    def visualize_partition(self, max, rng) -> None:
        """Visualize how the data is partitioned into each node by 
        randomly coloring the edge voxels.

        Parameters
        ----------
        max : int or float
            Max possible to randomly set the borders to. Uniformly
            choose from [0, max].
        rng : Generator or RandomState
            Random number generator used set border value.
        """
        # TODO: this might not work in 3D, but I'm not sure
        border = rng.random() * max
        slices = [slice(None)] * self.partition().ndim
        for axis in range(self.partition().ndim):
            slices_first = slices.copy()
            slices_last = slices.copy()

            slices_first[axis] = 0
            slices_last[axis] = -1

            self.partition()[tuple(slices_first)] = border
            self.partition()[tuple(slices_last)] = border

    def create_children(self, valid, rng, loss_metric) -> bool:
        """Check if node is valid to split. If so, split data amongst 2
        children.

        Parameters
        ----------
        valid : function
            Validation function to determine whether stopping criteria
            are met.
        rng : Generator or RandomState
            Random number generator used to randomly split when there is 
            a tie for best split direction.
        loss_metric : str
            Loss function to minimize when determining best split direction.

        Returns
        -------
        bool
            True if successfully split data and created children. 
            False otherwise.
        """
        if not valid(self):
            return False
        
        # Analyze potential split directions
        splits = [None] * self.data.ndim 
        # Use a list because the different 
        # splits have different dimensions
        masks = [None] * self.data.ndim
        losses = np.full(self.data.ndim, np.inf)
        valid_split = False
        for i in range(self.data.ndim):
            if self.data.shape[i] == 1: 
                # Basically < 2, but 1 is the only option
                continue
            valid_split = True
            splits[i] = np.array_split(self.data, 2, axis=i)
            masks[i] = np.array_split(self.mask, 2, axis=i)
            losses[i] = (regression_loss(splits[i][0], np.mean(splits[i][0]), 
                                            masks[i][0], loss_metric)
                        + regression_loss(splits[i][1], np.mean(splits[i][1]), 
                                            masks[i][1], loss_metric))
            #losses[i] = np.abs(np.mean(splits[i][0][masks[i][0]]) - np.mean(splits[i][1][masks[i][1]]))
        if not valid_split:
            # No valid split directions
            return False
        # Choose direction with max difference 
        # in mean (randomly choose ties)
        split_dim = rng.choice(np.flatnonzero(losses == np.min(losses)))
        # Split subarrays
        #masks = np.array_split(node.mask, 2, axis=split_dim)
        other_vertex = self.vertex.copy()
        other_vertex[split_dim] += math.ceil(self.data.shape[split_dim] / 2)
        
        self.left = Node(splits[split_dim][0], masks[split_dim][0], 
                            vertex=self.vertex)
        # Maybe use node.vertex.copy(), but I don't think 
        # using the same reference will cause issues
        self.right = Node(splits[split_dim][1], masks[split_dim][1], 
                            vertex=other_vertex)
        return True



    def valid_num_nonzero(self, threshold) -> bool:
        """Validation function for whether node has enough nonzero 
        voxels to split.

        Parameters
        ----------
        threshold : int or float
            Minimum number of nonzero voxels within the node to split.

        Returns
        -------
        bool
            Whether node is valid to split.
        """
        return np.count_nonzero(self.data) >= threshold
    
    def valid_num_masked(self, threshold) -> bool:
        """Validation function for whether node has enough voxels within 
        mask to split.

        Parameters
        ----------
        threshold : int or float
            Minimum number of masked voxels within the node to split.

        Returns
        -------
        bool
            Whether node is valid to split.
        """
        return np.count_nonzero(self.mask) >= threshold

    def valid_num_masked_nonzero(self, threshold) -> bool:
        """Validation function for whether node has enough nonzero 
        voxels within mask to split.
        
        Parameters
        ----------
        threshold : int or float
            Minimum number of masked nonzero voxels within the node to split.

        Returns
        -------
        bool
            Whether node is valid to split.
        """
        return np.count_nonzero(self.data[self.mask]) >= threshold

    def valid_percent_nonzero(self, threshold) -> bool:
        """Validation function for whether high enough percent of node 
        is nonzero to split.

        Parameters
        ----------
        threshold : float
            Minimum percent of nonzero voxels within the node to split.

        Returns
        -------
        bool
            Whether node is valid to split.
        """
        return np.count_nonzero(self.data) >= threshold * self.data.size
    
    def valid_percent_masked(self, threshold) -> bool:
        """Validation function for whether high enough percent of node 
        is within mask to split.

        Parameters
        ----------
        threshold : float
            Minimum percent of masked voxels within the node to split.

        Returns
        -------
        bool
            Whether node is valid to split.
        """
        return np.count_nonzero(self.mask) >= threshold * self.data.size
    
    def valid_percent_masked_nonzero(self, threshold) -> bool:
        """Validation function for whether high enough percent of node 
        is nonzero and within mask to split.

        Parameters
        ----------
        threshold : float
            Minimum percent of masked nonzero voxels within the node to split.

        Returns
        -------
        bool
            Whether node is valid to split.
        """
        return (np.count_nonzero(self.data[self.mask])
                >= threshold * self.data.size)
    
    def valid_num_counts(self, threshold) -> bool:
        """Validation function for whether node has enough counts to split.

        Parameters
        ----------
        threshold : int or float
            Minimum sum of node's voxels to split.

        Returns
        -------
        bool
            Whether node is valid to split.
        """
        return np.sum(self.data[self.mask]) >= threshold

    def valid_variance(self, threshold) -> bool:
        """Validation function for whether node has high enough variance 
        to split.

        Parameters
        ----------
        threshold : int or float
            Minimum variance of node's voxels to split.

        Returns
        -------
        bool
            Whether node is valid to split.
        """
        return (np.count_nonzero(self.mask)
                and np.var(self.data[self.mask]) >= threshold)
    
    def valid_variance_mean(self, threshold) -> bool:
        """Validation function for whether node's ratio of variance to 
        mean is high enough to split.

        Parameters
        ----------
        threshold : int or float
            Minimum ratio of node's voxels' variance to mean to split.

        Returns
        -------
        bool
            Whether node is valid to split.
        """
        return (np.count_nonzero(self.mask) != 0 and 
                np.var(self.data[self.mask]) / np.mean(self.data[self.mask]) 
                >= threshold)
    
    def valid_std_dev_mean(self, threshold) -> bool:
        """Validation function for whether node's ratio of standard 
        deviation to mean is high enough to split.

        Parameters
        ----------
        threshold : int or float
            Minimum ratio of node's voxels' std dev to mean to split.

        Returns
        -------
        bool
            Whether node is valid to split.
        """
        return (np.count_nonzero(self.mask) != 0 and 
                np.std(self.data[self.mask]) / np.mean(self.data[self.mask]) 
                >= threshold)
    
    def valid_width(self, threshold) -> bool:
        """Validation function for whether node's thinnest dimension is 
        wide enough to split.

        Parameters
        ----------
        threshold : int
            Minimum width for node's thinnest dimension to split.

        Returns
        -------
        bool
            Whether node is valid to split.
        """
        return np.min(self.data.shape) >= threshold
    
    def valid_percent_nonzero_masked(self, threshold) -> bool:
        """Validation function for whether high enough percent of voxels 
        within node's mask are nonzero to split.

        Parameters
        ----------
        threshold : float
            Minimum percent of masked voxels that are nonzero to split

        Returns
        -------
        bool
            Whether node is valid to split.
        """
        return np.count_nonzero(self.data[self.mask]) >= threshold


