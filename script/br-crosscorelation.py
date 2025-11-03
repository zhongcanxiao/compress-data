import glob, os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import argparse
# from network.atomNet import AtomNet
from skimage.io import imread_collection, imread
import math
import numpy.matlib
from skimage.morphology import disk, cube
from skimage import filters
from skimage.filters import median
from skimage.filters.thresholding import _cross_entropy
from skimage.segmentation import chan_vese
import numpy as np
from skimage.filters.rank import enhance_contrast, mean_bilateral, majority, minimum, mean, equalize, sum_bilateral, otsu
from skimage import exposure
import skimage.filters.rank as rank 
from skimage.filters import difference_of_gaussians
from skimage import exposure
from skimage.measure import label
from matplotlib.patches import Rectangle



def sigmoid(x, x0=0.0 , k=10):

    y = 1.0 / (1.0 + np.exp(-k*(x-x0)))

    return y



def main():

    # Use average images within each scan to remove powder lines
    scan_avg_flag = True

    # Plotting flag
    plot_flag = False

    # Peak detection threshold
    peak_thres = 0.0035
    prob_thres = 0.6

    # Load raw data
    work_folder = os.getcwd()
    
    case_path = './'
    
    data_path = case_path + 'Image/'
    os.chdir(data_path)

    filenames = []
    for file in glob.glob("*.png"):
        # print(file)
        filenames.append(file)
    filenames.sort()  

    os.chdir(work_folder)
    save_path = case_path+'Result/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    num_image = len(filenames)
    raw_image = np.zeros((num_image,384,384))
    new_image = np.zeros((num_image,384,384))

    for i in range(num_image):

        raw_image[i,:,:] = imread(data_path+filenames[i])[:,:,0]/255.0

    mean_image = np.mean(raw_image,axis=0)

    for i in range(num_image):

        ind1 = np.maximum(i-10,0)
        ind2 = np.minimum(ind1+20, num_image-1)
        mean_image = np.mean(raw_image[ind1:ind2,:,:],axis=0)

        new_image[i,:,:] = np.clip(raw_image[i,:,:] - mean_image, a_min=0.0, a_max=1.0) 

        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        axs[0].imshow(raw_image[i,:,:])         
        axs[1].imshow(new_image[i,:,:])
        plt.savefig(save_path+'tmp_'+filenames[i], bbox_inches = 'tight', pad_inches = 0.2, dpi=96)
        plt.close()


if __name__ == '__main__':

    main()