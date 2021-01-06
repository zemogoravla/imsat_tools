#!/usr/bin/env python
# Copyright (C) 2018, Drouyer Sebastien <sdrdis@gmail.com>

import numpy as np
import scipy.misc
import scipy.ndimage
import os
import os.path
from scipy.sparse import dia_matrix, dok_matrix, coo_matrix
import skimage
from skimage.morphology import disk
from skimage.filters.rank import gradient
import skimage.future.graph
import matplotlib.pyplot as plt
import time
import skimage.measure
from scipy.interpolate import griddata
#from scipy.spatial import ConvexHull
#import scipy.ndimage
from os.path import join
import argparse
import skimage.io


# MAIN FUNCTIONS -------------------------------------------------------------- 

# ALGORITHM 2
def remove_speckles(img, args): #discontinuity_th=10, area_th=100, max_distance = 100): 
    #args.speckle_removal_discontinuity_threshold, args.speckle_removal_area_threshold, args.speckle_removal_max_distance
    '''
    Set to "nan" all the small regions (pixel count less than area_th) that are 
    different than their surrounding by a value greater than discontinuity_th.
    '''
    # reserve output image that will have no speckles (speckles will be set to nan)
    despeckled_img = img.copy()
    
    #identify the undefined pixels
    undefined_pixels = np.isnan(img)
    
    # interpolate the undefined pixels of the image
    interpolated_img = interpolate_undefined_regions(img, method='nearest')
    
    # compute the morphological image gradient of the img
    img_gradient = scipy.ndimage.morphological_gradient(img, size=(3, 3))
    
    #compute the morphological image gradient of the interpolated img
    interpolated_img_gradient = scipy.ndimage.morphological_gradient(interpolated_img, 
                                                                     size=(3, 3))
    
    # identify speckle boundaries
    defined_stable_np = np.logical_or(np.isnan(img_gradient), 
                                      img_gradient >= args.speckle_removal_discontinuity_threshold)
    
    # This is just a hole filling strategy (fills from the hole boundaries)
    # Since the morphological gradient points out the boundary pixels, pixels in the
    # center of a speckle must also be added
    distance_map = scipy.ndimage.distance_transform_edt(defined_stable_np)
    separated_areas_np = np.logical_and(interpolated_img_gradient < args.speckle_removal_discontinuity_threshold,
                                        distance_map < args.speckle_removal_max_distance)
    label_img, cc_num = scipy.ndimage.label(separated_areas_np)
    label_img[undefined_pixels] = 0
    label_img = remove_cc_area_labels(label_img, args.speckle_removal_area_threshold)
    remove_areas = label_img == 0
    # end of hole filling strategy
    
    # set to nan all the speckle pixels
    despeckled_img[remove_areas] = np.nan
    
    return despeckled_img 
    

# ALGORITHM 3
def diffuse(disp_np, im_np, args):
    start_time = time.time()
    
    # if the stereo image is color, convert to gray
    if (len(im_np.shape) == 3):
        im_np = np.mean(im_np, 2).astype('uint8')

    # ALGORITHM 3 - step 1
    # compute the local gradient of the gray image (local maximum-local minimum) on a disk
    gr_np = gradient(im_np, disk(args.gradient_size))
    print ('Gradient:', time.time() - start_time)

    # ALGORITHM 3 - step 2
    # segment the gray image using watershed
    # regions are basins starting from local minima of the gradient
    markers_np = watershed_segmentation(im_np, gr_np, args)
    print ('Segmentation:', time.time() - start_time)

    # ALGORITHM 3 - step 3 
    # compute the mean disparity of each region (labels_vals) 
    # labels_fixed gives for each region if it has enough defined pixels
    labels_fixed, labels_vals,labels_ratio = get_labels_vals_fixed(markers_np, disp_np, args)
    print ('Average values and fixed:', time.time() - start_time)
    
    # ALGORITHM 3 - step 4  (Details in ALGORITHM 4) --------------------------
    # get the connections between labels and the min gradient value for each connection
    low_labels_np, high_labels_np, agr_vals_np,low_labels_mean_disp, high_labels_mean_disp = get_adjacency_graph_min_disp(markers_np, gr_np, labels_vals)

    print ('Adjacency:', time.time() - start_time)

    # build a bidirectional weighted graph 
    # origin vertices     :     low  high
    # detination vertices :     high low
    # weights             :     mingradient mingradient
    #
    # The -1 in the vertices makes that regions of undefined values are now -1
    # and the labels of interest are now starting from 0 
    labels_from = np.hstack((low_labels_np, high_labels_np)) - 1
    labels_to = np.hstack((high_labels_np, low_labels_np)) - 1
    weights = np.hstack((agr_vals_np, agr_vals_np)).astype(float)

    # From each originating vertex weight_min gives the minimum weight for all
    # the connections originating from that vertex
    weights_min, grouped_labels_from = min_by_group(weights, labels_from)

    # w_mins holds the minimum gradient for all the connections of each originating vertex
    w_mins = np.zeros(labels_from.shape)
    for i in range(labels_from.shape[0]):
        w_mins[i] = weights_min[labels_from[i]]

    # the minimum weights are subtracted
    weights -= w_mins

    # weights are moved to a 0-1 interval through a gaussian
    # weight==1 when the gradient is the minimum of all the connections
    weights = gaussian(weights, (w_mins + args.weights_alpha) / args.weights_beta)
    
    # very small weights are set to a minimum
    weights[weights < args.min_absorption] = args.min_absorption

    # accumulate the weights per originating vertex and normalize the weights
    sum_weights, groups = sum_by_group(weights, labels_from)
    for i in range(weights.shape[0]):
        sum_ws = sum_weights[labels_from[i]]
        weights[i] /= sum_ws

    print ('Weights construction:', time.time() - start_time)

    # fixed is 1 if the region has a low undefined/defined ratio
    fixed = labels_fixed[1:].astype(float)
    nb_elems = fixed.shape[0]
    center_labels = np.arange(nb_elems)
    
    # add to the graph "self" edges with -1 weight for all the regions 
    labels_from = np.hstack((labels_from, center_labels))
    labels_to = np.hstack((labels_to, center_labels))
    weights = np.hstack((weights, -np.ones(nb_elems)))

    # diagonal matrix M. Element in the diagonal is 1 if the region DOES NOT have a low undefined/defined ratio
    M = dia_matrix((np.array([1.0 - fixed]), np.array([0])), shape=(nb_elems, nb_elems))
    # diagonal matrix M2. Element in the diagonal is 1 if the region DOES have a low undefined/defined ratio
    M2 = dia_matrix((np.array([fixed]), np.array([0])), shape=(nb_elems, nb_elems))
    # f holds the mean disparity values of the regions
    f = labels_vals[1:]
    f[np.isnan(f)] = 0
    
    # W is a sparse matrix in COOrdinate format
    W = coo_matrix((weights, (labels_from, labels_to)), shape=(nb_elems, nb_elems))
       
    
    # ALGORITHM 3 - step 5
    # setup the system AX=B where 
    # A is the matrix of weights conditioned by the fixed flag of each labeled region
    # B is the mean value per region conditioned by the fixed flag of the region
    A = M.dot(W) + M2
    B = M2.dot(f)

    AT = A.transpose()
    A = AT.dot(A)
    B = AT.dot(B)

    print ('Equation preparation:', time.time() - start_time)

    ret = scipy.sparse.linalg.spsolve(A, B)

    print ('Equation solving:', time.time() - start_time)

    # ALGORITHM 3 - step 6
    props = skimage.measure.regionprops(markers_np)
    map_np = generate_map(im_np.shape, markers_np, ret, props, -1)
    print ('Generate map:', time.time() - start_time)

    return map_np

# ALGORITHM 5
def get_lrc_map(left_disp, right_disp, just_a_very_large_disp=1e6):
    '''
    The disparity is consistent in a certain location if, going from left to 
    right (via the left_disp) and back from right to left (via the right_disp)
    the resulting location column differs from the original location column in 
    less than a certain threshold 
    
    This function returns the consistency maps that holds the differences in each
    location when checking from one disparity to another and back to the former.
    '''

    # get  coordinates of pixels along the image (the row-major order, x are columns) 
    # xs and ys are one dimensional vectors
    nb_positions = left_disp.shape[0] * left_disp.shape[1]
    positions = np.arange(nb_positions)
    xs = (positions % left_disp.shape[1]).astype(int)
    ys = (positions / left_disp.shape[1]).astype(int)

    # reshape of the left dispárity as  a one dimensional vector
    reshaped_left_disp_np = left_disp.reshape(nb_positions)
    
    # get in disp_xs the x coordinate of the corresponding pixel
    disp_xs = np.round(xs - reshaped_left_disp_np).astype(int)
    undefined_values = np.logical_or(np.logical_or(np.isnan(reshaped_left_disp_np), disp_xs < 0),
                                     disp_xs >= left_disp.shape[1])
    disp_xs[undefined_values] = xs[undefined_values]
    
    # Get the difference in pixel x coordinate going right with the left2right 
    # disparity and back to left with right2left disparity
    diff = np.abs(right_disp[ys, disp_xs] + left_disp[ys, xs])

    #location with undefined disparity or undefined difference are set to a very large disparity
    diff[undefined_values] = just_a_very_large_disp
    diff[np.isnan(diff)] = just_a_very_large_disp

    left_consistency_np = np.zeros(left_disp.shape)
    left_consistency_np[ys, xs] = diff

    return left_consistency_np



def lr_check(disp_lr_np, disp_rl_np, args):
    #left-right differences
    lr_consistency_np = get_lrc_map(-disp_lr_np, -disp_rl_np)
    #right-left differences
    rl_consistency_np = get_lrc_map(-disp_rl_np, -disp_lr_np)
    # set to nan inconsistent pixels
    disp_lr_np[lr_consistency_np > args.lrc_threshold] = np.nan
    disp_rl_np[rl_consistency_np > args.lrc_threshold] = np.nan
    
    return disp_lr_np, disp_rl_np

# FUNCIONS --------------------------------------------------------------------


# called in ALGORITHM 2
def interpolate_undefined_regions(img, method='nearest'):
    '''
    Basic interpolation of the undefined (nan) pixels of the image using
    nearest neighbors
    '''
    # output image that will be filled with the interpolated values
    interpolated_img = img.copy()
    
    # get the coordinates of the defined pixels
    (defined_pixels_coord_y, defined_pixels_coord_x) = np.where(~np.isnan(img))
    defined_pixels = np.vstack((defined_pixels_coord_y, defined_pixels_coord_x)).T
    
    #get the values for the defined pixels
    defined_values = img[defined_pixels_coord_y, defined_pixels_coord_x]
    
    # get the coordinates of the undefined pixels to interpolate
    (undefined_pixels_coord_y, undefined_pixels_coord_x) = np.where(np.isnan(img))
    undefined_pixels = np.vstack((undefined_pixels_coord_y, undefined_pixels_coord_x)).T
    
    # interpolate the undefined values
    interpolated_values = griddata(defined_pixels, defined_values, 
                                   undefined_pixels, method=method)
    # set the interpolated values in the image
    interpolated_img[undefined_pixels_coord_y, undefined_pixels_coord_x] = interpolated_values
    
    return interpolated_img

# called in ALGORITHM 3 - step 2
def watershed_segmentation(im_np, gr_np, args):
    '''
    Watershed segmentation of the image starting from markers defined by the
    h_minima of the gradient
    '''
    start_time = time.time()
    hmin_np = skimage.morphology.h_minima(gr_np, args.h) > 0
    print ('H-minima:', time.time() - start_time)
    markers_np = skimage.morphology.label(hmin_np).astype('int32')
    print ('Markers:', time.time() - start_time)
    markers_np = skimage.morphology.watershed(gr_np, markers_np)
    print ('Watershed:', time.time() - start_time)

    return markers_np

# called in ALGORITHM 3 - step 3
def get_labels_vals_fixed(markers_np, disp_np, args):
    '''
    The function computes the nanmean of the disparity on each region (labels_vals)
    and if each region has a low enough undefined/defined pixel ratio (labels_fixed).
    The "low enough" is determined by args.defined_threshold_ratio
    
    markers_np are the labeled regions from the segmentation of the gray image
    disp_np is the disparity map
    
    '''
    nb_markers = np.max(markers_np) + 1
    labels_vals = np.zeros(nb_markers)
    labels_fixed = np.zeros(nb_markers)
    nan_np = np.isnan(disp_np)

    # get the proportion of undefined pixels for each region of the segmentation
    # nan.np is 1 on undefined pixels and 0 otherwise
    # undefined_values holds the proportions 
    undefined_indexes = np.unique(markers_np)
    undefined_values = scipy.ndimage.mean(nan_np.astype(float), markers_np, 
                                          undefined_indexes)

    # set label to 0 in all undefined pixels
    def_markers_np = markers_np.copy()
    def_markers_np[nan_np] = 0

    # get the mean disparity for each region ignoring nans
    # mean_indexes holds the indexes of the pixels of the regions
    mean_indexes = np.unique(def_markers_np)
    if (mean_indexes[0] == 0):
        mean_indexes = mean_indexes[1:]
    mean_values = scipy.ndimage.mean(disp_np, def_markers_np, mean_indexes)

    # save the mean_values of disparity in label_vals
    labels_vals[mean_indexes] = mean_values
    # save the proportions of undefined pixels in labels_fixed
    labels_fixed[undefined_indexes] = undefined_values
    # threshold the proportion. 
    #labels_fixed will indicate if the ratio low enough or not. 
    #That is, if the region has enough defined disparity pixels
    labels_ratio = labels_fixed
    labels_fixed = labels_fixed < args.defined_threshold_ratio

    return labels_fixed, labels_vals, labels_ratio


# called in ALGORITHM 3 - step 4
def get_adjacency_graph_min_disp(labels_np, gradient_np, labels_vals, ignore=0):
    '''
    Computes the connections between the regions and the min gradient for each 
    connection.  Returns the list of connections given by the two related 
    labels (low and high) and the minimum local gradient for each connection
    
    gradient_np is the local gradient of the gray image that originated the 
    labels (local gradient is computed as  local max-local min)
    '''
    
    # number of labels
    nb_labels = labels_np.max() + 1
    # structuring element is 3x3 cross
    ed_footprint = scipy.ndimage.generate_binary_structure(2,1)
    # gray dilation of the labels
    # maximum on the structuring element
    d_labels_np = scipy.ndimage.grey_dilation(labels_np, footprint=ed_footprint)

    # morphological contours of the regions 
    seg_np = (d_labels_np - labels_np) > 0

    # get the coordinates of the contours
    where_seg = np.where(seg_np)
    ys, xs = where_seg
    # for each coordinate of the contours, get the labels of the related regions
    low_labels_np = labels_np[ys, xs].astype('uint64')
    high_labels_np = d_labels_np[ys, xs].astype('uint64')

    # connections_np is just an identifier for a relation between two regions
    connections_np = low_labels_np * nb_labels + high_labels_np
    # for each pixel of the contours, get the local gradient 
    vals_np = gradient_np[ys, xs]

    # for each pair of connected regions get the min gradient between them
    # agr_vals_np holds the min gradients
    # connections_np is now the list of unique connections between regions
    agr_vals_np, connections_np = min_by_group(vals_np, connections_np)

    # get the low_labels and high_labels corresponding to the list of connections
    low_labels_np = (connections_np / nb_labels).astype('uint64')
    high_labels_np = (connections_np % nb_labels).astype('uint64')
    
    low_labels_mean_disp = labels_vals[low_labels_np]
    high_labels_mean_disp = labels_vals[high_labels_np]

    return low_labels_np, high_labels_np, agr_vals_np, low_labels_mean_disp, high_labels_mean_disp


# called in ALGORITHM 3 - step 6
def generate_map(shape, markers_np, labels_vals, props, label_decal=0):
    '''
    Build the disparity map
    '''
    map_np = np.zeros(shape)
    for prop in props:
        label_id = prop.label
        bb = prop.bbox
        ex_markers_np = markers_np[bb[0]:bb[2], bb[1]:bb[3]]
        ex_map_np = map_np[bb[0]:bb[2], bb[1]:bb[3]]
        where_np = ex_markers_np == label_id
        ex_map_np[where_np] = labels_vals[label_id+label_decal]
    return map_np





# AUXILIARY FUNCTIONS----------------------------------------------------------
def imread(filename):
    img = skimage.io.imread(filename, as_gray=True)
    return img

def imsave(filename, img):
    skimage.io.imsave(filename, img.astype(np.float32))

def normalize(im_np, percentile=1):
    '''
    Normalize image to 0..255 range
    '''
    lower = np.percentile(im_np, percentile)
    upper = np.percentile(im_np, 100-percentile)
    im_np = (im_np.astype(float) - lower) / (upper - lower)
    im_np[im_np < 0] = 0
    im_np[im_np > 1] = 1
    return (im_np * 255).astype('uint8')

def gaussian(val, sigma=2.0):
    return np.exp(-0.5*(val/sigma)**2)

def min_by_group(data, groups):
    '''
    Compute the min on data for each group in groups
    # https://stackoverflow.com/questions/8623047/group-by-max-or-min-in-a-numpy-array
    '''
    # sort with major key groups, minor key data
    order = np.lexsort((data, groups))
    groups = groups[order] # this is only needed if groups is unsorted
    data = data[order]
    # construct an index which marks borders between groups
    index = np.empty(len(groups), 'bool')
    index[0] = True
    index[1:] = groups[1:] != groups[:-1]
    return data[index], np.unique(groups)

def median_by_group(data, groups):
    '''
    Compute the median on data for each group in groups
    # https://stackoverflow.com/questions/8623047/group-by-max-or-min-in-a-numpy-array
    '''
    # sort with major key groups, minor key data
    order = np.lexsort((data, groups))
    groups = groups[order] # this is only needed if groups is unsorted
    data = data[order]
    # construct an index which marks borders between groups
    index = np.empty(len(groups), 'bool')
    index[0] = True
    index[1:] = groups[1:] != groups[:-1]
    last_index_of_group = np.roll(index,-1)
    min_indices = np.where(index)[0]
    median_indices = np.floor( (np.where(last_index_of_group)[0]+np.where(index)[0])/2 ).astype(np.int) 
    #print(min_indices)
    #print(median_indices)
    #print(data)
    return data[median_indices], np.unique(groups)

def sum_by_group(values, groups):
    '''
    Sum in data for each group in groups
    # https://stackoverflow.com/questions/4373631/sum-array-by-number-in-numpy    
    '''
    order = np.argsort(groups)
    groups = groups[order]
    values = values[order]
    values.cumsum(out=values)
    index = np.ones(len(groups), 'bool')
    index[:-1] = groups[1:] != groups[:-1]
    values = values[index]
    groups = groups[index]
    values[1:] = values[1:] - values[:-1]
    return values, groups

def remove_cc_area_labels(label_img, min_area=100):
    '''
    Given an image of labels, set label=0 for all the regions 
    with pixel_count<min_area
    '''
    map_np = label_img > 0
    cc_num = np.max(label_img)
    cc_areas = scipy.ndimage.sum(map_np, label_img, range(cc_num + 1))
    area_mask = (cc_areas < min_area)
    label_img[area_mask[label_img]] = 0
    return label_img


# MAIN-------------------------------------------------------------------------
if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser(description='Diffuses a disparity map.')
    parser.add_argument('reference_image_path', metavar='RIM', type=str,
                        help='Rectified reference image path')
    parser.add_argument('lr_disparity_path', metavar='LRDISP', type=str,
                        help='Left-right disparity map path')
    parser.add_argument('out_disparity_path', metavar='LRDISP', type=str,
                        help='Left right disparity map path')

    parser.add_argument('--secondary_image_path', metavar='SIM', type=str, default='',
                        help='Rectified secondary image path')
    parser.add_argument('--rl_disparity_path', metavar='RLDISP', type=str, default='',
                        help='Right-left disparity map path')

    parser.add_argument('--lrc_threshold', type=float, default=1.5,
                        help='Left Right consistency threshold (default: 1.5)')
    parser.add_argument('--nb_iterations', type=int, default=1,
                        help='How many iterations should the algorithm be applied (default: 1)')

    parser.add_argument('--speckle_removal_discontinuity_threshold', type=float, default=10,
                        help='How much change in disparity is considered a discontinuity (default: 10)')
    parser.add_argument('--speckle_removal_area_threshold', type=int, default=200,
                        help='Max area of speckles (measured in number of pixels) (default: 200)')
    parser.add_argument('--speckle_removal_max_distance', type=float, default=8,
                        help='If two neighboring speckles have a similar disparity, how much apart (in pixels) should they be so that they are considered different speckles (default: 8)')

    parser.add_argument('--gradient_size', type=int, default=1,
                        help='Gradient size (default: 1)')
    parser.add_argument('--h', type=int, default=1,
                        help='H-minima (default: 1)')
    parser.add_argument('--defined_threshold_ratio', type=float, default=0.5,
                        help='For each region, what is the minimum area ratio that must be defined in the disparity map so that the region remains fixed during diffusion (default: 0.5)')
    parser.add_argument('--min_absorption', type=float, default=0.00001,
                        help='Minimum weight affected to a connection between neighboring regions (default: 0.00001)')
    parser.add_argument('--weights_alpha', type=float, default=10.0,
                        help='Weights alpha parameter (default: 10)')
    parser.add_argument('--weights_beta', type=float, default=5.0,
                        help='Weights beta parameter (default: 5)')

    parser.add_argument('--debug', type=bool,
                        help='Debug mode')
    parser.add_argument('--debug_folder', type=str, default='',
                        help='Debug folder')

    args = parser.parse_args()

    # check if there is a secondary (right) image and a right-left disparity map
    simpath_defined = args.secondary_image_path != ''
    sdpath_defined = args.rl_disparity_path != ''
    is_secondary_defined = (simpath_defined and sdpath_defined)
    if (simpath_defined != sdpath_defined):
        raise Exception('Either should the secondary image and the right left disparity map both defined, either none of them.')

    # load and normalize reference (left) image
    l_np = normalize(interpolate_undefined_regions(imread(args.reference_image_path)))
    #load left-right disparity map
    disp_lr_np = imread(args.lr_disparity_path)

    # If available, load right image and right-left disparity map.
    # Check consistency between left-right and right-left consistency map
    if (is_secondary_defined):
        r_np = normalize(interpolate_undefined_regions(imread(args.secondary_image_path)))
        disp_rl_np = imread(args.rl_disparity_path)
        #
        lr_consistency_np = get_lrc_map(-disp_lr_np, -disp_rl_np)
        rl_consistency_np = get_lrc_map(-disp_rl_np, -disp_lr_np)
        disp_lr_np[lr_consistency_np > args.lrc_threshold] = np.nan
        disp_rl_np[rl_consistency_np > args.lrc_threshold] = np.nan

    # Algorithm
    time_start = time.time()
    for i in range(args.nb_iterations):
        print('---------------------------------------------------------------------------------')
        print('Iteration %d of %d' % (i+1,args.nb_iterations))
        if (args.debug): imsave(join(args.debug_folder, 'disp_lr.png'), disp_lr_np)
        remove_speckles(disp_lr_np, args.speckle_removal_discontinuity_threshold, args.speckle_removal_area_threshold, args.speckle_removal_max_distance)
        if (args.debug): imsave(join(args.debug_folder, 'disp_lr_2.png'), disp_lr_np)
        disp_lr_np = diffuse(disp_lr_np, l_np, args)
        if (args.debug): imsave(join(args.debug_folder, 'disp_lr_3.png'), disp_lr_np)

        if (is_secondary_defined):
            if (args.debug): imsave(join(args.debug_folder, 'disp_rl.png'), disp_rl_np)
            remove_speckles(disp_rl_np, args.speckle_removal_discontinuity_threshold, args.speckle_removal_area_threshold, args.speckle_removal_max_distance)
            if (args.debug): imsave(join(args.debug_folder, 'disp_rl_2.png'), disp_rl_np)
            disp_rl_np = diffuse(disp_rl_np, r_np, args)
            if (args.debug): imsave(join(args.debug_folder, 'disp_rl_3.png'), disp_rl_np)

            #left-right differences
            lr_consistency_np = get_lrc_map(-disp_lr_np, -disp_rl_np)
            #right-left differences
            rl_consistency_np = get_lrc_map(-disp_rl_np, -disp_lr_np)
            # set to nan inconsistent pixels
            disp_lr_np[lr_consistency_np > args.lrc_threshold] = np.nan
            disp_rl_np[rl_consistency_np > args.lrc_threshold] = np.nan

    print ('Total elapsed time:', time.time() - time_start)
    if (args.debug): scipy.misc.imsave(join(args.debug_folder, 'l.png'), l_np)
    if (args.debug): scipy.misc.imsave(join(args.debug_folder, 'r.png'), r_np)

    # Save result
    imsave(args.out_disparity_path, disp_lr_np)

