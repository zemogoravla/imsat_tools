#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: agomez
"""
import os
import sys
import argparse

import numpy as np
from skimage.io import imread,imsave
#import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors

#from typing import Any


def benchmark_figures_of_merit():
    return ['evaluated', 'bad', 'invalid', 'totalbad', 'completeness', 'accuracy', 'mean_abserr', 'median_abserr']

def compute_benchmark(reference_image, test_image, mask_image, z_tolerance=1, show_figures=False, 
                      save_figures=False, save_figures_filename_prefix=''):
    
    # ensure compatible sizes
    min_rows = min(reference_image.shape[0], test_image.shape[0])
    min_cols = min(reference_image.shape[1], test_image.shape[1])
    # max_rows = max(reference_image.shape[0], test_image.shape[0])
    # max_cols = max(reference_image.shape[1], test_image.shape[1])

    reference_image = reference_image[0:min_rows,0:min_cols]
    mask_image = mask_image[0:min_rows,0:min_cols]
    test_image = test_image[0:min_rows,0:min_cols]
    
    
    #difference images
    diff_image = test_image - reference_image
    abs_diff_image = np.abs(diff_image)

    # indices where the reference image is valid and the pixel is not masked
    reference_valid_indices = np.logical_and(np.isfinite(reference_image), mask_image!=0)
    reference_valid_point_count = np.sum(reference_valid_indices)

    #indices where the the reference image is valid and the test image is bad in z
    test_bad_indices = np.logical_and(reference_valid_indices, abs_diff_image > z_tolerance)
    test_bad_point_count = np.sum(test_bad_indices)

    # indices where the the reference image is valid and the test image has no value
    test_invalid_indices = np.logical_and(reference_valid_indices, ~np.isfinite(test_image))
    test_invalid_point_count = np.sum(test_invalid_indices)
    
    # indices where the the reference image and the test image are both valid
    test_valid_indices = np.logical_and(reference_valid_indices, np.isfinite(test_image))
    #test_valid_point_count = np.sum(test_valid_indices)

    test_good_indices = np.logical_and(reference_valid_indices, abs_diff_image <= z_tolerance)
    test_bad_soso_indices = np.logical_and(test_bad_indices, abs_diff_image <= 2*z_tolerance)
    test_bad_bad_indices = np.logical_and(test_bad_indices, abs_diff_image > 2 * z_tolerance)

    #Metrics --------------------------------------------------
    # fraction of points in the image that are evaluated
    evaluated = reference_valid_point_count/(min_rows*min_cols)
    # fraction of evaluated points that have bad z in the test image
    bad = test_bad_point_count/reference_valid_point_count
    # fraction of evaluated points that are invalid in the test image
    invalid = test_invalid_point_count/reference_valid_point_count
    #invalid+bad
    totalbad = bad + invalid
    #
    # # así estaba antes  #avgErr = np.nansum(abs_diff_image[reference_valid_indices])/(reference_valid_point_count-test_invalid_point_count)
    # avgErrOld = np.nansum(abs_diff_image[reference_valid_indices]) / (
    #             reference_valid_point_count - test_invalid_point_count)
    # #print(avgErrOld)
    # meanAbsErr = np.mean(abs_diff_image[test_valid_indices])
    # #print(avgErr)
    # assert(np.abs(avgErrOld-meanAbsErr)<1e5)

    meanAbsErr = np.mean(abs_diff_image[test_valid_indices])
    medianAbsErr = np.median(abs_diff_image[test_valid_indices])

    # completenessOld = (reference_valid_point_count-test_invalid_point_count-test_bad_point_count)/reference_valid_point_count
    # completeness = 1-totalbad
    # assert(np.abs(completenessOld - completeness)<1e5)

    completeness = 1 - totalbad

    # # accuracy as median of the abs difference
    # accuracyOld = np.nanmedian(abs_diff_image[reference_valid_indices])
    # # accuracy as median of the difference
    # accuracyMedian = np.median(diff_image[test_valid_indices])
    # # accuracy as mean of the difference
    # accuracy = np.mean(diff_image[test_valid_indices])
    # accuracy as RMSE
    accuracy = np.sqrt(np.mean(diff_image[test_valid_indices]**2))

    # #dispersion as stddev of the abs difference
    # dispersionOld = np.nanstd(abs_diff_image[reference_valid_indices])
    # # dispersion as stddev of the difference
    # dispersion = np.std(diff_image[test_valid_indices])


    
    if show_figures or save_figures:
        if save_figures:
            filename = save_figures_filename_prefix + '_altitude_diff_wrt_gt.tif'
            imsave(filename, diff_image.astype(np.float32))
        

        fig = plt.figure(figsize=[6.4,4.8], dpi=100)
        plt.imshow(diff_image, cmap='bwr', vmin=-10, vmax=10)
        plt.title('Altitude diff w.r.t GT')
        plt.colorbar(ticks=[-10,0,10])
        fig.tight_layout()
        if save_figures:
            filename = save_figures_filename_prefix + '_altitude_diff_wrt_gt.png'
            plt.savefig(filename, transparent=True)
        
#         fig = plt.figure()
#         plt.imshow(abs_diff_image, cmap='jet', vmax=5)
#         plt.title('Abs Diff image')
#         plt.colorbar()
#         fig.tight_layout()

        region_image = reference_image.copy()
        region_image[test_invalid_indices] = 0
        region_image[test_bad_soso_indices] = 1
        region_image[test_bad_bad_indices] = 2
        region_image[test_good_indices] = 3


        fig = plt.figure(figsize=[6.4,4.8], dpi=100)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['black', "yellow", "red", "green"])
        im = plt.imshow(region_image, cmap=cmap, vmin=0, vmax=3)
        plt.title('Altitude abs diff w.r.t. GT (z_tol={}m)'.format(z_tolerance))
        # legends https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
        values = range(4)
        legends = ['Invalid', 'Bad (z_tol<|dz|<=2*z_tol)', 'Bad (|dz|>2*z_tol)', 'Good(|dz|<=z_tol']

        # get the colors of the values, according to the
        # colormap used by imshow
        colors = [im.cmap(im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color
        patches = [mpatches.Patch(color=colors[i], label=legends[i]) for i in range(len(values))]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.001, 1), loc='upper left', borderaxespad=0., fontsize='small')

        fig.tight_layout()
        
        if save_figures:
            filename = save_figures_filename_prefix + '_altitude_error_regions.png'
            plt.savefig(filename, transparent=True)
        
        if show_figures:
            plt.show()

    return evaluated, bad, invalid, totalbad, completeness, accuracy, meanAbsErr, medianAbsErr



def compute_benchmark_ex(reference_image_path, test_image_path, mask_image_path='', z_tolerance=1, z_min=-np.inf, z_max=np.inf, show_figures=False, save_figures=False):
    ref_image = imread(reference_image_path)
    ref_image[ref_image==-9999] = np.nan

    test_image = imread(test_image_path)

    if mask_image_path != '':
        mask_image = imread(mask_image_path).astype(np.uint8)
    else:
        mask_image = np.ones_like(ref_image,dtype=np.uint8)


    mask_image[ref_image < z_min] = 0
    mask_image[ref_image > z_max] = 0

    #evaluated, bad, invalid, totalbad, avgErr, accuracy, completeness, dispersion = compute_benchmark(ref_image, test_image,mask_image,z_tolerance)
    metrics = compute_benchmark(ref_image, test_image, mask_image, z_tolerance, show_figures, save_figures)
    return (metrics)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Computes a benchmark')
    parser.add_argument('reference_image_path', metavar='RIM', type=str,
                        help='Reference image path')
    parser.add_argument('test_image_path', metavar='TIP', type=str,
                        help='Test image path')

    parser.add_argument('--z_tolerance', type=float, default=1,
                        help='Z tolerance in meters to compute completion (default: 1)')

    parser.add_argument('--mask_image_path', metavar='MIP', type=str, default='',
                        help='Mask_image_path')

    parser.add_argument('--z_min', type=float, default=-np.inf,
                        help='Min Z to consider in meters (default: -inf)')

    parser.add_argument('--z_max', type=float, default=np.inf,
                        help='Max Z to consider in meters (default: +inf)')

    parser.add_argument('--show_figures', action='store_true',
                        help='Show the figures')

    parser.add_argument('--save_figures', action='store_true',
                        help='Save the figures')

    args = parser.parse_args()
    
    metrics = compute_benchmark_ex(args.reference_image_path, args.test_image_path, args.mask_image_path, args.z_tolerance, args.z_min, args.z_max, args.show_figures, args.save_figures)

    print(*metrics)
