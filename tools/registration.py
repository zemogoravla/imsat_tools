import os
import numpy as np
import matplotlib.pyplot as plt
import skimage
import subprocess

import skimage.feature
from skimage.transform import EuclideanTransform, SimilarityTransform, AffineTransform
from skimage.measure import ransac
import cv2
from skimage.transform import warp

#------------------------------------------------------------------------
# NCC shift registration
#
#
#------------------------------------------------------------------------

def read_nccshift_transformation(filename):
    transform = np.loadtxt(filename, delimiter=' ', ndmin=2)
    return transform

def write_nccshift_transformation(filename, transform):
    np.savetxt(filename,transform, delimiter=' ')
    
def register_images_with_nccshift(reference_image, moving_image, scale=False, pixel_range=9,
                                  bdint_interpolate_reference_image=False, bdint_interpolate_moving_image=False, bdint_min_max_avg_option='avg',
                                  adjust_dz_method = None):

    #adjust_dz_method  select one of nanmean, nanmedian

    reference_image_filename = 'tmp_registration_reference_image.tif'
    moving_image_filename = 'tmp_registration_moving_image.tif'
    registered_image_filename = 'tmp_registration_registered_image.tif'
    registration_result_filename = 'tmp_registration_result.txt'

    # ref & mov are the images to be registered
    # reference_image & moving_image are the original images (eventually of different sizes)
    ref = reference_image.copy()
    mov = moving_image.copy()

    # if bdint_interpolate_reference_image:
    #     ref = interpolate_image_with_bdint(ref, bdint_min_max_avg_option)
    # if bdint_interpolate_moving_image:
    #     mov = interpolate_image_with_bdint(mov, bdint_min_max_avg_option)

    #adjust sizes for the original and operative images
    rows = max(ref.shape[0], mov.shape[0])
    cols = max(ref.shape[1], mov.shape[1])
    # create the big images initialized with NaN
    ref_resized = np.ones((rows,cols))*np.nan
    mov_resized = np.ones((rows,cols))*np.nan
    reference_image_resized = np.ones((rows, cols)) * np.nan
    moving_image_resized = np.ones((rows, cols)) * np.nan

    ref_resized[:ref.shape[0],:ref.shape[1]] = ref
    mov_resized[:mov.shape[0],:mov.shape[1]] = mov
    reference_image_resized[:ref.shape[0],:ref.shape[1]] = reference_image
    moving_image_resized[:mov.shape[0],:mov.shape[1]] = moving_image

    #interpolate NaNs
    if bdint_interpolate_reference_image:
        ref_resized = interpolate_image_with_bdint(ref_resized, bdint_min_max_avg_option)
    if bdint_interpolate_moving_image:
        mov_resized = interpolate_image_with_bdint(mov_resized, bdint_min_max_avg_option)

    skimage.io.imsave(reference_image_filename, ref_resized)
    skimage.io.imsave(moving_image_filename, mov_resized)
    compute_nccshift_registration(reference_image_filename,
                                  moving_image_filename, 
                                  registered_image_filename, 
                                  registration_result_filename, 
                                  scale=scale, pixel_range=pixel_range)
    
    registered_image = skimage.io.imread(registered_image_filename)
    registration_transform = read_nccshift_transformation(registration_result_filename)
    # print('registration_transform (antes)', registration_transform)

    # update the registration delta_z using only originally valid values
    # delta_z is not valid if images were previously interpolated
    if bdint_interpolate_reference_image or bdint_interpolate_moving_image:
        transf = registration_transform.copy()
        transf[0][3] = 0  # set dz=0
        registered_moving_image_resized = apply_nccshift_transformation(moving_image_resized, transf[0])
        #compute dz
        dz = np.nanmedian(reference_image_resized - registered_moving_image_resized)
        #update the transform and the registered image

        registration_transform[0][3] = dz
        registered_image = registered_moving_image_resized + dz

    # nccshift uses mean, this is in case you want to do median instead
    if adjust_dz_method=='nanmean':
        dz = np.nanmean(reference_image_resized - registered_image)
        registration_transform[0][3] += dz   # added adjustment to the previous of nccshift
        registered_image = registered_image + dz
    elif adjust_dz_method=='nanmedian':
        dz = np.nanmedian(reference_image_resized - registered_image)
        registration_transform[0][3] += dz # added adjustment to the previous of nccshift
        registered_image = registered_image + dz

    # print('registration_transform (despues)', registration_transform)

    return registered_image, registration_transform

def register_images_height(reference_image, moving_image, mask_image,
                                  adjust_dz_method = 'nanmedian'):
    # ref & mov are the images to be registered
    # reference_image & moving_image are the original images (eventually of different sizes)
    ref = reference_image.copy()
    mov = moving_image.copy()

    #adjust sizes for the original and operative images
    rows = max(ref.shape[0], mov.shape[0])
    cols = max(ref.shape[1], mov.shape[1])
    # create the big images initialized with NaN
#     ref_resized = np.ones((rows,cols))*np.nan
#     mov_resized = np.ones((rows,cols))*np.nan
    reference_image_resized = np.ones((rows, cols)) * np.nan
    moving_image_resized = np.ones((rows, cols)) * np.nan
    mask_image_resized = np.ones((rows, cols)) * np.nan
    

#     ref_resized[:ref.shape[0],:ref.shape[1]] = ref
#     mov_resized[:mov.shape[0],:mov.shape[1]] = mov
    reference_image_resized[:ref.shape[0],:ref.shape[1]] = reference_image
    moving_image_resized[:mov.shape[0],:mov.shape[1]] = moving_image
    mask_image_resized[:mask_image.shape[0],:mask_image.shape[1]] = mask_image
    
    moving_image_resized *= mask_image_resized
    
    if adjust_dz_method=='nanmean':
        dz = np.nanmean(reference_image_resized - moving_image_resized)
        mov = mov + dz
    elif adjust_dz_method=='nanmedian':
        dz = np.nanmedian(reference_image_resized - moving_image_resized)
        mov = mov + dz
    else:
        raise ValueError('Wrong method, accepted: nanmedian and nanmean')
    
    return mov, dz
                         
                         
                    
    
def compute_nccshift_registration(reference_image_filename, moving_image_filename, registered_image_filename, registration_result_filename, scale=False, pixel_range=9 ):
    REGISTRATION_EXE = '/home/agomez/ownCloud/Documents/doctorado/satelite/registrado/nccshift/nccshift'
    REGISTRATION_COMMAND_FORMAT = '%s %s %s %s %d %s > %s'
    
    if scale:
        scale_option = ''
    else:
        scale_option = '-noscale'
    
    
    command = REGISTRATION_COMMAND_FORMAT % (REGISTRATION_EXE,
                                             reference_image_filename,
                                             moving_image_filename,
                                             scale_option,
                                             pixel_range,
                                             registered_image_filename,
                                             registration_result_filename) 
    
    
    print('Running: ', command)
    subprocess.call('pwd', shell=True)
    subprocess.call(command, shell=True)



def apply_nccshift_transformation(img, nccshift_transformation):
    
    dx = nccshift_transformation[0].astype(int) #ya es un entero pero sino se queja al poner indices en la matriz
    dy = nccshift_transformation[1].astype(int)
    alpha = nccshift_transformation[2]
    dr = nccshift_transformation[3]
    
    rows, cols = img.shape
    transformed_img = np.ones_like(img) * np.nan
    for y in range(rows):
        for x in range(cols):
            if 0<=x+dx<cols and 0<=y+dy<rows:
                transformed_img[y,x] = img[int(y+dy), int(x+dx)] + dr
    
    return transformed_img

#------------------------------------------------------------------------
# Lowest neighbor interpolation
#
# Used before doing the registration
#------------------------------------------------------------------------

def interpolate_image_with_bdint(image, min_max_avg_option = 'min'):
    image_to_interpolate_filename = 'tmp_image_to_bdint_interpolate.tif'
    interpolated_image_filename = 'tmp_bdint_interpolated_image_filename.tif'

    skimage.io.imsave(image_to_interpolate_filename, image)

    compute_bdint_interpolation(image_to_interpolate_filename,
                                interpolated_image_filename,
                                min_max_avg_option)

    interpolated_image = skimage.io.imread(interpolated_image_filename)

    return interpolated_image

def compute_bdint_interpolation(image_filename, interpolated_image_filename, min_max_avg_option = 'min'):
    # BDINT
    # lowest neighbor interpolation
    # optionally: highest - neighbor and average - neighbor
    # Used before nccshift registration
    # usage:
    # 	bdint [-a {min|max|avg}] [in.tiff [out.tiff]]


    BDINT_EXE = '/home/agomez/ownCloud/Documents/doctorado/satelite/registrado/bdint/bdint'
    BDINTPC5_EXE = '/home/agomez/ownCloud/Documents/doctorado/satelite/registrado/bdint/bdint5pc'

    BDINT_COMMAND_FORMAT = '%s -a %s %s %s'

    command = BDINT_COMMAND_FORMAT % (BDINTPC5_EXE if min_max_avg_option=='pc5' else BDINT_EXE,
                                     min_max_avg_option,
                                     image_filename,
                                     interpolated_image_filename,
                                     )

    print('Running: ', command)
    subprocess.call('pwd', shell=True)
    subprocess.call(command, shell=True)


#------------------------------------------------------------------------
# Phase cross correlation (scikit image)
# https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation
# https://scikit-image.org/docs/stable/auto_examples/registration/plot_masked_register_translation.html#sphx-glr-auto-examples-registration-plot-masked-register-translation-py
#------------------------------------------------------------------------





#------------------------------------------------------------------------
# Feature based registration
#
#
#------------------------------------------------------------------------

class TranslationTransform(object):
    r"""TranslationTransform"""
    def __init__(self, translation=[0,0]):
        self.translation = translation
    
    def residuals(self, src, dst):
        return np.sqrt(np.sum((dst - src - self.translation)**2, axis=1))
    
    def estimate(self, src, dst):
        self.translation = np.mean(dst-src, axis=0)
        return True
        
    
def find_correspondences(img1, img2):
    a = utils.simple_equalization_8bit(img1)
    b = utils.simple_equalization_8bit(img2)

    # KP
    sift = kaze = cv2.KAZE_create() #cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(a, None)
    kp2, des2 = sift.detectAndCompute(b, None)

    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    # cv2.drawMatchesKnn expects list of lists as matches.
    #img3 = cv2.drawMatchesKnn(a,kp1,b,kp2,good,a,flags=2)
    #display_image(img3)

    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    
    return pts1, pts2


def register_images_with_features(reference_image, moving_image, transformation_type='translation', debug=False):
    
    pts1, pts2 = find_correspondences(reference_image, moving_image)
    
    if transformation_type=='translation':
        model = TranslationTransform
        min_samples = 1
    elif transformation_type=='euclidean':
        model = EuclideanTransform
        min_samples = 2
    elif transformation_type=='similarity':
        model = SimilarityTransform
        min_samples = 3
    else:
        raise ValueError('Not a valid transformation_type. Select from "translation", "euclidean" or "similarity" ')
                         
    
    # robustly estimate affine transform model with RANSAC
    model_robust, inliers = ransac((pts1, pts2), model, min_samples=min_samples,
                               residual_threshold=1, max_trials=1000)

    if debug:
        #print(model_robust.translation)
        #print(inliers)
        plt.figure()
        plt.imshow(ref,cmap='gray')
        for i in range(pts1.shape[0]):
            plt.plot(pts1[i,0],pts1[i,1],'r.')
            plt.text(pts1[i,0],pts1[i,1],'{}'.format(i), color='red')
            if inliers[i]:
                plt.plot(pts1[i,0],pts1[i,1],'y.')
                plt.text(pts1[i,0],pts1[i,1],'{}'.format(i), color='yellow')
        plt.title('Reference image')

        plt.figure()
        plt.imshow(mov,cmap='gray')
        for i in range(pts1.shape[0]):
            plt.plot(pts2[i,0],pts2[i,1],'r.')
            plt.text(pts2[i,0],pts2[i,1],'{}'.format(i), color='red')
            if inliers[i]:
                plt.plot(pts2[i,0],pts2[i,1],'y.')
                plt.text(pts2[i,0],pts2[i,1],'{}'.format(i), color='yellow')
        plt.title('Moving image')

     
        

    registered_image = warp(moving_image, model_robust, cval=np.nan, output_shape=reference_image.shape) 
    registration_transform = model_robust