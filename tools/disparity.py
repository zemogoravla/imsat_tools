import os
import sys
import errno
import datetime

import numpy as np
import matplotlib.pyplot as plt
import skimage
import subprocess


'''
  usage: mgm_multi  [-r dmin -R dmax] [-m dminImg -M dmaxImg] [-O NDIR: 2, (4), 8, 16] u v out [cost [backflow]]
    [-P1 (8) -P2 (32)]: sgm regularization parameters P1 and P2
    [-p PREFILT(none)]: prefilter = {none|census|sobelx|gblur} (census is WxW)
    [-t      DIST(ad)]: distance = {census|ad|sd|ncc|btad|btsd}  (ncc is WxW, bt is Birchfield&Tomasi)
    [-truncDist (inf)]: truncate distances at nch*truncDist  (default INFINITY)
    [-s  SUBPIX(none)]: subpixel refinement = {none|vfit|parabola|cubic}
    [-aP1         (1)]: multiplier factors of P1 and P2 when
    [-aP2         (1)]:    \sum |I1 - I2|^2 < nch*aThresh^2
    [-aThresh     (5)]: Threshold for the multiplier factor (default 5)
    [-S scales    (3)]: Number of scales
    [-Rd  fname]: right disparity map
    [-Rc  fname]: right cost map
    [-wl  fname]: regularization weights for the left disparity map
    [-wr  fname]: regularization weights for the right disparity map
    [-confidence_costL      fnameL -confidence_costR      fnameR]: left and right cost confidence maps
    [-confidence_pkrL       fnameL -confidence_pkrR       fnameR]: left and right PKR confidence maps
    [-confidence_consensusL fnameL -confidence_consensusR fnameR]: left and right consensus confidence maps
    [-inputCostVolume filename]: file containing the costvolume of the left image
    ENV: CENSUS_NCC_WIN=3   : size of the window for census and NCC
    ENV: TESTLRRL=1   : lrrl
    ENV: REMOVESMALLCC=0 : remove connected components of disp. smaller than (recomended 25)
    ENV: MINDIFF=-1   : remove disp. inconsistent with minfilter on a window of size CENSUS_NCC_WIN (recommended 1)
    ENV: TSGM=4       : regularity level
    ENV: TSGM_ITER=1  : iterations
    ENV: TSGM_FIX_OVERCOUNT=1   : fix overcounting of the data term in the energy
    ENV: TSGM_DEBUG=0 : prints debug informtion
    ENV: SUBPIX=1     : subpixel steps
    ENV: USE_TRUNCATED_LINEAR_POTENTIALS=0 : use the Felzenszwalb-Huttenlocher
                      : truncated linear potential (when=1). P1 and P2 change meaning
                      : The potential they describe becomes:  V(p,q) = min(P2,  P1*|p-q|)

'''

# ESTE ES EL DE S2P PERO DA PROBLEMAS AL CORRERLO EN UN NOTEBOOK
'''
https://github.com/dsavoiu/kafe/issues/6

"As I understand it, when working in a Jupyter Notebook, the standard input and output streams are replaced by a custom implementation, which is specific to Jupyter/IPython/ipykernel. This is presumably necessary in order to communicate with the Python kernel that is currently active.

This implementation, however, does not use file descriptors directly, so there is no fileno() available for the output stream, which is why the error shows up in Jupyter."

'''
def run_command(cmd, env=os.environ):
    """
    Runs a shell command, and print it before running.

    Arguments:
        cmd: string to be passed to a shell
        env (optional, default value is os.environ): dictionary containing the
            environment variables

    Both stdout and stderr of the shell in which the command is run are those
    of the parent process.
    """
    print("\nRUN: %s" % cmd)
    t = datetime.datetime.now()
    try:
        subprocess.check_call(cmd, shell=True, stdout=sys.stdout,
                              stderr=sys.stderr, env=env)
        print(datetime.datetime.now() - t)

    except subprocess.CalledProcessError as e:
        # raise a custom exception because the CalledProcessError causes the
        # pool to crash
        raise RunFailure({"command": e.cmd, "environment": env, "output":
                          e.output})

def run_command_ag(cmd, env=os.environ):
    """
    Runs a shell command, and print it before running.

    Arguments:
        cmd: string to be passed to a shell
        env (optional, default value is os.environ): dictionary containing the
            environment variables

    Both stdout and stderr of the shell in which the command is run are those
    of the parent process.
    """
    print("\nRUN: %s" % cmd)
    t = datetime.datetime.now()
    try:
        subprocess.call('pwd', shell=True)
        subprocess.call(cmd, shell=True, env = env )
        
#         subprocess.check_call(cmd, shell=True, stdout=sys.stdout,
#                               stderr=sys.stderr, env=env)
        print(datetime.datetime.now() - t)

    except subprocess.CalledProcessError as e:
        # raise a custom exception because the CalledProcessError causes the
        # pool to crash
        raise RunFailure({"command": e.cmd, "environment": env, "output":
                          e.output})
        
def compute_mgm_multi(rectified_ref, rectified_sec, disp_min, disp_max ):
    '''
    Adaptado de s2p (versión de github)
    '''
    MGM_MULTI_EXE = '/home/agomez/Software/satelite/s2p_ultimate/s2p/bin/mgm_multi'
    rectified_ref_filename = 'tmp_rectified_ref.tif'
    rectified_sec_filename = 'tmp_rectified_sec.tif'
    rectified_ldisp_filename = 'tmp_rectified_ldisp.tif'
    rectified_rdisp_filename = 'tmp_rectified_rdisp.tif'
    
    # define environment variables
    env = os.environ.copy()
    
    
    #
    env['REMOVESMALLCC'] = '25' #str(cfg['stereo_speckle_filter'])
    env['MINDIFF'] = '1'
    env['CENSUS_NCC_WIN'] = '5' #str(cfg['census_ncc_win'])
    env['SUBPIX'] = '2'
    # it is required that p2 > p1. The larger p1, p2, the smoother the disparity
    regularity_multiplier = 1.0 #cfg['stereo_regularity_multiplier']
    P1 = 8*regularity_multiplier   # penalizes disparity changes of 1 between neighbor pixels
    P2 = 32*regularity_multiplier  # penalizes disparity changes of more than 1
    #conf = '{}_confidence.tif'.format(os.path.splitext(disp)[0])
    
    
    #save the images
    skimage.io.imsave(rectified_ref_filename, rectified_ref)
    skimage.io.imsave(rectified_sec_filename, rectified_sec)
    
    
    #run the command
    # -O 8 agregado por AG , no está en la versión original de s2p
    run_command_ag('{0} -r {1} -R {2} -S 6 -s vfit -t census -O 8 -Rd {3} {4} {5} {6}'.format(MGM_MULTI_EXE,
                                                                                     disp_min,
                                                                                     disp_max,
                                                                                     rectified_rdisp_filename,
                                                                                     rectified_ref_filename,
                                                                                     rectified_sec_filename,
                                                                                     rectified_ldisp_filename,
                                                                                     ),                env)

    
    # load results
    rectified_ldisp = skimage.io.imread(rectified_ldisp_filename)
    rectified_rdisp = skimage.io.imread(rectified_rdisp_filename)
    
    return rectified_ldisp, rectified_rdisp
    