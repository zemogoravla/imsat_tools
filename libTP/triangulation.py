#!/usr/bin/env python
# coding: utf-8

# # TP Triangulation
# 
# The objective of this lesson is to study the creation and processing of 3D
# models computed from satellite images.  The extraction of 3D points from
# image correspondences is called *triangulation*.  The processing of generic
# 3D point clouds is a very general problem, out of the scope of the present
# course.  Here, we will content ourselves with the much simpler 2.5D models
# that, in the context of geographic imaging, are called *digital elevation
# models* (D.E.M.).
# 
# We cover the following topics:
# * How to find the vertical direction of an image
# * How to obtain a 3D point from a match between two images
# * How the precision of the point varies according to the baseline
# * Computation of a dense point cloud
# * Computation of a D.E.M. from a point cloud
# * Visualization, ~~filtering, interpolation, registration~~ and fusion of D.E.M.
# 
# 
# #### Instructions
# To solve this TP, answer the questions below. Then, clear all the output cells using the menu option **Cell->All Output->Clear** and export the notebook with your answers using  the menu option **File->Download as->Notebook**. Send the resulting *.ipynb* file by email to [enric.meinhardt@cmla.ens-cachan.fr](mailto:enric.meinhardt@cmla.ens-cachan.fr) with subject "Report TP of Name SURNAME", by next week. You will receive an acknowledgement of receipt.
# 
# There are **3 Questions** in the notebook and corresponding code cells to fill-in with your answers.  There are also a few **Exercices** whose solution is already given, but should be understood.
# 
# <!--
# ## Overview of notations
# 
# $(x,y)$, $(i,j)$ pixel coordinates in the domain of an image
# 
# $(\lambda,\theta,h)$ latitude, longitude, height of a 3D point
# 
# $(e,n,h)$ easting, northing, height of a 3D point (the UTM zone is implicit)
# 
# $A, B, \ldots$ gray-level images
# 
# $A(x,y)$ pixel value at coordinates $(x,y)$ of image $A$
# 
# $P_A(\lambda,\theta,h), L_A(x,y,h)$ projection and localization functions
# of image $A$
# 
# $u, v, \ldots$ raster images representing digital elevation models in
# meters
# 
# $u(i,j)$ value of $u$ at the pixel $(i,j)$
# 
# $u(e,n)$ height of the point at geographic coordinates $(e,n)$
# 
# $u(\lambda,\theta)$ height of the point at geographic coordinates
# $(\lambda,\theta)$
# -->

# In[1]:


# # setup the notebook environement
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'notebook')

# general imports
import numpy as np                   # numeric linear algebra
from scipy import ndimage            # only for ndimage.affine_transform
import matplotlib.pyplot as plt      # plotting

# imports specific to this course
import utils          # IO and conversion tools (from TP1)
import vistools       # display tools (from TP1)
import rectification  # rectification tools (from TP2)
import stereo         # stereo matching tools (from TP4)

# display hacks
np.set_printoptions(linewidth=80)


# ## The Tokyo dataset
# 
# For this session we use the Tokyo series of 23 Pléiades images, acquired
# during a single orbit.  Notice that the satellite has to rotate very fast
# and very accurately to point the camera towards the city as it flies all over
# it in a few seconds.

# In[2]:


import requests
import bs4
import os
proxies = {'https': 'httpproxy.fing.edu.uy:3128', 'http': 'httpproxy.fing.edu.uy:3128' }

def is_absolute(url):
    return bool(requests.utils.urlparse(url).netloc)

# extension ='TIF'
# urls=['https://iie.fing.edu.uy/~agomez/', 'https://www.google.com/', 'http://boucantrin.ovh.hw.ipol.im:9861/20130103/']
# url=urls[2]
# r = requests.get(url)
# soup = bs4.BeautifulSoup(r.text, 'html.parser')
# files = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith(extension)]
# folders = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith('/')]

# files_urls = [f if is_absolute(f) else os.path.join(url, os.path.basename(f)) for f in files]
# folders_urls = [f if is_absolute(f) else os.path.join(url, os.path.basename(f.rstrip('/'))) for f in folders]



# # print(files_urls,folders_urls)


# # In[3]:


# # list the tiff images available in the remote folder
# proxies = {'https': 'httpproxy.fing.edu.uy:3128', 'http': 'httpproxy.fing.edu.uy:3128' }
# proxies = {}
# myimages = utils.find('http://boucantrin.ovh.hw.ipol.im:9861/20130103/', 'TIF')
# #print(f"Found {len(myimages)} images")
# print("Found {len(myimages)} images")
# myimages[0:4]


# # In[56]:


# print("Found {} images".format(6))


# # In[ ]:


# # keep only the panchromatic (P) images, discard the multispectral (MS)
# myimages = [x for x in myimages if "_P_" in x]
# myimages[0:4]


# In[ ]:





# In the exercices below we propose to use a fixed area of interest around the Skytree tower.  Optionally, you can select a different scene of your choice, using the `clickablemap` function from TP1.

# In[6]:


# # create a map widget
# m = vistools.clickablemap(zoom=10)

# # display the footprint polygons of each image on this map
# for f in myimages:
#     footprint = utils.lon_lat_image_footprint(f)
#     m.add_GeoJSON(footprint)

# # center the map on the center of the last footprint
# m.center = np.mean(footprint['coordinates'][0], axis=0).tolist()[::-1]
# display(m)


# ## Finding the vertical direction

# The first question is just for *warming up* using the RPC and image extraction
# functions from the previous TPs.
# 
# **Question 1.** Implement the 'crop_vertical' function.  This function
# should extract the requested *Area of interest* (AOI) from an image and rotate it so that the vertical direction in space points upwards in the rotated image domain.  You
# can compute the vertical direction by evaluating the RPC functions on 2
# points at different heights.  The structure of this function is already given, you only need to complete the 'find_vertical_direction' function.
# 
# Apply your function to an area around Skytree Tower (coordinates $35.710139, 139.810833$) or Tokyo Tower (coordinates $35.658611, 139.745556$) to verify that the building is indeed vertical.

# In[6]:


# # coordinates around Sky Tree's base (the second tallest building in the world)
# aoi_skytree = {'type': "Polygon", 'coordinates': [[
#     [139.808, 35.7119],
#     [139.807, 35.7078],
#     [139.815, 35.7078],
#     [139.815, 35.7123],
#     [139.808, 35.7119] 
# ]]
# }


# # In[7]:


# # crop the region of interest on the first image
# crop, offset_x, offset_y = utils.crop_aoi(myimages[0], aoi_skytree)
# rpc = utils.rpc_from_geotiff(myimages[0])


# # In[8]:


# # display the obtained crop
# # Notice that the image is not well oriented and is difficult to interpret
# vistools.display_image(crop)
# vistools.display_image(crop/8)
# vistools.display_image(utils.simplest_color_balance_8bit(crop))


# In[9]:


# code for Question 1
def find_vertical_direction(rpc):
    """
    Return the vertical direction associated to an image
    
    Arguments:
        rpc: RPC function of the image
        
    Return:
        p,q : the vertical direction in the image domain (a normalized 2D vector)
    """
    
    # QUESTION 1 : WRITE THE CODE FOR THIS FUNCTION
    
    # hints:
    # 1. select a base height
    base_height = 0
    y = rpc.row_offset
    x = rpc.col_offset
    # 2. localize the resulting point in space
    lon, lat = rpc.localization(x, y, base_height)
    # 3. project a point that is just 10 meters higher, into the original image domain
    x2, y2 = rpc.projection(lon, lat, base_height+10)
    # 4. return the normalized displacement vector between the two points
    disp = np.array([x2-x, y2-y])
    disp = disp/(np.linalg.norm(disp))
    
    return disp[0], disp[1]
    

# build a rotation matrix that sets the 3D vertical direction upwards
def build_verticalizing_rotation(rpc, shape):
    p, q = find_vertical_direction(rpc)                   # cosine and sine
    p, q = -q, p                                          # ndimage convention for rows/cols
    x, y = shape[1]/2, shape[0]/2                         # center of rotation (middle of the image domain)
    T = np.array([[ 1,  0, -x], [0,  1, -y], [0, 0, 1]])  # translate (x,y) to the origin
    R = np.array([[ p, q,  0], [-q,  p,  0], [0, 0, 1]])  # rotate by the requested angle
    R = np.linalg.inv(T) @ R @ T                          # full rotation matrix
    return R

# like crop_aoi, but rotates the image after cropping
def crop_vertical(image, aoi, base_h=0):
    rpc   = utils.rpc_from_geotiff(image)
    a,_,_ = utils.crop_aoi(image, aoi, base_h)
    R     = build_verticalizing_rotation(rpc, a.shape)
    Ra    = ndimage.affine_transform(a.T, np.linalg.inv(R)).T
    return Ra


# In[10]:


# # compare the crop of an image with and without the verticalizing rotation
# base_h = 45
# a,_,_ = utils.crop_aoi(myimages[0], aoi_skytree, base_h)
# b    = crop_vertical(myimages[0], aoi_skytree, base_h)

# qa = utils.simplest_color_balance_8bit(a)
# qb = utils.simplest_color_balance_8bit(b)
# vistools.display_gallery([qa, qb])


# The following code is used to test your implementation of vertical crop.  You should obtain a timeseries of images such that the vertical direction always points upwards.  **Note:** to evaluate your answer to Question 1 we will run the code below and check visually if the sequence of vertical crops is, indeed, vertical.

# In[11]:


# auxiliary function with the same interface as crop_vertical
def crop_rectangular(image, aoi, base_h=0):
    a,_,_ = utils.crop_aoi(image, aoi, base_h)
    return a


# # In[12]:


# # build a timeseries of crops (without rotation)
# crops = [ crop_rectangular(x, aoi_skytree, base_h) for x in myimages ]


# # In[13]:


# # build a timeseries of crops (rotated in the vertical direction)
# vcrops = [ crop_vertical(x, aoi_skytree, base_h) for x in myimages]


# # In[14]:


# # quantize the crops to 8 bits
# q_crops  = [ utils.simplest_color_balance_8bit(x) for x in crops]
# q_vcrops = [ utils.simplest_color_balance_8bit(x) for x in vcrops]


# # In[15]:


# # display the original images
# vistools.display_gallery(q_crops)


# # In[16]:


# # display the rotated images
# vistools.display_gallery(q_vcrops)


# ## Triangulation of a point
# 
# Triangulation consists in finding the position of a 3D point from its projection into two images.  It is also called *intersection* because it can be interpreted as the intersection of two rays in space.  If $x$ is a point in the domain of image $A$ and $x'$ is a point in the domain of image $A'$, the intersection is found by solving the following equation for $h$ : $L_A(x,h)=L_{A'}(x',h)$.  Then, the 3D point of intersection is either $L_A(x,h)$ or $L_{A'}(x',h)$.  This is a system of two equations with a single unknown $h$, so in general it will not have a solution.  In that case, we can define the "solution" by the value of $h$ that minimizes, for example, the error $e(h)=\|L_A(x,h)-L_{A'}(x',h)\|$.  Then, we obtain the 3D point by evaluating the localization function $L_A$.  Thus, the solution is a 3D point that, when projected into image $A$, it falls exactly on $x$, but when projected on image $A'$ it may be a bit far from $x'$.

# **Exercise 2** Implement the `triangulation` function that finds the position of a 3D point given a correspondence between two images.  This function should be reasonably fast, because you will want to apply it to all the pixels of an image to obtain a dense 3D cloud.  We give you a slow, naive implementation based on iterative minimization of the error. You should write a fast implementation based on the affine approximation of the RPC functions developed for TP Stereo. Notice that in this case the triangulation function has a closed, linear form.  Verify that your implementation gives the same results as triangulation_iterative (up to a few centimeters).

# In[17]:


# evaluate the epipolar line between two images at a value of h
def epipolar_correspondence(rpc_A, rpc_B, x, y, h):
    lon, lat = rpc_A.localization(x, y, h)
    return rpc_B.projection(lon, lat, h)


# In[18]:


# slowish implementation of the triangulation, based on iterative approximation
def triangulation_iterative(rpc1, rpc2, x1, y1, x2, y2):
    """
    Triangulate a match between to images
    
    Arguments:
        rpc1, rpc2 : calibration data for each image
        x1, y1 : pixel coordinates in the domain of the first image
        x2, y2 : pixel coordinates in the domain of the second image
        
    Return value: a 4-tuple (lon,lat,h,e)
        lon, lat, h, e : coordinates of the triangulated point, reprojection error
    """
    
    # initial guess for h
    h = rpc1.alt_offset
    hstep = 1
    err = 0
    
    # iteratively improve h to minimize the error
    for i in range(10):
        # two points on the epipolar curve of (x1, y1)
        # are used to approximate it by a straight line
        px, py = epipolar_correspondence(rpc1, rpc2, x1, y1, h)
        qx, qy = epipolar_correspondence(rpc1, rpc2, x1, y1, h + hstep)
        
        # displacement vectors between these two points and with the target
        ax, ay = qx-px, qy-py
        bx, by = x2-px, y2-py
        
        # projection of the target into the straight line
        l = (ax*bx + ay*by) / (ax*ax + ay*ay)   
        rx, ry = px+l*ax, py+l*ay
        
        # error of this projection
        err = np.hypot(rx - x2, ry - y2)
        
        # new value for h
        h = h + l * hstep
        
        # stop if l becomes too small (max 2 or 3 iterations are performed in practice)
        if np.all(np.fabs(l) < 1e-3):
            break
    
    lon, lat = rpc1.localization(x1, y1, h)
    return lon, lat, h, err


# In[18]:


# invert the affine map represented by a 2x4 matrix
def invert_affine_map(A):
    X = np.vstack([A , [0,0,1,0] ])
    B = np.linalg.inv(X)
    return B[0:3]
                   

# code for exercice 2
def triangulation_affine(PA, PB, x1, y1, x2, y2):
    """
    Triangulate a (list of) match(es) between two images of affine cameras.

    Arguments:
        PA, PB : affine (projection) camera matrices of the two images
        x1, y1 : pixel coordinates in the domain of the first image
        x2, y2 : pixel coordinates in the domain of the second image

    Return value: a 4-tuple (lon, lat, h, e)
        lon, lat, h, e : coordinates of the triangulated point(s), reprojection error
    """
    
    # build projection and localization matrices as 4x4 homogeneous (x,y,h,1) <-> (lon,lat,h,1)
    PA = np.vstack([ PA[0:2], [0,0,1,0], [0,0,0,1]])  # pick only first two rows
    PB = np.vstack([ PB[0:2], [0,0,1,0], [0,0,0,1]])  # pick only first two rows
    LA = np.linalg.inv(PA)  # inverse of a 4x4 matrix
    
    # affine epipolar correspondence
    E = PB @ LA
    
    # Now, the linear system E * (x1, y1, h, 1) = (x2, y2, h, 1)
    # has two different equations and one unknown h.  We solve it by 
    # least squares (a simple projection, in this case).
    
    # give names to the 8 non-trivial coefficients of E
    a, b, p, r = E[0]
    c, d, q, s = E[1]
    
    # coefficients of the affine triangulation
    # (obtained by solving manually the normal equation)
    f = [-p*a - q*c, -p*b - q*d, p, q, -p*r - q*s ] / (p*p + q*q)
    
    # apply the triangulation (first use of the input points)
    h = f[0]*x1 + f[1]*y1 + f[2]*x2 + f[3]*y2 + f[4]
    
    # finish the computation and return the 4 required numbers (or vectors)
    lon = LA[0,0] * x1 + LA[0,1] * y1 + LA[0,2] * h + LA[0,3]
    lat = LA[1,0] * x1 + LA[1,1] * y1 + LA[1,2] * h + LA[1,3]
    ex = E[0,0] * x1 + E[0,1] * y1 + E[0,2] * h + E[0,3] - x2
    ey = E[1,0] * x1 + E[1,1] * y1 + E[1,2] * h + E[1,3] - y2
    e = ex * ex + ey * ey
    return lon, lat, h, e

# compute the affine approximations and triangulate a list of points
def triangulation_affine_rpc(rpc1, rpc2, x1, y1, x2, y2, base_lon, base_lat, base_h):
    P_A = rectification.rpc_affine_approximation(rpc1, (base_lon, base_lat, base_h))
    P_B = rectification.rpc_affine_approximation(rpc2, (base_lon, base_lat, base_h))
    return triangulation_affine(P_A, P_B, x1, y1, x2, y2)


# # Now that we have the triangulation function, we can produce points in 3D space.  Let us compute the height of the Skytree tower.  https://en.wikipedia.org/wiki/Tokyo_Skytree

# # In[20]:


# # extract the rpcs of all images
# myrpcs = [ utils.rpc_from_geotiff(x) for x in myimages ]


# # In[21]:


# # check whether the localization function gives reasonable results
# myrpcs[3].localization(1000, 1000, 10)


# # In[22]:


# # let us work with images 9 and 14
# idx_a = 9
# idx_b = 14


# # In[25]:


# # extract a crop of each image and SAVE THE CROP OFFSETS FOR LATER
# crop_a, offx_a, offy_a = utils.crop_aoi(myimages[idx_a], aoi_skytree)
# crop_b, offx_b, offy_b = utils.crop_aoi(myimages[idx_b], aoi_skytree)
# #print(f"x0_a, y0_a = {offx_a}, {offy_a}")
# #print(f"x0_b, y0_b = {offx_b}, {offy_b}")


# # In[26]:


# # show the two extracted crops
# vistools.display_gallery([utils.simplest_color_balance_8bit(x) for x in [crop_a, crop_b]])


# # In[27]:


# # display the two crops, and a manually selected point
# p = [807, 861]
# q = [944, 38]
# _,f = plt.subplots(2, 1, figsize=(7,9))
# f[0].imshow(utils.simplest_color_balance_8bit(crop_a,0.1), cmap="gray")
# f[1].imshow(utils.simplest_color_balance_8bit(crop_b,0.1), cmap="gray")
# f[0].plot(*p, "ro")
# f[1].plot(*q, "ro")


# # In[31]:


# # triangulate this single match to find (longitude, latitude, height, error_in_pixels)
# x = triangulation_iterative(myrpcs[idx_a], myrpcs[idx_b],
#                             p[0] + offx_a, p[1] + offy_a, q[0] + offx_b, q[1] + offy_b)
# x


# # In[32]:


# get_ipython().run_cell_magic('time', '', '# observe that the iterative triangulation is rather slow\nfor i in range(100):\n    triangulation_iterative(myrpcs[idx_a], myrpcs[idx_b],\n                            p[0] + offx_a, p[1] + offy_a, q[0] + offx_b, q[1] + offy_b)')


# # In[33]:


# # choose a base point for the affine approximation
# base_of_tower = [x[0], x[1], 40]


# # In[34]:


# # triangulate this match using the affine approximation
# triangulation_affine_rpc(myrpcs[idx_a], myrpcs[idx_b],
#                          p[0] + offx_a, p[1] + offy_a, q[0] + offx_b, q[1] + offy_b,
#                          *base_of_tower)


# # **Observation:** Notice that the exact and the affine projection models give essentially the same 3D point.  The affine approximation turns out to be much faster.  In the following, we will study the precision of the affine approximation.

# # ## Triangulation of a track
# # 
# # The following array contains the image coordinates at the top of the Skytree tower (carefully picked by hand), for each of the 23 images in the series.  The Skytree tower has a height of $634m$ above the ground, which at this location is $45m$ above the surface of the WGS84 ellipsoid.

# # In[33]:


# # position of the top of the antenna, selected by hand using an interactive image viewer
# top_of_skytree = np.array([
#     [26542, 6688 ], [27004, 6733 ], [27472, 6792 ], [27959, 6898 ], [28484, 7609 ],
#     [29012, 8122 ], [29524, 8501 ], [30045, 9211 ], [30501, 9782 ], [30829, 10125],
#     [31043, 10296], [31094, 10321], [31001, 10448], [30751, 10933], [30372, 11198],
#     [29902, 11236], [29377, 11445], [28828, 11390], [28297, 11096], [27581, 10991],
#     [27072, 10838], [26600, 10742], [26157, 10605]
# ])


# **Exercice 3** Fill-in a $23\times23$ matrix (minus the diagonal) with the height of the top of the tower, as computed from all the possible pairs of images.  Display the matrix in color and see if it has a visible pattern.  Display the matrix also for the reprojection errors.

# In[34]:


# this function is used to fill the matrix of heights
def top_of_skytree_height_from_pair_exact(i, j):
    if (i == j):
        return 0,0
    else:
        pi = top_of_skytree[i]
        pj = top_of_skytree[j]
        lon, lat, h,e = triangulation_iterative(myrpcs[i], myrpcs[j], *pi, *pj)
        return h,e


# In[35]:


def skytree_top_matrices_exact():
    """
    Compute the height of the skytree tower top using all possible pairs of images
    
    Arguments:
        none, use the global variables "top_of_skytree" and "myrpcs"
        
    Return value:
        Mh, Me : matrices of size 23x23 containing the height and the reprojection error for each pair
    """
    
    Mh = np.zeros((23,23))
    Me = np.zeros((23,23))
    for j in range (23):
        for i in range(23):
            h,e = top_of_skytree_height_from_pair_exact(i,j)
            #print((i,j,h))
            Mh[i,j] = h
            Me[i,j] = e
    return Mh,Me


# # In[36]:


# get_ipython().run_cell_magic('time', '', '# build the matrix of heights using the EXACT triangulation    # THIS CAN TAKE UP TO A MINUTE\nex_Mh, ex_Me = skytree_top_matrices_exact()')


# # In[37]:


# # display the matrices in color to interpret the results
# vistools.display_imshow(ex_Mh,[650,720],cmap="jet")
# plt.title("Height computed for each pair")
# vistools.display_imshow(ex_Me,cmap="jet")
# plt.title("Epipolar error of each pair")


# # In[38]:


def skytree_top_matrices_affine():
    """
    Compute the height of the skytree tower top using all possible pairs of images
    
    Arguments:
        none, use the global variables "top_of_skytree" and "myrpcs"
        
    Return value:
        Mh, Me : matrices of size 23x23 containing the height and the reprojection error for each pair
    """
    
    # first compute a base point for the approximations
    p10 = top_of_skytree[10]
    p11 = top_of_skytree[11]
    base_lon, base_lat, base_h, _ = triangulation_iterative(myrpcs[10], myrpcs[11], *p10, *p11)
    
    # we move the base point a bit
    base_lat += 0.005 # about 500m
    base_lon += 0.005
    base_h = 0
    
    # compute approximate projection matrices for each image
    P = [
        rectification.rpc_affine_approximation(myrpcs[i], (base_lon, base_lat, base_h))
        for i in range(23)
    ]
    
    # fill height and error matrices
    Mh = np.zeros((23,23))
    Me = np.zeros((23,23))
    for j in range (23):
        for i in range(23):
            pi = top_of_skytree[i]
            pj = top_of_skytree[j]
            if (i != j):
                _, _, h, e = triangulation_affine(P[i], P[j], *pi, *pj)
                Mh[i,j] = h
                Me[i,j] = e
    return Mh,Me


# # In[39]:


# get_ipython().run_cell_magic('time', '', '# build the matrix of heights using the AFFINE APPROXIMATED triangulation\nar_Mh, ar_Me = skytree_top_matrices_affine()')


# # In[40]:


# # display the matrices in color to interpret the results
# vistools.display_imshow(ar_Mh,[650,720],cmap="jet")
# plt.title("Height computed for each pair")
# vistools.display_imshow(np.sqrt(ar_Me),cmap="jet")
# plt.title("Epipolar error for each pair")


# # In[41]:


# # show the difference between the exact and the approximate heights
# vistools.display_imshow(np.fmin(0.5,np.abs(ar_Mh - ex_Mh)))
# plt.title("Height difference between exact and\naffine-approximated triangulation,\nin meters")


# # ## Precision of the triangulation depending on the baseline
# # 
# # Notice that the affine triangulation is essentially instantaneous.  This will allow to triangulate a dense set of matches.  But first, we study the stability and precision of the triangulation depending on the pair of images (that have different baselines).
# # 
# # **Exercice 4** Repeat the previous experiment adding gaussian noise of $\sigma=2\mathrm{pixels}$ to the positions of the input points.  The experiment should be run a large number of times (say, one hundred), and the variance of the measure acquired for each image pair must be plotted in matrix form.  Can you identify the pairs of images where the measure is more precise?

# # In[42]:


def skytree_top_variances():
    """
    Compute the height of the skytree tower top using all possible pairs of images
    
    Arguments:
        none, use the global variables "top_of_skytree" and "myrpcs"
        
    Return value:
        Mvar : Mvar[i,j] = estimated variance of height errors with respect to Mh
    """
    
    n = 100 # number of tests
    sigma = 1 # error in pixels
    
    Mvar = np.zeros((23,23))
    
    # first compute a base point for the approximations
    p10 = top_of_skytree[10]
    p11 = top_of_skytree[11]
    base_lon, base_lat, base_h, _ = triangulation_iterative(myrpcs[10], myrpcs[11], *p10, *p11)
    
    # compute approximate projection matrices for each image
    P = [
        rectification.rpc_affine_approximation(myrpcs[i], (base_lon, base_lat, base_h))
        for i in range(23)
    ]
    
    # add gaussian noise to the points before computing the matrices,
    Mh_all = np.zeros((n,23,23))
    for k in range(n):
        Mh = np.zeros((23,23))
        for j in range (23):
            for i in range(23):
                pi = top_of_skytree[i] + sigma * np.random.randn(2)
                pj = top_of_skytree[j] + sigma * np.random.randn(2)
                if (i != j):
                    _, _, h, _ = triangulation_affine(P[i], P[j], *pi, *pj)
                    Mh[i,j] = h
        Mh_all[k,:,:] = Mh
        
    
    return Mh_all.std(axis=0)


# # In[43]:


# # compute the matrix of variances for each image pair
# MM = skytree_top_variances()


# # In[44]:


# # show the precision matrix in color
# vistools.display_imshow(MM, cmap='jet')
# plt.title("Mean-square error of heights computed\nfrom noisy pairs, in meters")


# # ## Dense triangulation
# # 
# # Until now, we have been triangulating a *single* point!
# # To compute a dense 3D point cloud, we will apply the triangulation function to matches computed using the stereo matching algorithms seen on TP-stereo.  For that, we need to rectify the images using the techniques seen on TP-rectification.

# # In[7]:


# # preparation : select two images and an area of interest

# # select two central images
# idx_a = 11
# idx_b = 12

# # coordinates around the Meji Memorial Picture Museum (too large, do not use as it will be very slow),
# # or Skytree Tower
# aoi_meji = {'type': 'Polygon', 'coordinates': [[[139.806985, 35.707857],
#      [139.806985, 35.712143],
#      [139.815306, 35.712143],
#      [139.815306, 35.707857],
#      [139.806985, 35.707857]]]}


# # In[8]:


# m = vistools.clickablemap(zoom=15)
# m.add_GeoJSON(aoi_meji)

# # center the map on the center of the last footprint
# m.center = np.mean(aoi_meji['coordinates'][0], axis=0).tolist()[::-1]
# display(m)


# # In[9]:


# # look at the images before rectification
# a, offx_a, offy_a = utils.crop_aoi(myimages[idx_a], aoi_meji)
# b, offx_b, offy_b = utils.crop_aoi(myimages[idx_b], aoi_meji)
# vistools.display_gallery([utils.simplest_color_balance_8bit(x) for x in [a,b]])


# # In[11]:


# # rectify the images using the techniques of T
# rect1, rect2, S1, S2, dmin, dmax, PA, PB = rectification.rectify_aoi(
#                                     myimages[idx_a], myimages[idx_b], aoi_meji)
# S1, S2, dmin, dmax


# # In[13]:


# # show the rectified images
# vistools.display_gallery([utils.simplest_color_balance_8bit(x) for x in [rect1,rect2]])


# # Now that we have small, rectified images, we can feed them to any standard stereo matching algorithms to find a dense set of matches.  Here, we use the implementation of Semi Global Matching seen on TP Stereo, with standard filtering options.

# # In[14]:


# # compute a disparity map between two rectified images
# LRS, _, _ = stereo.compute_disparity_map(rect1, rect2, dmin-50, dmax+10 )


# # In[15]:


# # display the resulting disparity map
# vistools.display_imshow(LRS, cmap='jet')


# # **Question 2.** Compute a few disparity maps between a central image and other images in the sequence.  How does a high baseline affect the quality of the result? (in terms of precision of the matches and density of valid points).
# # 
# # **Answer.**  ...

# # **Exercice 6** Write a function that transforms a disparty map into 3D point cloud.  For that, you have to invert the rectifying transformations to go back to the coordinates of the original (uncropped) image domain, and then triangulate these correspondences.  If your tringulation function admits vectorial inputs, then this step can be be blazingly fast!

# # In[19]:


# code for exercice 6
def triangulate_disparities(dmap, S1, S2, PA, PB):
    """
    Triangulate a disparity map
    
    Arguments:
        dmap : a disparity map between two rectified images
        S1, S2 : rectifying affine maps (from the domain of the original, full-size images)
        PA, PB : the affine approximations of the projection functions of each image
        
    Return:
        xyz : a matrix of size Nx3 (where N is the number of finite disparites in dmap)
              this matrix contains the coordinates of the 3d points in "lon,lat,h" or "e,n,h"
    """
    
    # 1. unroll all the valid (finite) disparities of dmap into a vector
    m = np.isfinite(dmap.flatten())
    x = np.argwhere(np.isfinite(dmap))[:,1]    # attention to order of the indices
    y = np.argwhere(np.isfinite(dmap))[:,0]
    d = dmap.flatten()[m]
    
    # 2. for each disparity
    # 2.1. produce a pair of points in the original image domain by composing with S1 and S2
    p = np.linalg.inv(S1) @ np.vstack( (x+0, y, np.ones(len(d))) )
    q = np.linalg.inv(S2) @ np.vstack( (x+d, y, np.ones(len(d))) )
    # 2.2. triangulate the pair of image points to find a 3D point (in UTM coordinates)
    lon, lat, h, e = triangulation_affine(PA, PB, p[0,:], p[1,:], q[0,:], q[1,:])
    east, north = utils.utm_from_latlon(lat, lon)
    # 2.3. add this point to the output list
    xyz = np.vstack((east, north, h)).T
    
    # map of triangulation errors
    err = dmap.copy()
    err.flat[m] = e 
    return xyz, err

#AG devuelve las coordenadas en la imagen original de referencia y la secundaria 
def triangulate_disparities_ex(dmap, S1, S2, PA, PB):
    """
    Triangulate a disparity map
    
    Arguments:
        dmap : a disparity map between two rectified images
        S1, S2 : rectifying affine maps (from the domain of the original, full-size images)
        PA, PB : the affine approximations of the projection functions of each image
        
    Return:
        xyz : a matrix of size Nx3 (where N is the number of finite disparites in dmap)
              this matrix contains the coordinates of the 3d points in "lon,lat,h" or "e,n,h"
    """
    
    # 1. unroll all the valid (finite) disparities of dmap into a vector
    m = np.isfinite(dmap.flatten())
    x = np.argwhere(np.isfinite(dmap))[:,1]    # attention to order of the indices
    y = np.argwhere(np.isfinite(dmap))[:,0]
    d = dmap.flatten()[m]
    
    # 2. for each disparity
    # 2.1. produce a pair of points in the original image domain by composing with S1 and S2
    p = np.linalg.inv(S1) @ np.vstack( (x+0, y, np.ones(len(d))) )
    q = np.linalg.inv(S2) @ np.vstack( (x+d, y, np.ones(len(d))) )
    # 2.2. triangulate the pair of image points to find a 3D point (in UTM coordinates)
    lon, lat, h, e = triangulation_affine(PA, PB, p[0,:], p[1,:], q[0,:], q[1,:])
    east, north = utils.utm_from_latlon(lat, lon)
    # 2.3. add this point to the output list
    xyz = np.vstack((east, north, h)).T
    
    # map of triangulation errors
    err = dmap.copy()
    err.flat[m] = e 
    
    return xyz, err, p.T, q.T
    
# In[50]:


# # triangulate the disparities of the previously computed case
# xyz, err = triangulate_disparities(LRS, S1, S2, PA, PB)
# xyz.shape


# # In[51]:


# # compute the UTM bounding box of the obtained 3D points
# min_e, min_n = np.min(xyz,axis=0)[0:2]
# max_e, max_n = np.max(xyz,axis=0)[0:2]

# print((min_e, max_e, max_e-min_e))
# print((min_n, max_n, max_n-min_n))


# # Now, if all went well, you should be able to display a 3D point cloud using the `display_cloud` function below

# # In[52]:


# # open the POTREE visualizer
# vistools.display_cloud(xyz)


# ## Creation of a D.E.M.
# 
# Point clouds in 3D are beautiful, but in a geographical context it is often useful and easier to work with 2.5D models, called digital elevation models (DEM).  We can build a DEM by projecting a point cloud into a fixed square grid in UTM coordinates and accumulate into each square all the 3D points that fall into it.
# 
# 
# **Exercice 7**  Write a function that projects a 3D point cloud into a DEM.  This function receives as input the desired resolution of the DEM.

# In[27]:


from numba import jit

# this function accumulates all the 3D points into a 2.5D grid
# it computes the maximum height (or the average height, by a small change)
@jit
def reducemax( w,h,  ix, iy, z ):
    D_sum = -np.ones((h,w))*np.inf
    D_cnt = np.zeros((h,w))
    for t in range(len(ix)):
        ty = iy[t]
        tx = ix[t]
        if tx >=0 and ty >= 0 and tx < w and ty < h:
            D_sum[ty,tx] = max(D_sum[ty,tx], z[t])
            D_cnt[ty,tx] += 1         
#    D_sum /= D_cnt  # needed for computing average

    return D_sum 

#AG
@jit
def reducemean( w,h,  ix, iy, z ):
    D_mask = np.zeros((h,w), dtype=np.bool)
    D_sum = np.zeros((h,w))
    D_cnt = np.zeros((h,w))
    for t in range(len(ix)):
        ty = iy[t]
        tx = ix[t]
        if tx >=0 and ty >= 0 and tx < w and ty < h:
            D_mask[ty,tx] = True
            D_sum[ty,tx] += z[t]
            D_cnt[ty,tx] += 1         
    D_sum[D_mask] /= D_cnt[D_mask]  # compute average
    D_sum[~D_mask] = np.inf         # set to inf the unfilled pixels
    return D_sum 

def project_cloud_into_utm_grid(xyz, emin, emax, nmin, nmax, resolution=1):
    """
    Project a point cloud into an utm grid to produce a DEM
    The algorithm is the simplest possible: just average all the points that fall into each square of the grid.
    
    Arguments:
        xyz : a Nx3 matrix representing a point cloud in (easting,northing,h) coordinates
        emin,emax,nmin,nmax : a bounding box in UTM coordinates
        resolution : the target resolution in meters (by default, 1 meter)
        
    Return:
        dem : a 2D array of heights in meters
    """
    
    # width and height of the image domain
    w = int(np.ceil((emax - emin)/resolution))
    h = int(np.ceil((nmax - nmin)/resolution))
    
    # extract and quantize columns
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    
    ix = np.asarray((x - emin)/resolution, dtype="int")
    iy = np.asarray((nmax - y)/resolution, dtype="int")
   
    dem = reducemax (w,h,  ix, iy, z )
    #dem = reducemean (w,h,  ix, iy, z )
    
    return dem

def dem_from_xyz(xyz, resolution=1):
    min_e, min_n = np.min(xyz,axis=0)[0:2] - resolution
    max_e, max_n = np.max(xyz,axis=0)[0:2] + resolution
    return project_cloud_into_utm_grid(xyz, min_e, max_e, min_n, max_n, resolution)


# # In[28]:


# # compute the DEM from the 3D point cloud obtained above
# dem = dem_from_xyz(xyz, resolution=1)


# # In[29]:


# # visualize the DEM using a color palette
# # see other palettes here: https://matplotlib.org/examples/color/colormaps_reference.html
# vistools.display_imshow(dem, [25,100], cmap="terrain")


# # If you have reached this point, you have produced a high-resolution D.E.M. from a pair of satellite images.  This is no small feat!  You have repeated in a single day a program that took us many years!
# # 
# # ## Multi-view stereo
# # 
# # Now let us see how can we merge the D.E.M. computed from several pairs of images.

# # In[32]:


# function that computes a D.E.M. from a pair of images
def dem_from_pair(img_a, img_b, aoi, resolution=1):
    
    # run the whole pipeline
    R1, R2, S1, S2, dmin, dmax, PA, PB = rectification.rectify_aoi(img_a, img_b, aoi)
    #print(f"dmin,dmax = {dmin},{dmax}")
    LRS, _, _ = stereo.compute_disparity_map(R1, R2, dmin-10, dmax+10, cost="census")
    #print(f"done computing disparity map")
    xyz, _ = triangulate_disparities(LRS, S1, S2, PA, PB)
    emin, emax, nmin, nmax = utils.utm_bounding_box_from_lonlat_aoi(aoi)
    dem = project_cloud_into_utm_grid(xyz, emin, emax, nmin, nmax, resolution)
    
    return dem


# # In[33]:


# # compute DEM from images 11 and 12
# dem1 = dem_from_pair(myimages[11], myimages[12], aoi_meji)


# # In[34]:


# # compute DEM from images 12 and 13
# dem2 = dem_from_pair(myimages[12], myimages[13], aoi_meji)


# # In[35]:


# # compute DEM from images 13 and 14
# dem3 = dem_from_pair(myimages[13], myimages[14], aoi_meji)


# # In[36]:


# # visualize the computed DEMs
# vistools.display_imshow(dem1,cmap="terrain")
# vistools.display_imshow(dem2,cmap="terrain")
# vistools.display_imshow(dem3,cmap="terrain")


# # In[37]:


# # compute the average of the three
# dem123 = (dem1 + dem2 + dem3)/3


# # In[38]:


# # view the three DEM and their average
# vistools.display_gallery([2*dem1, 2*dem2, 2*dem3, 2*dem123])


# Notice that the average of the three DEM is only computed at the points where all the three DEMs are defined.  Thus, by merging more and more images we will obtain less and less dense DEM.
# 
# **Question 3.** Write a function that merges several DEMs into a single one, by computing the average (or the median) of the all the _avaliable_ heights at each point.  Verify that by merging more and more images you obtain denser and denser DEMs.

# In[39]:


def dem_fusion(dems):
    """
    Merge a list of several DEMs into a single one
    
    Arguments:
        dems: a list of 2D arrays of the same size
        
    Return:
        dem : a 2D array
    """
    
    # WRITE THE CODE OF QUESTION 2 HERE
    array_of_dems = np.zeros((dems[0].shape[0], dems[0].shape[1], len(dems)))
    for i in range(len(dems)):
        array_of_dems[:,:,i] = dems[i]
    
    array_of_dems[np.isinf(array_of_dems)]=np.nan
    dem_fused = np.nanmean(array_of_dems,axis=2)
    
    
    
    return dem_fused

# # view the three DEM and their average
# dem_fused = dem_fusion([dem1, dem2, dem3])

# vistools.display_gallery([2*dem1, 2*dem2, 2*dem3, 2*dem_fused])


# **Question 4. (BONUS)**  Write a function that "elevates" a DEM into a 3D point cloud, and visualize the result of your dem_fusion as a 3D point cloud.

# In[40]:


import utm
def dem_elevate_to_3d(dem, emin, emax, nmin, nmax):
    """
    Elevate a DEM in UTM coordinates into a 3D point cloud
    
    Arguments:
        dem: a 2D array representing a DEM
        emin,emax,nmin,mnax: the corresponding UTM bounding box
        
    Return:
        xyz : a 2D array of size Nx3, representing a 3D point cloud
    """
    # WRITE THE CODE OF QUESTION 3 HERE
    rows,cols = dem.shape
    cloud = np.zeros((cols*rows,3))
    num_points = 0
    for x in range(cols):
        for y in range(rows):
            easting = emin + x * (emax-emin) / (cols-1)
            northing = nmax - y * (nmax-nmin) / (rows-1)
            z = dem[y,x]
            
            if np.isfinite(z):
                cloud[num_points,:] = [easting,northing,z]
                num_points += 1
    cloud = cloud[:num_points,:]
    return cloud


# In[41]:


# # OPTIONAL: test the "elevation" code
# emin, emax, nmin, nmax = utils.utm_bounding_box_from_lonlat_aoi(aoi_meji)
# print(emin, emax, nmin, nmax)
# cloud = dem_elevate_to_3d(dem_fused, emin, emax, nmin, nmax)

# # compute the UTM bounding box of the obtained 3D points
# min_e, min_n, min_z = np.min(cloud,axis=0)
# max_e, max_n, max_z = np.max(cloud,axis=0)

# print((min_e, max_e, max_e-min_e))
# print((min_n, max_n, max_n-min_n))
# print((min_z, max_z, max_z-min_z))


# # In[54]:


# # visualize the computed cloud using the POTREE viewer
# vistools.display_cloud(cloud)


# # In[ ]:



def get_grid_latitudes_longitudes_from_aoi(aoi, resolution=1, **kwargs):
    r""" get_grid_latitudes_longitudes_from_aoi
    
    Returns the arrays of latitudes and longitudes for a certain grid resolution on the aoi .
    
    Parameters
    ----------
    aoi :   dict with 'coordinates' key
            Area of interest (a rectangle is expected)
    resolution : double
            Resolution of the grid. Defaults to 1 (meter)
    **kwargs
        Arbitrary optional keyword argument.
        grid_shape : tuple or list or array of two integers defining the grid shape (rows (latitudes), cols (longitudes))
                     This parameter overrides resolution
    
    Returns
    -------
    latitudes:  1D numpy array
            Array of latitudes of the grid
    longitudes: 1D numpy array
            Array of longitudes of the grid
    """
    
    min_easting,  max_easting, min_northing, max_northing = utils.utm_bounding_box_from_lonlat_aoi(aoi)
    
    if 'grid_shape' in kwargs.keys():
        m,n = kwargs.get('grid_shape')
        
        latitudes = np.arange(start = np.min(np.array(aoi['coordinates'])[0, :, 1]),
                              stop  =  np.max(np.array(aoi['coordinates'])[0, :, 1]),
                              step  = (np.max(np.array(aoi['coordinates'])[0, :, 1])-np.min(np.array(aoi['coordinates'])[0, :, 1]))/m)

        longitudes = np.arange(start =  np.min(np.array(aoi['coordinates'])[0, :, 0]),
                               stop  =  np.max(np.array(aoi['coordinates'])[0, :, 0]),
                               step  = (np.max(np.array(aoi['coordinates'])[0, :, 0])-np.min(np.array(aoi['coordinates'])[0, :, 0]))/n)
    
    else:
    

        Northings = np.arange(min_northing, max_northing, resolution)
        Eastings = np.arange(min_easting, max_easting, resolution)

        zone_number = utm.latlon_to_zone_number(aoi['coordinates'][0][0][1], aoi['coordinates'][0][0][0])
        zone_letter = utm.latitude_to_zone_letter(aoi['coordinates'][0][0][1])
        latitudes = [utm.to_latlon(Eastings[0], Northings[i], zone_number, zone_letter)[0] for i in range(len(Northings))]
        longitudes = [utm.to_latlon(Eastings[i], Northings[0], zone_number, zone_letter)[1] for i in range(len(Eastings))]

        latitudes = np.array(latitudes)
        longitudes = np.array(longitudes)
        
    return latitudes, longitudes



def get_occlusion_mask(img, rpc, aoi, dem, dem_resolution):
    r""" get_occlusion_mask
    
    Returns an image the same size of the dem with a mask of the locations that cannot be 
    seen by the image "img" given the altitudes in the dem
    
    Parameters
    ----------
    img:    satellite view
    rpc:    rpc model corresponding to that view
    aoi :   dict with 'coordinates' key
            Area of interest corresponding to the dem
    dem_resolution : double
            Resolution of the grid of the dem. 
    
    Returns
    -------
    mask:  numpy array (0=visible location, 1=occluded location)
    
    
    ----------USAGE example----------------
    pair_index=1

    left_image_index = pair_list[pair_index][0]
    right_image_index = pair_list[pair_index][1]
    i=right_image_index
    
    Latitudes, Longitudes = get_grid_latitudes_longitudes_from_aoi(aoi, resolution=0.5 )
    image_name = image_filenames_list[i]
    img = utils.readGTIFF(image_name)
    rpc = utils.rpc_from_geotiff(image_name)
    dem = dem_ganet_list[pair_index]
    
    """

    Latitudes, Longitudes = get_grid_latitudes_longitudes_from_aoi(aoi, resolution=dem_resolution )

    
    #meshgrid of longitudes and latitudes
    lon,lat = np.meshgrid(Longitudes, Latitudes, indexing='ij' )
    #flipud the latitudes because the DEM has the north pointing up
    lat = np.flipud(lat)

    lon = np.ravel(lon)
    lat = np.ravel(lat)
    alt = np.ravel(dem)

    # project the locations to the image 
    x,y = rpc.projection(lon, lat, alt)
    
    #integer coordinates
    x_int = np.round(x).astype(int)
    y_int = np.round(y).astype(int)
    coords = np.vstack((x_int,y_int)).T

    condition_in_image = np.logical_and.reduce([x_int>=0,x_int<img.shape[1], y_int>=0, y_int<img.shape[0]])
    
    #sort altitudes in descending order
    sorted_alt_indices = np.argsort(-alt)

    # get the visible coordinates 
    visible_coords, visible_indices = np.unique(coords[sorted_alt_indices], axis=0, return_index=True)
    
    # create the mask: initially all the pixels are occluded (=1)
    mask = np.ones_like(dem, dtype=np.int)

    # set the visible pixels to 0
    mask.ravel()[sorted_alt_indices[visible_indices]] = False 

    # fill the gaps in the occluded pixels (caused by the sampling, dem resolution and rounding)
    import skimage.morphology
    mask_closed = skimage.morphology.closing(mask, selem=np.ones((3,3)))

    #set the undefined pixels to -1
    mask_closed[~np.isfinite(dem)] = -1
    
    
    return mask_closed





