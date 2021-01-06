"""
stereo matching tools

Copyright (C) 2017-2018, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
"""

from __future__ import print_function
import numpy as np

# in case numba jit is not installed
try:
    from numba import jit
except:
    print('WARNING: numba package is not installed')
    def jit(x):
        return x

    
    
# cost volume functions

def censustransform_64(img, cw=5, cp=None, sep=1):
    '''
    Efficiently compute the census transform (CT) of img
    using windows limited to 8 * 8 (cw<=8)
    
    Args:
        img: numpy array containing the input image
        cw:  size of the census window    cw*cw-1 <= 64
        cp:  optional image with centralpixel values of all pixels,
             useful for implementing the modified census transform
        sep: optional control the spacing of the CT samples (default 1)
        
    Returns:
        a numpy array containing the CT at each pixel packed as a uint64 image
    
    derived from: http://stackoverflow.com/questions/38265364/census-transform-in-python-opencv
    '''
    if cw > 8:
        printf('census window cannot be larger than 8x8')
    cw  = min(cw,8)
    hcw = int(cw/2)

    # Initialize 64bit output array
    census = np.zeros(img.shape, dtype='uint64')

    # Center pixels
    if cp is None:
        cp = img

    # Offsets of non-central pixels
    offsets = [(u-hcw, v-hcw) for v in range(cw)
                              for u in range(cw)
                              if not u == hcw == v]
    # Fill census bitstring
    for u,v in offsets:
        census = (census << 1) | (np.roll(img,(-v*sep,-u*sep), axis=(0,1)) >= cp)

    return census




def censustransform(img, cw=5, cp=None, sep=1):
    '''
    Efficiently compute the census transform (CT) of img
    sing windows of size cw * cw
    
    Args:
        img: numpy array containing the input image
        cw:  size of the census window, the transform will have cw*cw-1 bits 
        cp:  optional image with centralpixel values of all pixels,
             useful for implementing the modified census transform
        sep: optional control the spacing of the CT samples (default 1)
        
    Returns:
        a numpy array containing the CT at each pixel packed as as many 
        uint64 image planes as needed to represent the (cw*cw-1) bits
    
    derived from: http://stackoverflow.com/questions/38265364/census-transform-in-python-opencv
    '''

    hcw = int(cw/2)

    # Initialize 64bit output array
    census = None

    # Center pixel values
    if cp is None:
        cp = img

    # Offsets of non-central pixels
    offsets = [(u-hcw, v-hcw) for v in range(cw)
                              for u in range(cw)
                              if not u == hcw == v]

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    for Loffsets in chunks(offsets, 64):
        # Initialize 64bit output array
        Lcensus = np.zeros(img.shape, dtype='uint64')

        # Fill census bitstring
        for u,v in Loffsets:
            Lcensus = (Lcensus << 1) | (np.roll(img,(-v*sep,-u*sep), axis=(0,1)) >= cp)

        if census is None:
            census = Lcensus
        else:  # concatenate along third axis if more than 64 bits are needed
            if Lcensus.ndim==2:
                Lcensus = np.expand_dims(Lcensus,axis=2)
            if census.ndim==2:
                census = np.expand_dims(census,axis=2)
            census = np.dstack((census,Lcensus))
    return census




def countbits(n):
    '''
    Count the number of bits set for all the elements of the numpy array up to uint64
    https://stackoverflow.com/questions/9829578/fast-way-of-counting-non-zero-bits-in-positive-integer
    
    Args:
        n: numpy array of integer type (interpreted as uint64)
        
    Returns:
        numpy array with the number of bits for each element of n 
    '''
    import numpy as np
    if type(n) == np.ndarray:  # force type in case of np.uint32
        n = n.astype(np.uint64)
    else:                      # in case of python number
        n = int(n)
    n = (n & 0x5555555555555555) + ((n & 0xAAAAAAAAAAAAAAAA) >> 1)
    n = (n & 0x3333333333333333) + ((n & 0xCCCCCCCCCCCCCCCC) >> 2)
    n = (n & 0x0F0F0F0F0F0F0F0F) + ((n & 0xF0F0F0F0F0F0F0F0) >> 4)
    n = (n & 0x00FF00FF00FF00FF) + ((n & 0xFF00FF00FF00FF00) >> 8)
    n = (n & 0x0000FFFF0000FFFF) + ((n & 0xFFFF0000FFFF0000) >> 16)
    n = (n & 0x00000000FFFFFFFF) + ((n & 0xFFFFFFFF00000000) >> 32) # This last & isn't strictly necessary.
    return n








# efficinent definition of countbits for a single integer (for numba) 
@jit
def countbitsX(n):
    '''
    Count the number of bits set in the uint64 integer n
    https://stackoverflow.com/questions/9829578/fast-way-of-counting-non-zero-bits-in-positive-integer
    
    Args:
        n: an uint64 integer number
        
    Returns:
       number of bits set in n 
    '''
    n = int(n)  #.astype(np.uint64) # force type in case of uint32
    n = (n & 0x5555555555555555) + ((n & 0xAAAAAAAAAAAAAAAA) >> 1)
    n = (n & 0x3333333333333333) + ((n & 0xCCCCCCCCCCCCCCCC) >> 2)
    n = (n & 0x0F0F0F0F0F0F0F0F) + ((n & 0xF0F0F0F0F0F0F0F0) >> 4)
    n = (n & 0x00FF00FF00FF00FF) + ((n & 0xFF00FF00FF00FF00) >> 8)
    n = (n & 0x0000FFFF0000FFFF) + ((n & 0xFFFF0000FFFF0000) >> 16)
    n = (n & 0x00000000FFFFFFFF) + ((n & 0xFFFFFFFF00000000) >> 32) # This last & isn't strictly necessary.
    return n


@jit
def computeHamming(im1,im2,i,j,k,l):
    '''
    efficient function for computing the Hamming distance between two
    census transformed images
    
    Args:
        im1, im2: numpy arrays (of size [col x rows x channels])  
                  containing census transformed images
        i,j: row/column position of a point in im1
        k,l: row/column position of a point in im2
        
    Returns:
       Hamming distance between  im1[i,j,:] and im2[k,l,:] 
    '''
    s = 0
    assert(len(im1.shape)==3 and len(im2.shape)==3)
    for t in range(im1.shape[2]):
        s += countbitsX( im1[i,j,t] ^ im2[k,l,t])
    return s





def costvolumeSD(im1, im2, dmin=-20, dmax=20): 
    '''
    creates a Squared Difference stereo cost volume
    
    Args:
        im1,im2: numpy arrays containing the stereo pair (im1 is reference)
        dmin,dmax: minimum and maximum disparity to be explored

    Returns:
        numpy array containing cost volume of size [im1.shape[0], im1.shape[0], dmax+1 - dmin]
    '''
    imshape = im1.shape
    
    CV = np.zeros((imshape[0], imshape[1], dmax+1-dmin))
    
    offsets = range(dmin,dmax+1)
    for i in range(len(offsets)):
        sd = (im1 - np.roll(im2,(0,offsets[i]), axis=(0,1)))**2
        if sd.ndim == 3: # in case of color images
            sd = np.sum(sd, axis=2)/sd.shape[2]
        CV[:,:,i] = sd 
        
    return CV


def costvolumeCT(im1, im2, dmin=-20, dmax=20, cw=7): 
    '''
    creates a stereo cost volume for the Census cost: the Hamming 
    distance between the census transformed patches of the two images
    
    Args:
        im1,im2: numpy arrays containing the stereo pair (im1 is reference)
        dmin,dmax: minimum and maximum disparity to be explored
        cw: size of the census transform window (default 7x7)

    Returns:
        numpy array containing cost volume of size [im1.shape[0], im1.shape[0], dmax+1 - dmin]    
    '''
    
    #AG sugerencia de Rafa de cambiar el costo pesando con la binomial
    from scipy.stats import binom
    import scipy.special
    n = cw*cw-1  # numero de bits
    p = 0.5   
    k = np.arange(0, n/2+1)  # el costo maximo es tener n/2 discrepancias
    #binomial_cost = k + binom.cdf(k, n, p) / binom.cdf(n/2, n, p) * n/2 
    #binomial_cost = k
    binomial_cost = np.log(1+k)/np.log(1+n/2)*n/2
    print('costvolumeCT: using log(1+hamming) as cost')
    #binomial_cost = scipy.special.binom(n,k) 
    
    # bug ???  el calculo de hamming da valores mayores a n/2
    # como paliativo extiendo el vector de costos
    binomial_cost = np.pad(binomial_cost, [0,n], mode='edge')
    
    #--------------------------------------------------------------------
    
    
    imshape = im1.shape
    
    CV = np.zeros((imshape[0], imshape[1], dmax+1-dmin), dtype=np.float32)  #para ahorrar: float32
    
    # this creates multi-channel images containing the census bitstrings for each pixel
    im1ct = censustransform(im1, cw)
    im2ct = censustransform(im2, cw)
    
    offsets = range(dmin,dmax+1)
    for i in range(len(offsets)):
        # XOR the bitstrings and count the bits 
        xorbits = im1ct ^ np.roll(im2ct,(0,offsets[i]), axis=(0,1))
        hamming = countbits(xorbits)
        if hamming.ndim == 3:  # in case of multiple bitplanes
            hamming = np.sum(hamming.astype(float), axis=2)
        CV[:,:,i] = binomial_cost[hamming]            # AG original:      CV[:,:,i] = hamming
        
    return CV

def costvolumeCT_original(im1, im2, dmin=-20, dmax=20, cw=7): 
    '''
    creates a stereo cost volume for the Census cost: the Hamming 
    distance between the census transformed patches of the two images
    
    Args:
        im1,im2: numpy arrays containing the stereo pair (im1 is reference)
        dmin,dmax: minimum and maximum disparity to be explored
        cw: size of the census transform window (default 7x7)

    Returns:
        numpy array containing cost volume of size [im1.shape[0], im1.shape[0], dmax+1 - dmin]    
    '''
    
       
    imshape = im1.shape
    
    CV = np.zeros((imshape[0], imshape[1], dmax+1-dmin))
    
    # this creates multi-channel images containing the census bitstrings for each pixel
    im1ct = censustransform(im1, cw)
    im2ct = censustransform(im2, cw)
    
    offsets = range(dmin,dmax+1)
    for i in range(len(offsets)):
        # XOR the bitstrings and count the bits 
        xorbits = im1ct ^ np.roll(im2ct,(0,offsets[i]), axis=(0,1))
        hamming = countbits(xorbits)
        if hamming.ndim == 3:  # in case of multiple bitplanes
            hamming = np.sum(hamming.astype(float), axis=2)
        CV[:,:,i] = hamming
        
    return CV


def aggregateCV(CV, win_w, win_h):
    '''
    filters the cost volume with a rectangular spatial window of 
    size win_w * win_h and uniform weight = 1.0/(win_w * win_h)
    
    Args:
        CV: numpy array containing the cost volume
        win_w,win_h: width and height of the rectangular window

    Returns:
        numpy array containing the filtered cost volume
    '''
    import scipy.signal
    K = np.ones((win_h,win_w))/(win_w*win_h)
    for i in range(CV.shape[2]):
        CV[:,:,i] = scipy.signal.convolve2d(CV[:,:,i], K, mode='same', boundary='symm')
    return CV
    

  


def leftright(offL, offR, maxdiff=1):
    '''
    Filters the disparity maps applying the left-right consistency test
        | offR(round(x - offL(x))) + offR(x)| <= maxdiff

    Args:
        offL, offR: numpy arrays containing the Left and Right disparity maps
        maxdiff: threshold for the uniqueness constraint  

    Returns:
        numpy array containing the offL disparity map, 
        where the rejected pixels are set to np.inf      
    '''
    sh = offL.shape
    X, Y = np.meshgrid(range(sh[1]), range(sh[0]))
    X = np.minimum(np.maximum(X - offL.astype(int), 0), sh[1]-1)
    m = np.abs(offL + offR[Y,X] ) > maxdiff 
    out = offL.copy()
    out[m] = np.Infinity
    return out





def specklefilter(off, area=25, th=0):
    '''
    speckle filter of dispairt map off
    
    Args:
        off:  numpy array with the input disparity map
        area: the surface (in pixels) of the smallest allowed connected component of disparity 
        th:   similarity threshold used to determin if two neighboring pixels have the same value
        
    Returns:
       numpy array with the filtered disparity map, removed points are set to infinity
    '''
    
    @jit
    def find(i,idx):     # finds the root of a dsf
        if idx.flat[i] == i:
            return i
        else:
            ret = find(idx.flat[i],idx)
            #idx.flat[i] = ret    // path compression is useles with idx passed by value
            return ret

    @jit 
    def dsf(D, th=0):    # builds a dsf
        h,w = D.shape[0],D.shape[1]
        idx = np.zeros([h,w],dtype=int)
        for j in range(h):
            for i in range(w):
                idx[j,i] = j*w + i    

        for j in range(h):
            for i in range(w):
                if(i>0):
                    if( abs(D[j,i] - D[j,i-1])<= th ):
                        a = find(idx[j,i],idx)
                        b = find(idx[j,i-1],idx)
                        idx[j,i] = idx[j,i-1]
                        idx.flat[a] = b

                if(j>0):
                    if( abs(D[j,i] - D[j-1,i])<= th ): 
                        a = find(idx[j,i],idx)
                        b = find(idx[j-1,i],idx)
                        idx[j,i] = idx[j-1,i]
                        idx.flat[a] = b

        return idx

    @jit
    def labels(idx):
        h,w=idx.shape[0],idx.shape[1]
        lab = idx*0
        
        for i in range(h*w):
            ind = find(i,idx)
            lab.flat[i] = ind
        return lab

    @jit
    def areas(lab):
        h,w=lab.shape[0],lab.shape[1]
        area = np.zeros(lab.shape,dtype=int)
        LL = np.zeros(lab.shape,dtype=int)
        for i in range(w*h):
            area.flat[lab.flat[i]] += 1 
        for i in range(w*h):
            LL.flat[i] = area.flat[lab.flat[i]]
        return LL

    
    # build the dsf 
    ind = dsf(off, th=th)
    # extract the labels of all the regions
    lab = labels(ind)
    # creat e map where all the regions are tagged with their area
    are = areas(lab)
    # filter the disparity map 
    filtered = np.where((are>area), off, np.inf)
    
    return filtered 




def mismatchFiltering(dL,dR, area=50, tau=1):
    '''
    Applies left-right and speckle filter

    Args:
        dL,dR: are numpy arrays containing the left and right disparity maps
        area: is the minimum area parameter of the speckle filter
        tau: maximum left-right disparity difference
    Returns:
        numpy array containing a filtered version of the left disparity map,
        where rejected pixels are set to infinity
    '''
    dLR = leftright(dL, dR, tau)
    dLRspeckle = specklefilter(dLR,area=area,th=1)
    return dLRspeckle



### SGM RELATED FUNCTIONS ###



#@jit
def filterViterbiV(c, lam=8, P1=1.0, P2=4.0):
    '''
    The function filters the cost volume by computing  
       L_+(p,d) =  C_{p}(d) + \min_{d'}(L_+(p-1,d') +  \lambda V(d, d'))
                       | 0 , if  x=y
       with   V(x,y) = | P1, if |x-y|=1 
                       | P2, otherwise 
    and parameters P1=1 and P2=4.   
                                                             
    Args: 
        cv: numpy array of shape [nodes M, disparities L] containing a cost volume slice
        lam: lambda parameter of the energy

    Returns:
        numpy array containing the filtered costvolume slice
    '''
    #P1=1.0
    #P2=4.0
    sh = c.shape
    M = sh[0]
    L = sh[1]
    S = c.copy().astype(np.float)
    
    for i in range(1,M): # loop over the nodes
        
        ### YOUR CODE HERE ###
            
        minSim1 = np.min(S[i-1,:])  # precompute min of the previous node 
        for l in range(L):   
            minS =  lam * P2   + minSim1   # 0 because of the normalization of the previous node
            for lp in (l-1, l, l+1):
                if lp>=0 and lp<L:
                    newS = S[i-1,lp] + lam * P1 * np.abs(l-lp)
                    if minS > newS:
                        minS = newS
            S[i,l] = S[i,l] + minS
        
        # this normalization removes the min of the previous node 
        #S[i,:] = S[i,:] - np.min(S[i,:]) #normalize
    return S


#@jit
def filterViterbiV_with_segmentation(c,seg, lam=8, P1=1.0, P2=4.0):
    '''
    The function filters the cost volume by computing  
       L_+(p,d) =  C_{p}(d) + \min_{d'}(L_+(p-1,d') +  \lambda V(d, d'))
                       | 0 , if  x=y
       with   V(x,y) = | P1, if |x-y|=1 
                       | P2, otherwise 
    and parameters P1=1 and P2=4.   
                                                             
    Args: 
        cv: numpy array of shape [nodes M, disparities L] containing a cost volume slice
        seg: numpy array of shape [nodes M] containing the segmentation values of the slice
        lam: lambda parameter of the energy

    Returns:
        numpy array containing the filtered costvolume slice
    '''
    #P1=1.0
    #P2=4.0
    sh = c.shape
    M = sh[0]
    L = sh[1]
    S = c.copy().astype(np.float)
    
    # TODO: hacer mas eficiente
#    P2_seg = np.ones_like(seg)*P2
#     for i in range(1,M):
#         if seg[i]==seg[i-1]:
#             P2_seg[i] = P2 * 2
    
    
    for i in range(1,M): # loop over the nodes
        if seg[i]==seg[i-1]:
            P2_seg = P2 * 1e10
        else:
            P2_seg = P2
        
        ### YOUR CODE HERE ###
            
        minSim1 = np.min(S[i-1,:])  # precompute min of the previous node 
        for l in range(L):   
            minS =  lam * P2_seg   + minSim1   # 0 because of the normalization of the previous node
            for lp in (l-1, l, l+1):
                if lp>=0 and lp<L:
                    newS = S[i-1,lp] + lam * P1 * np.abs(l-lp)
                    if minS > newS:
                        minS = newS
            S[i,l] = S[i,l] + minS
        
        # this normalization removes the min of the previous node 
        #S[i,:] = S[i,:] - np.min(S[i,:]) #normalize
    return S



def sgmfilter(CV, lam=8, P1=1.0, P2=4.0):
    '''
    SGM cost volume filtering along 4 directions 
    using the truncated regularity term V(with parameters P1=1,P2=4)
    
    Args: 
        CV: numpy array of size [width, height, disparity] containing the costvolume 
        lam: lambda regularity parameter
        
    Returns:
        numpy array containing the filtered costvolume
    '''
    # compile the filterViterbiV function
    viterbiV = jit(filterViterbiV)
    
    S = np.zeros(CV.shape)
    
    ### YOUR CODE HERE ### 
    
    for i in range(CV.shape[0]):
        fw = viterbiV(CV[i,:,:],lam, P1, P2)
        bw = viterbiV(CV[i,::-1,:],lam, P1, P2)
        S[i,:,:] += fw + bw[::-1]
        
    for i in range(CV.shape[1]):
        fw = viterbiV(CV[:,i,:],lam, P1, P2)
        bw = viterbiV(CV[::-1,i,:],lam, P1, P2)
        S[:,i,:] += fw + bw[::-1]
        
    return S - 3*CV


def sgmfilter_NEW(CV, lam=8, P1=1.0, P2=4.0, eight_directions=False):
    S = np.zeros(CV.shape)
    
    if 1:
        for i in range(CV.shape[0]):
            lin=CV[i,:,:]
            fw = filterViterbiV(lin,lam,P1,P2)
            bw = filterViterbiV(lin[::-1,:],lam,P1,P2)
            S[i,:,:] += fw + bw[::-1]

        for i in range(CV.shape[1]):
            lin = CV[:,i,:]
            fw = filterViterbiV(lin,lam,P1,P2)
            bw = filterViterbiV(lin[::-1,:],lam,P1,P2)
            S[:,i,:] += fw + bw[::-1]
    
    if eight_directions:            
        H,W=CV.shape[0], CV.shape[1]

        if 0:
            # my code... 
            for i in range(-H,W):
                Yj = ([ j for j in range(H) if j+i<W and j+i>=0])
                Xj = ([ j+i for j in range(H) if j+i<W and j+i>=0])
                lin = CV[Yj,Xj,:]
                fw = filterViterbiV(lin,lam,P1,P2)
                bw = filterViterbiV(lin[::-1,:],lam,P1,P2)        
                S[Yj,Xj,:] += fw + bw[::-1]

            for i in range(-H,W):
                Yj = ([ H-1-j for j in range(H) if j+i<W and j+i>=0])
                Xj = ([ j+i for j in range(H) if j+i<W and j+i>=0])
                lin = CV[Yj,Xj,:]
                fw = filterViterbiV(lin,lam,P1,P2)
                bw = filterViterbiV(lin[::-1,:],lam,P1,P2)        
                S[Yj,Xj,:] += fw + bw[::-1]
           
        
        else:
            # SOLUTION BY Simon ERDMANN
            rows, cols = np.indices((H, W))

            for i in range(-H + 1, W - 1):
                fw = filterViterbiV(CV.diagonal(i),lam,P1,P2)
                bw = filterViterbiV(CV.diagonal(i)[::-1,:],lam,P1,P2)
                S[np.diag(rows, i), np.diag(cols, i), :] = S[np.diag(rows, i), np.diag(cols, i), :] + (fw + bw[::-1]).transpose()

            S_fl = np.flipud(S)
            CV_fl = np.flipud(CV)

            for i in range(-H, W):
                fw = filterViterbiV(CV_fl.diagonal(i),lam,P1,P2)
                bw = filterViterbiV(CV_fl.diagonal(i)[::-1,:],lam,P1,P2)
                S_fl[np.diag(rows, i), np.diag(cols, i), :] = S_fl[np.diag(rows, i), np.diag(cols, i), :] + (fw + bw[::-1]).transpose()        
            
            S = np.flipud(S_fl)    

    return S


def sgmfilter_with_segmentation(CV, SEG, lam=8, P1=1.0, P2=4.0):
    '''
    SGM cost volume filtering along 4 directions 
    using the truncated regularity term V(with parameters P1=1,P2=4)
    
    Args: 
        CV: numpy array of size [width, height, disparity] containing the costvolume 
        S: numpy array of size [width, height] containing the segmentation of the reference image 
        lam: lambda regularity parameter
        
    Returns:
        numpy array containing the filtered costvolume
    '''
    # compile the filterViterbiV function
    ViterbiV_with_segmentation = jit(filterViterbiV_with_segmentation)

    S = np.zeros(CV.shape)
    
    ### YOUR CODE HERE ### 
    
    for i in range(CV.shape[0]):
        fw = ViterbiV_with_segmentation(CV[i,:,:], SEG[i,:],lam, P1, P2)
        bw = ViterbiV_with_segmentation(CV[i,::-1,:], SEG[i,:],lam, P1, P2)
        S[i,:,:] += fw + bw[::-1]
        
    for i in range(CV.shape[1]):
        fw = ViterbiV_with_segmentation(CV[:,i,:], SEG[:,i], lam, P1, P2)
        bw = ViterbiV_with_segmentation(CV[::-1,i,:], SEG[:,i],lam, P1, P2)
        S[:,i,:] += fw + bw[::-1]
        
    return S - 3*CV


@jit
def VfitMinimum(v):
    ''' 
    interpolates the position of the subpixel minimum 
    given the samples (v) around the discrete minimum
    according to the Vfit method illustrated below
    
      v[0] * 
            \
             \
              \      *  v[2] 
               \    /
           v[1] *  /
                 \/
                  v_min
                  ^
       ____|____|_|__|____|_
           -1   0 xm 1  

    Returns:
        position of the minimum x_min (xm) and its value v_min  
       
       
    '''
    # if we can't fit a V in the range [-1,1] then we leave the center
    if( (v[1] > v[0])  and (v[1] > v[2]) ) :
        v_min = v[1]
        x_min = 0
        return x_min, v_min
    
    # select the maximum slope
    slope = v[2] - v[1]
    if ( slope < (v[0] - v[1]) ):
        slope = v[0] - v[1]

    # interpolate
    x_min = (v[0] - v[2]) / (2*slope);
    v_min = v[2] + (x_min - 1) * slope;
    return x_min, v_min


  

# define the trivial winner-takes-all function
def WTA(CV):
    '''computes the winner takes all of the cost volume CV'''
    return  np.argmin(CV,axis=2)



@jit
def VfitWTA(CV, min_disp, min_cost):
    '''computes the subpixel refined winner takes all of the cost volume CV'''
    sh = CV.shape

    for y in range(sh[0]):
        for x in range(sh[1]):
            md = int(min_disp[y,x])
            #can only interpolate if the neighboring disparities are available
            if md > 0 and md < sh[2]-1:
                dmd, mc = VfitMinimum([float(CV[y,x,md-1]), float(CV[y,x,md]), float(CV[y,x,md+1])])
                min_disp[y,x] = dmd +md
                min_cost[y,x] = mc

    return min_disp, min_cost



def stereoSGM(im1,im2,dmin,dmax,lam=10,cost='census',cw=3, win=1, subpix_refine=False):
    '''
    computes the disparity map from im1 to im2 using SGM
    and optionally post filters the CV with a window of size (win X win).
    cost selects the matching cots: sd, or census

    Args:
        im1,im2: numpy arrays containing the stereo pair (im1 is reference)
        dmin,dmax: minimum and maximum disparity to be explored
        lam: 
        cost: type of cost volume can be: 'sd' or 'census'
        cw:  census window size, used when cost='census'
        win: aggregateCV window size (set to 1 to disable)
        subpix_refine: activates the Vfit subpixel refinement (default False)
        
    Returns:
        numpy array containing the disparity map
    '''
    import time
    start_time = time.time()

    # generate the cost volume
    if cost=='sd':
        CV=costvolumeSD(im1, im2, dmin, dmax)
    else:
        CV=costvolumeCT(im1, im2, dmin, dmax, cw=7)

    print ('t={:2.4f} done building CV'.format(time.time() - start_time))
    CV = sgmfilter(CV,lam)         # SGM

    print ('t={:2.4f} done sgmfilter'.format(time.time() - start_time))

    if win>1:
        CV = aggregateCV(CV,win,win)

    if subpix_refine:  # WTA 
        d,_ = VfitWTA(CV, np.argmin(CV,axis=2).astype(np.float32), np.min(CV,axis=2).astype(np.float32))
    else:
        d = WTA(CV).astype(np.float32)    # i.e. # d= np.argmin(CV,axis=2)

    print ('t={:2.4f} done aggregation and WTA refinement'.format(time.time() - start_time))

    # map from idx to disparity
    ## drange = np.array(range(dmin, dmax+1), dtype=float) # old code
    ## return drange[d]
    return d+dmin

def stereoSGM_with_segmentation(im1,im2,dmin,dmax,lam=10,cost='census',cw=3, win=1, subpix_refine=False, 
                                segmentation = None, segmentation_strategy='filter_cv'):
    '''
    computes the disparity map from im1 to im2 using SGM
    and optionally post filters the CV with a window of size (win X win).
    cost selects the matching cots: sd, or census
    
    AG: takes into account the segmentation of im1

    Args:
        im1,im2: numpy arrays containing the stereo pair (im1 is reference)
        dmin,dmax: minimum and maximum disparity to be explored
        lam: 
        cost: type of cost volume can be: 'sd' or 'census'
        cw:  census window size, used when cost='census'
        win: aggregateCV window size (set to 1 to disable)
        subpix_refine: activates the Vfit subpixel refinement (default False)
        
    Returns:
        numpy array containing the disparity map
    '''
    import time
    start_time = time.time()

    # generate the cost volume
    if cost=='sd':
        CV=costvolumeSD(im1, im2, dmin, dmax)
    else:
        CV=costvolumeCT(im1, im2, dmin, dmax, cw=7)
    
    print ('t={:2.4f} done building CV'.format(time.time() - start_time))
    
#     #-------------------------------------------
#     # SEGMENTACION
#     gradient_disk_size = 2
#     h_minima_th = 3
    
#     #normalize
#     im_np = normalize(im1)

#     # ALGORITHM 3 - step 1
#     # compute the local gradient of the gray image (local maximum-local minimum) on a disk
#     gr_np = gradient(im_np, disk(gradient_disk_size))
    
#     # ALGORITHM 3 - step 2
#     # segment the gray image using watershed
#     # regions are basins starting from local minima of the gradient
#     img_watershed = watershed_segmentation(im_np, gr_np, h_minima_th=h_minima_th)
#     print ('Segmentation:', time.time() - start_time)
    
        
    if segmentation is None:
        raise ValueError('Function sgmfilter_with_segmentation. Must receive a valid segmentation')
        
        
        
    if segmentation_strategy=='filter_cv':
        #-------------------------------------------
        # VERSION DRASTICA
        # Se impone que el costo sea el mismo en cada region de la segmentacion
        #print('CV.shape', CV.shape)
        ''' ESTA ANDA PERO MUY LENTO
        for d in range(CV.shape[2]):
            print('Computing CV mean by regions, disparity level {:04d}/{:04d}'.format(d+1, CV.shape[2]), end='\r', flush=True)
            for i in range(np.max(img_watershed)+1):
                condition = img_watershed==i
                CV[:,:,d][condition] = np.mean(CV[:,:,d][condition])

        print ('CV mean by regions:', time.time() - start_time)
        '''
        
        sbg = jit(sum_by_group)
        avbg = jit(asign_values_by_group)
        
        
        for d in range(CV.shape[2]):
            print('Computing CV mean by regions, disparity level {:04d}/{:04d}'.format(d+1, CV.shape[2]), end='\r', flush=True)
            
            data = CV[:,:,d].ravel()
            groups = segmentation.ravel()
            
            
            values, groups = sbg(data, groups)
            #print('values.shape, groups.shape, np.min(img_watershed), np.max(img_watershed)', values.shape, groups.shape, np.min(img_watershed) , np.max(img_watershed) )
            
            
            sorted_indices = np.argsort(groups)

            sorted_groups = groups[sorted_indices]
            sorted_values = values[sorted_indices]
            
            CV[:,:,d] = np.reshape(sorted_values[segmentation.ravel()-segmentation.min()],CV.shape[:2])   # el -img_watershed.min() es porque los valores del WS van de 1 a max(WS)
            
#             for i in range(len(groups)):
#                 data[data==groups[i]] = values[i]/np.sum(data==groups[i])
            
        print ('CV mean by regions:', time.time() - start_time)
        
       
        
        
        
        
        #print('CV.shape', CV.shape)
        #-------------------------------------------
        
        CV = sgmfilter(CV,lam)         # SGM
        
    elif segmentation_strategy=='modify_P2':
        CV = sgmfilter_with_segmentation(CV,segmentation ,lam)         # SGM affecting P2 with the segmentation
    else:
        raise ValueError('Function sgmfilter_with_segmentation. segmentation_strategy:{} not implemented. Valid startegies: ["filter_cv","modify_P2"]'.format(segmentation_strategy))
    
    

    print ('t={:2.4f} done sgmfilter'.format(time.time() - start_time))

    if win>1:
        CV = aggregateCV(CV,win,win)

    if subpix_refine:  # WTA 
        d,_ = VfitWTA(CV, np.argmin(CV,axis=2).astype(np.float32), np.min(CV,axis=2).astype(np.float32))
    else:
        d = WTA(CV).astype(np.float32)    # i.e. # d= np.argmin(CV,axis=2)

    print ('t={:2.4f} done aggregation and WTA refinement'.format(time.time() - start_time))

    # map from idx to disparity
    ## drange = np.array(range(dmin, dmax+1), dtype=float) # old code
    ## return drange[d]
    return d+dmin


# a generic function to compute disparity maps from two rectified images using SGM
def compute_disparity_map(rect1, rect2, dmin, dmax, cost='census', lam=10, 
                          segmentation_ref=None, segmentation_sec=None, segmentation_strategy='filter_cv'):
    '''
    computes and filters the disparity map from im1 to im2 using SGM
    cost selects the matching cots: sd, or census

    Args:
        im1,im2: numpy arrays containing the stereo pair (im1 is reference)
        dmin,dmax: minimum and maximum disparity to be explored
        cost: type of cost volume can be: 'sd' or 'census'
        lam: lambda is the regularity parameter of SGM

    Returns:
        numpy array containing the filtered disparity map
    '''    
    im1 , im2  = rect1, rect2
    dmin, dmax = int(np.floor(-dmax)), int(np.ceil(-dmin))

    # some reasonable parameters
    #lam = 10  # lambda is a regularity parameter
    cw  = 5   # small census windows are good
    win = 1   # this removes some streaking artifacts
    subpix_refine = True

    if not segmentation_ref is None:
        # compute left and right disparity maps
        dL =  stereoSGM_with_segmentation(im1,im2,dmin,dmax,lam=lam,cost=cost, cw=cw, win=win, subpix_refine=subpix_refine,
                                          segmentation=segmentation_ref, segmentation_strategy=segmentation_strategy)
        dR =  stereoSGM_with_segmentation(im2,im1,-dmax,-dmin,lam=lam,cost=cost, cw=cw, win=win, subpix_refine=subpix_refine,
                                          segmentation=segmentation_sec, segmentation_strategy=segmentation_strategy)
    else:
        # compute left and right disparity maps
        dL =  stereoSGM(im1,im2,dmin,dmax,lam=lam,cost=cost, cw=cw, win=win, subpix_refine=subpix_refine)
        dR =  stereoSGM(im2,im1,-dmax,-dmin,lam=lam,cost=cost, cw=cw, win=win, subpix_refine=subpix_refine)

    # apply mismatch filtering
    LRS = mismatchFiltering(dL, dR, 50)
    
    # minus sign here (due to different disparity conventions)
    return -LRS, -dL, -dR


#----------------------------------
'''
AG SEGMENTATION
'''

import skimage
from skimage.morphology import disk
from skimage.filters.rank import gradient
import time

def normalize(im_np):
    '''
    Normalize image to 0..255 range
    '''
    lower = np.percentile(im_np, 1)
    upper = np.percentile(im_np, 99)
    im_np = (im_np.astype(float) - lower) / (upper - lower)
    im_np[im_np < 0] = 0
    im_np[im_np > 1] = 1
    return (im_np * 255).astype('uint8')
    
# called in ALGORITHM 3 - step 2
def watershed_segmentation(im_np, gr_np, h_minima_th=5):
    '''
    Watershed segmentation of the image starting from markers defined by the
    h_minima of the gradient
    '''
    start_time = time.time()
    hmin_np = skimage.morphology.h_minima(gr_np, h_minima_th) > 0
    print ('H-minima:', time.time() - start_time)
    markers_np = skimage.morphology.label(hmin_np).astype('int32')
    print ('Markers:', time.time() - start_time)
    markers_np = skimage.morphology.watershed(gr_np, markers_np)
    print ('Watershed:', time.time() - start_time)

    return markers_np
    

    
def colorize_image_segmentation(img_segmentation):
    colorized_img_segmentation = np.zeros((img_segmentation.shape[0],img_segmentation.shape[1],3) , dtype=np.uint8)
    print()
    img_max_value = np.max(img_segmentation)
    random_lut = np.random.randint(0,256,[img_max_value+1,3])
    for f in range(img_segmentation.shape[0]):
        for c in range(img_segmentation.shape[1]):
            colorized_img_segmentation[f,c,:] = random_lut[img_segmentation[f,c],:]
            
    return colorized_img_segmentation


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


def asign_values_by_group(data, groups, values):
    for i in range(len(groups)):
        data[data==groups[i]]=values[i]
        
    return data
