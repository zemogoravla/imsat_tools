import numpy as np

#------ USAGE---------------
# img = rectified_left
# tile_height = opt.crop_height
# tile_width = opt.crop_width

# tiles, tile_origins =  tile_image(img, tile_width, tile_height)
# img2 = untile_image(tiles, tile_origins, tile_width, tile_height)
# img2.shape


def tile_image(img, tile_width, tile_height):

    r""" tile_image
    
    Tiles an image (tiles overlap if width not divisible by tile_width or
    height not divisible by tile_height)
    
    Parameters
    ----------
    img : Image, numpy array 
    tile_width : int
    tile_height: int 
    
    Returns
    -------
    tiles: List of (tile_height x tile_width) numpy arrays of same type as img.
    tile_origins: (Nx2) numpy array with the origins (row, col) of the tiles
        
    """
    img_height = img.shape[0]
    img_width = img.shape[1]
    number_of_vertical_tiles = int(np.ceil(img_height / tile_height))
    number_of_horizontal_tiles = int(np.ceil(img_width / tile_width))

    if number_of_vertical_tiles==1:
        total_vertical_overlap = 0
        tile_height = img_height
    else:
        total_vertical_overlap = number_of_vertical_tiles * tile_height - img_height
        tile_vertical_overlap = total_vertical_overlap // (number_of_vertical_tiles - 1)

    if number_of_horizontal_tiles == 1:
        total_horizontal_overlap = 0
        tile_width = img_width
    else:
        total_horizontal_overlap = number_of_horizontal_tiles * tile_width - img_width
        tile_horizontal_overlap = total_horizontal_overlap // (number_of_horizontal_tiles - 1)



#     print(img_height, img_width)
#     print(tile_height, tile_width)
#     print(number_of_horizontal_tiles,number_of_vertical_tiles)
#     print(tile_horizontal_overlap, tile_vertical_overlap)

    tiles = []
    tile_origins=np.zeros((number_of_vertical_tiles*number_of_horizontal_tiles,2), dtype=np.int)
    k=0
    for i in range(number_of_vertical_tiles):
        for j in range(number_of_horizontal_tiles):
            if i==0:
                tile_origin_row = 0
            elif i==number_of_vertical_tiles-1:
                tile_origin_row = img_height - tile_height
            else:
                tile_origin_row = i * (tile_height - tile_vertical_overlap)
            if j==0:
                tile_origin_col = 0
            elif j==number_of_horizontal_tiles-1:
                tile_origin_col = img_width - tile_width
            else:
                tile_origin_col = j * (tile_width - tile_horizontal_overlap)

            tile_origins[k,:] = [tile_origin_row, tile_origin_col]
            crop = img[int(tile_origins[k,0]):int(tile_origins[k,0]+tile_height), int(tile_origins[k,1]):int(tile_origins[k,1]+tile_width) ]
            tiles.append(crop)
            
            k += 1
    
    return tiles, tile_origins
    



def untile_image_old(tiles, tile_origins, tile_width, tile_height):
    r""" tile_image
    
    Rebuild an image from the tiles. If tiles overlap, just use the values from one of the tiles
    
    Parameters
    ----------
    tiles: List of (tile_height x tile_width) numpy arrays 
    tile_origins: (Nx2) numpy array with the origins (row, col) of the tiles
    tile_width : int
    tile_height: int 
    
    Returns
    -------
    img: Rebuilt image
    
        
    """
    img_height = tile_origins[-1,0] + tile_height
    img_width =  tile_origins[-1,1] + tile_width
    
    img = np.zeros((img_height, img_width), dtype = tiles[0].dtype)
    for k in range(len(tiles)):
        img[int(tile_origins[k,0]):int(tile_origins[k,0]+tile_height), int(tile_origins[k,1]):int(tile_origins[k,1]+tile_width) ] = tiles[k]
    
    return img


def untile_image(tiles, tile_origins):
    r""" tile_image
    
    Rebuild an image from the tiles. If tiles overlap, weight the contribution
    
    Parameters
    ----------
    tiles: List of (tile_height x tile_width) numpy arrays 
    tile_origins: (Nx2) numpy array with the origins (row, col) of the tiles
    tile_width : int
    tile_height: int 
    
    Returns
    -------
    img: Rebuilt image
        
    """

    tile_height, tile_width = tiles[0].shape[:2]

    img_height = tile_origins[-1,0] + tile_height
    img_width =  tile_origins[-1,1] + tile_width
    
    # when overlapping, the pixels of the inside of the tile are more important than 
    # the pixels near the border
    from scipy import ndimage
    tile_border = np.zeros((tile_height, tile_width))
    tile_border[1:-1,1:-1] = 1
    edt = ndimage.distance_transform_edt(tile_border) 
    tile_weight = edt + 0.01 
    assert(np.sum(tile_weight==0)==0)
    
    
    img = np.zeros((img_height, img_width))
    img_weight = np.zeros((img_height, img_width))
    
    for k in range(len(tiles)):
        img[int(tile_origins[k,0]):int(tile_origins[k,0]+tile_height), int(tile_origins[k,1]):int(tile_origins[k,1]+tile_width) ] += tiles[k] * tile_weight
    
        img_weight[int(tile_origins[k,0]):int(tile_origins[k,0]+tile_height), int(tile_origins[k,1]):int(tile_origins[k,1]+tile_width) ] += tile_weight
        
        
    return (img/img_weight).astype(tiles[0].dtype)


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    tile_width = 5
    tile_height = 5
    I = np.random.random((11,11))+20

    tiles, tile_origins = tile_image(I, tile_width, tile_height)
    print(len(tiles))

    J = untile_image(tiles, tile_origins)


    print(np.min(I-J), np.max(I-J))

    plt.figure()
    plt.imshow(I)
    plt.figure()
    plt.imshow(J)
    plt.show()