import os

def get_jax_image_names(image_directory, gt_directory, set_number, image_numbers):
    r"""get_jax_image_names    
    Returns a list of image full filenames and the gt full filename
    """
    image_extension = 'PAN.tif'
    gt_extension = 'DSM.tif'

    image_name_template = 'JAX_{:03d}_{:03d}_PAN.tif'  # set_number, image_number 
    gt_name_template = 'JAX_{:03d}_DSM.tif'  # set_number

    image_filenames = [os.path.join(image_directory, image_name_template.format(set_number, image_number)) 
               for image_number in image_numbers]
    gt_filename = os.path.join(gt_directory, gt_name_template.format(set_number))

    return image_filenames, gt_filename

def get_all_possible_pairs_from_list(L):
    r"""get_all_possible_pairs_from_list    
    Returns a list of all the possible pairs of the list L.
    """
    # https://stackoverflow.com/questions/18201690/get-unique-combinations-of-elements-from-a-python-list
    from itertools import combinations
    all_pairs = [list(comb) for comb in combinations(L,2)]
    return all_pairs



