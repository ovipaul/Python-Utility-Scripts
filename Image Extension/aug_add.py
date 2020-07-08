
import numpy as np
from numba import jit

@jit(nopython=True)
def row_aug_add(org_img):



    #Calulations###############################################

    # shape
    img_row, img_col, chan = org_img.shape
    # current no of col
    no_row =  int(img_row/513)
    #col required to divide without fraction
    req_row = ((no_row + 1)*513)-img_row

    
    #cropping required column portion
    ##########################################################3
    
    crop_img = org_img[(img_row-req_row):img_row,:,:]
    

    
    #horizontal flip array
    crop_img = np.flipud(crop_img)  

    return crop_img

def col_aug_add(org_img):



    #Calulations###############################################

    # shape
    img_row, img_col, chan = org_img.shape
    # current no of col
    no_col =  int(img_col/513)
    #col required to divide without fraction
    req_col = ((no_col + 1)*513)-img_col

    
    #cropping required column portion
    ##########################################################3
    
    crop_img = org_img[:,(img_col-req_col):img_col,:]
    
    
    #vertical flip array
    crop_img = np.fliplr(crop_img)  

    return crop_img