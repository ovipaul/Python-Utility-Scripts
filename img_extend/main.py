from PIL import Image
import numpy as np
import gc
import os

#txt_loc = '/home/akmmrahman/doombringer/pytorch-deeplab-xception/dataset/VOC2012/ImageSets/Segmentation/'


current_loc = os.getcwd()
txt_folder= 'readfolder'

txt_filename = 'img.txt'
txt_loc = current_loc+'/'+txt_folder+'/'


list = open(txt_loc+txt_filename,'r')
img_list = list.readlines()

#org_loc ='original/'  
tar_loc ='target/'
#pred_loc ='prediction/'

ext ='.png'
for fname in img_list:

    fname = fname.rstrip('\n')

    org_file = tar_loc + fname + ext
    org_file = Image.open(org_file)

    org_file = np.array(org_file)
    
    #Calulations###############################################

    # shape
    img_col, img_row, channel = org_file.shape
    # current no of col
    no_col =  int(img_col/513)
    #col required to divide without fraction
    req_col = ((no_col + 1)*513)-img_col
    print(req_col)
    #total required size
    final_col = img_col+req_col
    
    print(org_file.shape)
    #cropping required column portion
    ##########################################################3
    
    crop_img = org_file[(img_col-req_col):img_col,:,:]
    
    print(crop_img.shape)
    org_file = Image.fromarray( org_file)
    org_file.save( 'target_result/hor3.png')
    #horizontal flip array
    crop_img = np.fliplr(crop_img)

     
