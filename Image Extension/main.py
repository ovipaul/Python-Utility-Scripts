from PIL import Image
import numpy as np
import gc
import os
from aug_add import row_aug_add
from aug_add import col_aug_add


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

    org_img = tar_loc + fname + ext
    org_img = Image.open(org_img)

    org_img = np.array(org_img)

    row_img = row_aug_add(org_img)
    row_img = np.concatenate((org_img, row_img))

    col_img = col_aug_add(row_img)
    final_img = np.concatenate((row_img, col_img), axis=1)

    final_img = Image.fromarray( final_img)
    #Adding on bottom#########################################

    final_img.save( 'target_result/hor35.png')
     
