

from PIL import Image
import glob
import numpy as np
dir = 'img_folder'

for filename in glob.glob(dir+'/*.png'): #assuming gif
    img=Image.open(filename)
    img = np.array(img)
    img = img[:,:,0:3]
    row, col, cha = img.shape


    img_percentage = 20
    col_range = int((col/100)*img_percentage)
    col_iteration = int(col/col_range)
    col_initial = 0
     
    col_aug_ext = img[:,col_initial:col_range,:] 
    col_aug_dup = img[:,col_initial:col_range,:]
    
    col_aug = np.concatenate((col_aug_ext,col_aug_dup), axis=1)

    col_in = col_initial
    col_fin = col_range
    col_img_list =[]
    for i in range(col_iteration):
        print(col_in,col_fin)
        
        col_aug = img[:,col_in:col_fin,:]
        col_img_list.append(col_aug)
        #print(col_in,col_fin)
        #final_aug = np.concatenate((final_aug,col_aug), axis=1)

        # frow , fcol, fcha = final_aug.shape
        col_in = col_fin
        col_fin = col_fin+col_range

        #col_aug = img[:,col_initial:col_range,:]
    
    for i in range(col_iteration):


        final_aug = np.concatenate((col_img_list[i],col_img_list[i]), axis=1)


    print(final_aug.shape)
 
        
        

