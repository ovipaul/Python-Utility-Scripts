from PIL import Image
from plot_img import plot_img
import gc
#txt_loc = '/home/akmmrahman/doombringer/pytorch-deeplab-xception/dataset/VOC2012/ImageSets/Segmentation/'
txt_loc='/home/paul/for_science/research/processing/plot/'
list = open(txt_loc+'val.txt','r')
img_list = list.readlines()

org_loc ='original/'  
tar_loc ='target/'
pred_loc ='prediction/'

for fname in img_list:

    fname = fname.rstrip('\n')
    org_file = org_loc + fname + '.png'
    tar_file = tar_loc + fname + '.png'
    pred_file = pred_loc + fname + '.png'

    org_img = Image.open(org_file)
    tar_img = Image.open(tar_file)
    pred_img = Image.open(pred_file)

    plot_img(fname, org_img, tar_img, pred_img)

    gc.collect()


