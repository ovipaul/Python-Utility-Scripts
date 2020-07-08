from PIL import Image
import glob
import numpy as np
from overlap_grid import overlap_grid

dir = 'original'
output_dir = 'JPEGImages'
init_exclude = len(dir)+1
overlap_percentage = 100
for filename in glob.glob(dir+'/*.png'): 

    overlap_grid(filename,overlap_percentage,output_dir, init_exclude)
