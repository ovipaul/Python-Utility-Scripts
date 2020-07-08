import matplotlib.pyplot as plt
import glob, os
from im2index import im2index
import numpy as np
from scipy.misc import toimage
from PIL import Image


for infile in glob.glob("*.png"):
    file, ext = os.path.splitext(infile)
    img = Image.open(infile)
    im = np.asarray(img)
    print(im.shape)
    print(type(im))
    img = im2index(im)
    toimage(img).save(file+ext)
