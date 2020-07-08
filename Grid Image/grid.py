from ImageSlicer import ImageSlicer
from PIL import Image
import glob
import os




for image in glob.glob('*.png'):
    if image.endswith('.png'):
        img_name = str(image).rstrip('.png')
        slicer = ImageSlicer(image, (513,513)) #Provide image path and slice size you desire
        transformed_image = slicer.transform()
        slicer.save_images(transformed_image,'splittedFolder',img_name)#splittedFolder is folder name where images will be splitted
