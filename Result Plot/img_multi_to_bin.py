import numpy as np
from PIL import Image
from tqdm import tqdm
from numba import jit

@jit(nopython=True)
def img_multi_to_bin(src, ind):
    src = np.asarray(src)
    row, col, cha = src.shape
    proc_img = np.zeros((row, col))
    # 0=Unrecognized
    if ind == 0:

        for i in tqdm(range(row)):
            for j in range(col):
                if src[i, j, 0] == 0 and src[i, j, 1] == 0 and src[i, j, 2] == 0:
                    proc_img[i, j] = 1

                else:
                    proc_img[i, j] = 0
        return proc_img
    # 1=Forest
    elif ind == 1:


        for i in tqdm(range(row)):
            for j in range(col):
                if src[i, j, 0] == 0 and src[i, j, 1] == 255 and src[i, j, 2] == 255:
                    proc_img[i, j] = 1

                else:
                    proc_img[i, j] = 0
        return proc_img
    # 2=BUilt-Up
    elif ind == 2:

        for i in tqdm(range(row)):
            for j in range(col):
                if src[i, j, 0] == 255 and src[i, j, 1] == 0 and src[i, j, 2] == 0:
                    proc_img[i, j] = 1

                else:
                    proc_img[i, j] = 0
        return proc_img
    # 3=Water
    elif ind == 3:

        for i in tqdm(range(row)):
            for j in range(col):
                if src[i, j, 0] == 0 and src[i, j, 1] == 0 and src[i, j, 2] == 255:
                    proc_img[i, j] = 1

                else:
                    proc_img[i, j] = 0
        return proc_img
    #4=Farmland
    elif ind == 4:

        for i in tqdm(range(row)):
            for j in range(col):
                if src[i, j, 0] == 0 and src[i, j, 1] == 255 and src[i, j, 2] == 0:
                    proc_img[i, j] = 1

                else:
                    proc_img[i, j] = 0
        return proc_img
    #5=Meadow
    elif ind == 5:

        for i in tqdm(range(row)):
            for j in range(col):
                if src[i, j, 0] == 255 and src[i, j, 1] == 255 and src[i, j, 2] == 0:
                    proc_img[i, j] = 1

                else:
                    proc_img[i, j] = 0
        return proc_img
    else:
        print("No class index found")
