

from multiToBinMetrics import multiToBinImg
from multiToBinMetrics import confusionMatrix
from multiToBinMetrics import precision
from multiToBinMetrics import recall
from multiToBinMetrics import iou
from multiToBinMetrics import accuracy
from img_multi_to_bin import img_multi_to_bin


import csv

import numpy as np
from PIL import Image
from PIL import ImageFilter
import os
from shutil import rmtree
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# import pandas as pd
from numba import jit

@jit(nopython=True)
def plot_img(fname, orgImg, tarImg, predImg):

    ac = []
    io = []
    pr = []
    rc = []
    pe = []
    numClass = 6

    widhth = 7182
    height = 6669
    total_pixel = widhth*height
    for i in range(numClass):

        
        tarVal = img_multi_to_bin(tarImg,i)
        predVal = img_multi_to_bin(predImg,i)

        [tn, fp, fn, tp]=confusionMatrix(tarVal, predVal)


        ac_re = accuracy(tp, tn, fp, fn)
        io_re = iou(tarVal, predVal)
        pr_re = precision(tp, fp)
        rc_re = recall(tp, fn)


        class_pixel = sum(map(sum, tarVal))

        class_percentage = (class_pixel/total_pixel)*100


        if class_percentage == 0:
            io_re = 0
            pr_re = 0
            rc_re = 0


          

        ac.append(round(ac_re,2))
        io.append(round(io_re,2))
        pr.append(round(pr_re,2))
        rc.append(round(rc_re,2))
        pe.append(round(class_percentage,2))



    f, axarr = plt.subplots(1, 3, figsize=(12, 8))#6 for gaofen/ 8 for dhaka
    f.suptitle(fname, fontsize=14)

    axarr[0].imshow(orgImg)
    axarr[0].title.set_text('Input')

    axarr[1].imshow(tarImg)
    axarr[1].title.set_text('Ground Truth')

    axarr[2].imshow(predImg)
    axarr[2].title.set_text('Output')

    # acc_str = 'accuracy: ' + str(round(accuracy[i], 4))
    # iou_str = 'iou: ' + str(round(iou[i], 4))
    # rec_str = 'recall: ' + str(round(recall[i], 4))
    # pre_str = 'precision: ' + str(round(precision[i], 4))

    #io = np.nan_to_num(io)
    #io = np.where(io==0, 1, io)

    weight_acc = (pe[0]*ac[0] + pe[1]*ac[1] + pe[2]*ac[2] + pe[3]*ac[3] + pe[4]*ac[4] + pe[5]*ac[5]) / (pe[0] + pe[1] + pe[2] + pe[3] + pe[4] +pe[5] )
    weight_iou = (pe[0]*io[0] + pe[1]*io[1] + pe[2]*io[2] + pe[3]*io[3] + pe[4]*io[4] + pe[5]*io[5]) / (pe[0] + pe[1] + pe[2] + pe[3] + pe[4] +pe[5] )
    weight_rec = (pe[0]*pr[0] + pe[1]*pr[1] + pe[2]*pr[2] + pe[3]*pr[3] + pe[4]*pr[4] + pe[5]*pr[5]) / (pe[0] + pe[1] + pe[2] + pe[3] + pe[4] +pe[5] )
    weight_pre = (pe[0]*rc[0] + pe[1]*rc[1] + pe[2]*rc[2] + pe[3]*rc[3] + pe[4]*rc[4] + pe[5]*rc[5]) / (pe[0] + pe[1] + pe[2] + pe[3] + pe[4] +pe[5] )

    weight_acc = round(weight_acc,2)
    weight_iou = round(weight_iou,2)
    weight_rec = round(weight_rec,2)
    weight_pre = round(weight_pre,2)


    unrecognized = mpatches.Patch(color="#000000", label="Unrecognized")
    forest = mpatches.Patch(color='#00FFFF', label='Forest')
    builtUp = mpatches.Patch(color='#FF0000', label='BuiltUp')
    water = mpatches.Patch(color='#0000FF', label='Water')
    farmland = mpatches.Patch(color='#00FF00', label='Farmland')
    meadow = mpatches.Patch(color='#FFFF00', label='Meadow')
    f.legend(loc='upper right', fontsize='12', handles=[unrecognized, forest, builtUp, water, farmland, meadow])

    for i in range (len(ac)):
        if io[i] == 0:
            io[i] = '-'
        if pr[i] == 0:
            pr[i] = '-'
        if rc[i] == 0:
            rc[i] = '-'

    axarr[0].plot([], [],color='#FFFFFF', label="Metric\nPercentage :\nAccuracy :\nIOU :\nPrecision: \nRecall:")
    axarr[0].plot([], [],color='#FFFFFF', label="Weighted \n100%\n"+str(weight_acc)+"\n"+str(weight_iou)+"\n"+str(weight_rec)+"\n"+str(weight_pre))
    axarr[0].plot([], [],color='#000000', label="Unrecognized\n"+str(pe[0])+"%\n"+str(ac[0])+"\n"+str(io[0])+"\n"+str(pr[0])+"\n"+str(rc[0]))
    axarr[0].plot([], [],color='#00FFFF', label="Forest\n"+str(pe[1])+"%\n"+str(ac[1])+"\n"+str(io[1])+"\n"+str(pr[1])+"\n"+str(rc[1]))
    axarr[0].plot([], [],color='#FF0000', label="BuiltUp\n"+str(pe[2])+"%\n"+str(ac[2])+"\n"+str(io[2])+"\n"+str(pr[2])+"\n"+str(rc[2]))
    axarr[0].plot([], [],color='#0000FF', label="Water\n"+str(pe[3])+"%\n"+str(ac[3])+"\n"+str(io[3])+"\n"+str(pr[3])+"\n"+str(rc[3]))
    axarr[0].plot([], [],color='#00FF00', label="Farmland\n"+str(pe[4])+"%\n"+str(ac[4])+"\n"+str(io[4])+"\n"+str(pr[4])+"\n"+str(rc[4]))
    axarr[0].plot([], [],color='#FFFF00', label="Meadow\n"+str(pe[5])+"%\n"+str(ac[5])+"\n"+str(io[5])+"\n"+str(pr[5])+"\n"+str(rc[5]))



    f.legend(loc='lower center', bbox_to_anchor=(0.485, 0.00), shadow=False, ncol=10, fontsize='12')
    plt.savefig('result/'+fname, dpi=1000)

    






